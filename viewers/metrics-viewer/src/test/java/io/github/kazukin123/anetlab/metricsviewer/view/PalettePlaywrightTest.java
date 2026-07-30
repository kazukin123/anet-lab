package io.github.kazukin123.anetlab.metricsviewer.view;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.server.LocalServerPort;

import com.microsoft.playwright.Browser;
import com.microsoft.playwright.BrowserContext;
import com.microsoft.playwright.BrowserType;
import com.microsoft.playwright.CDPSession;
import com.microsoft.playwright.Page;
import com.microsoft.playwright.Playwright;
import com.microsoft.playwright.PlaywrightException;
import com.microsoft.playwright.Route;
import com.microsoft.playwright.options.WaitUntilState;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.fasterxml.jackson.databind.JsonNode;

@SpringBootTest(
		webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT,
		properties = "metricsviewer.runs-dir=target/playwright-test-empty-runs")
class PalettePlaywrightTest {

	private static final String TAG_KEY = "palette/test";
	private static final String TAG_A = "tag/a";
	private static final String TAG_B = "tag/b";
	private static final String TAG_C = "tag/c";
	private static final String RELOAD_RUN = "run_reload";
	private static final String RELOAD_OLD_TAG = "reload/old";
	private static final String RELOAD_NEW_TAG = "reload/new";
	private static final String SIGNED_LOG_TAG = "signed/log";
	private static final String GENERATION = "00000000-0000-0000-0000-000000000001";
	private static final Path TEST_RUNS_DIR = Path.of("target/playwright-test-empty-runs");

	private static final List<String> RUN_COLORS = List.of(
			"#2F7DE1", "#F2C230", "#7A5CFF", "#008B8B", "#F05A28",
			"#C678DD", "#FF9F1C", "#00A99D", "#A3C720", "#E23B4F",
			"#2FBF71", "#E75A9B", "#D1D83B", "#B83280", "#00B36B",
			"#A6761D", "#4656D9", "#C85A17", "#D83BD2", "#A65A2E");

	private static final List<String> RUN_IDS = List.of(
			"run_01", "run_02", "run_03", "run_04", "run_05", "run_06",
			"run_07", "run_08", "run_09", "run_10", "run_11");

	@LocalServerPort
	private int port;

	static {
		try {
			Files.createDirectories(TEST_RUNS_DIR);
		} catch (IOException e) {
			throw new ExceptionInInitializerError(e);
		}
	}

	@Test
	void paletteColorsRenderForRunChipsAndInitialTrace() throws Exception {
		final String baseUrl = "http://127.0.0.1:" + port;
		assertServedPaletteScript(baseUrl);

		assumeTrue(isMicrosoftEdgeInstalled(), "Microsoft Edge is not installed.");

		try (Playwright playwright = Playwright.create()) {
			final Browser browser = launchMicrosoftEdge(playwright);
			try {
				final BrowserContext context = browser.newContext(new Browser.NewContextOptions()
						.setViewportSize(1280, 720));
				try {
					final Page page = context.newPage();
					page.route("**/api/runs.json", route -> fulfillJson(route, runsJson()));
					page.route("**/api/metrics.json", route -> fulfillJson(route, metricsJson()));

					page.navigate(baseUrl + "/?paletteTest=" + System.nanoTime(),
							new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
					page.waitForFunction("document.querySelectorAll('#run-list .run-color').length === 11",
							null, new Page.WaitForFunctionOptions().setTimeout(30000));
					page.waitForFunction("document.querySelectorAll('.js-plotly-plot path.js-line').length > 0",
							null, new Page.WaitForFunctionOptions().setTimeout(30000));

					final List<Map<String, String>> chips = readRunChips(page);
					assertEquals(RUN_IDS.size(), chips.size());
					assertEquals(expectedDisplayedChips(), chips);

					final String selectedLineColor = readFirstLineStroke(page);
					assertEquals(hexToRgb(RUN_COLORS.get(RUN_IDS.size() - 1)), selectedLineColor);
					assertEquals(chips.get(0).get("color"), selectedLineColor);
				} finally {
					context.close();
				}
			} finally {
				browser.close();
			}
		}
	}

	@Test
	void plotWidthDoesNotCreateHorizontalMainAreaOverflow() throws Exception {
		final String baseUrl = "http://127.0.0.1:" + port;
		assumeTrue(isMicrosoftEdgeInstalled(), "Microsoft Edge is not installed.");

		try (Playwright playwright = Playwright.create()) {
			final Browser browser = launchMicrosoftEdge(playwright);
			try {
				final BrowserContext context = browser.newContext(new Browser.NewContextOptions()
						.setViewportSize(1280, 720));
				try {
					final Page page = context.newPage();
					page.route("**/api/runs.json", route -> fulfillJson(route, runsJson()));
					page.route("**/api/metrics.json", route -> fulfillJson(route, metricsJson()));

					page.navigate(baseUrl + "/?horizontalOverflowTest=" + System.nanoTime(),
							new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
					page.waitForFunction("document.querySelector('.js-plotly-plot .main-svg') !== null",
							null, new Page.WaitForFunctionOptions().setTimeout(30000));

					@SuppressWarnings("unchecked")
					final Map<String, Number> widths = (Map<String, Number>) page.evaluate("""
							() => {
								const main = document.getElementById('main-area');
								const graph = document.querySelector('.graph-block');
								const svg = document.querySelector('.js-plotly-plot .main-svg');
								if (!main || !graph || !svg) throw new Error('graph layout not ready');
								return {
									mainClient: main.clientWidth,
									mainScroll: main.scrollWidth,
									graphWidth: graph.getBoundingClientRect().width,
									svgWidth: svg.getBoundingClientRect().width
								};
							}
							""");
					assertTrue(
							widths.get("mainScroll").doubleValue()
									<= widths.get("mainClient").doubleValue(),
							"Plotly must not create horizontal main-area overflow: " + widths);
					assertTrue(
							widths.get("svgWidth").doubleValue()
									<= widths.get("graphWidth").doubleValue() + 1.0,
							"Plotly SVG must fit its graph block: " + widths);
				} finally {
					context.close();
				}
			} finally {
				browser.close();
			}
		}
	}

	@Test
	void runRowsToggleImmediatelyAllowEmptySelectionAndSoloOnTheSecondClick() {
		final String baseUrl = "http://127.0.0.1:" + port;
		assumeTrue(isMicrosoftEdgeInstalled(), "Microsoft Edge is not installed.");

		try (Playwright playwright = Playwright.create()) {
			final Browser browser = launchMicrosoftEdge(playwright);
			try {
				final BrowserContext context = browser.newContext(new Browser.NewContextOptions()
						.setViewportSize(1280, 720));
				try {
					final Page page = context.newPage();
					page.route("**/api/runs.json", route -> fulfillJson(route, runsJson()));
					page.route("**/api/metrics.json", route -> fulfillJson(route, metricsJson()));
					page.route("**/api/runs/prioritize", PalettePlaywrightTest::fulfillNoContent);

					page.navigate(baseUrl + "/?runToggleTest=" + System.nanoTime(),
							new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
					page.waitForFunction("document.querySelectorAll('#run-list .run-row').length === 11");
					assertEquals(List.of("run_11"), readActiveRunIds(page));

					page.click("#run-list .run-row.active");
					page.waitForFunction("document.querySelectorAll('#run-list .run-row.active').length === 0");
					assertEquals("No selection.", page.textContent("#main-area"));

					page.click("#run-list .run-row[data-run-id='run_03']");
					page.click("#run-list .run-row[data-run-id='run_04']");
					assertEquals(List.of("run_04", "run_03"), readActiveRunIds(page));

					page.evaluate("""
							runId => {
								const find = () => document.querySelector(
									`#run-list .run-row[data-run-id="${runId}"]`);
								find().click();
								find().click();
							}
							""", "run_03");
					assertEquals(List.of("run_03"), readActiveRunIds(page));
				} finally {
					context.close();
				}
			} finally {
				browser.close();
			}
		}
	}

	@Test
	void lodDisplayModesReuseOneProjectionAndPersistTheSelection() {
		final String baseUrl = "http://127.0.0.1:" + port;
		assumeTrue(isMicrosoftEdgeInstalled(), "Microsoft Edge is not installed.");

		try (Playwright playwright = Playwright.create()) {
			final Browser browser = launchMicrosoftEdge(playwright);
			try {
				final BrowserContext context = browser.newContext(new Browser.NewContextOptions()
						.setViewportSize(1280, 720));
				try {
					final Page page = context.newPage();
					final AtomicInteger metricsRequests = new AtomicInteger();
					page.route("**/api/runs.json", route -> fulfillJson(
							route,
							"{\"runs\":[" + runJson("run_lod_ui", 31, TAG_KEY) + "]}"));
					page.route("**/api/metrics.json", route -> {
						metricsRequests.incrementAndGet();
						fulfillJson(route, lodMetricsJson());
					});
					page.route("**/api/runs/prioritize", PalettePlaywrightTest::fulfillNoContent);

					page.navigate(baseUrl + "/?lodModeTest=" + System.nanoTime(),
							new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
					page.waitForFunction("""
							() => {
								const plot = document.querySelector('.js-plotly-plot');
								return plot?.data?.length === 1
									&& JSON.stringify(Array.from(plot.data[0].x))
										=== JSON.stringify([2,8,15,18,25,31]);
							}
							""");
					assertEquals(1, metricsRequests.get());

					page.selectOption("#lod-display-mode", "Mean");
					page.waitForFunction("""
							() => {
								const plot = document.querySelector('.js-plotly-plot');
								return plot?.data?.length === 1
									&& JSON.stringify(Array.from(plot.data[0].y))
										=== JSON.stringify([3,4]);
							}
							""");
					assertEquals(1, metricsRequests.get());

					page.selectOption("#lod-display-mode", "Band");
					page.waitForFunction("""
							() => {
								const plot = document.querySelector('.js-plotly-plot');
								return plot?.data?.length === 3
									&& plot.data[1].fill === 'tonexty'
									&& plot.data[1].fillcolor.endsWith(',0.28)');
							}
							""");
					assertEquals(1, metricsRequests.get());

					page.reload(new Page.ReloadOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
					page.waitForFunction("document.querySelector('#lod-display-mode')?.value === 'Band'");
					page.waitForFunction("document.querySelector('.js-plotly-plot')?.data?.length === 3");
					assertEquals(2, metricsRequests.get());
				} finally {
					context.close();
				}
			} finally {
				browser.close();
			}
		}
	}

	@Test
	void graphHeaderCombinesTagStatsAndShowsRunAndTagIssues() {
		final String baseUrl = "http://127.0.0.1:" + port;
		assumeTrue(isMicrosoftEdgeInstalled(), "Microsoft Edge is not installed.");

		try (Playwright playwright = Playwright.create()) {
			final Browser browser = launchMicrosoftEdge(playwright);
			try {
				final BrowserContext context = browser.newContext(new Browser.NewContextOptions()
						.setViewportSize(1280, 720));
				try {
					final Page page = context.newPage();
					page.route("**/api/runs.json", route -> fulfillJson(route, statsWarningRunsJson()));
					page.route("**/api/metrics.json", route -> fulfillJson(route, statsWarningMetricsJson()));
					page.route("**/api/runs/prioritize", PalettePlaywrightTest::fulfillNoContent);

					page.navigate(baseUrl + "/?statsWarningTest=" + System.nanoTime(),
							new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
					page.click("#btn-select-all-runs");
					waitForMultiRunGraph(page);
					page.waitForFunction("document.querySelector('.graph-warning') !== null");

					assertEquals(
							"Min -1 / Max 11 / Avg 5 / Std 5.09902",
							page.textContent(".graph-stats"));
					assertTrue(page.getAttribute(".graph-stats", "title").contains("count=4"));
					assertTrue(page.getAttribute(".graph-warning", "title").contains("invalid_json"));
					assertTrue(page.getAttribute(".graph-warning", "title").contains("tag_step_regression"));
					assertEquals("⚠1", page.textContent(
							"#run-list .run-row[data-run-id='run_stats_b'] .run-warning"));
					assertTrue(page.textContent("#tag-list li[data-tag-key='" + TAG_KEY + "']")
							.contains("⚠"));
				} finally {
					context.close();
				}
			} finally {
				browser.close();
			}
		}
	}

	@Test
	void disappearingRunIsRemovedFromSelectionWindowsAndColorCache() {
		final String baseUrl = "http://127.0.0.1:" + port;
		assumeTrue(isMicrosoftEdgeInstalled(), "Microsoft Edge is not installed.");

		try (Playwright playwright = Playwright.create()) {
			final Browser browser = launchMicrosoftEdge(playwright);
			try {
				final BrowserContext context = browser.newContext(new Browser.NewContextOptions()
						.setViewportSize(1280, 720));
				try {
					final Page page = context.newPage();
					final AtomicInteger runsRequests = new AtomicInteger();
					page.route("**/api/runs.json", route -> fulfillJson(
							route,
							runsRequests.incrementAndGet() == 1
									? "{\"runs\":[" + runJson("run_vanish", 2, TAG_KEY) + "]}"
									: "{\"runs\":[]}"));
					page.route("**/api/metrics.json", route -> fulfillJson(
							route,
							"{\"data\":["
									+ rawSeriesJson(
											"run_vanish",
											TAG_KEY,
											new double[] {0, 1, 2},
											new float[] {1, 2, 3})
									+ "]}"));
					page.route("**/api/runs/prioritize", PalettePlaywrightTest::fulfillNoContent);

					page.navigate(baseUrl + "/?runDisappearanceTest=" + System.nanoTime(),
							new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
					waitForGraph(page);
					page.click("#btn-reload");
					page.waitForFunction("""
							() => document.querySelectorAll('#run-list .run-row').length === 0
								&& document.getElementById('main-area')?.textContent === 'No selection.'
							""");
					assertTrue(Boolean.TRUE.equals(page.evaluate("""
							() => app.cache.getRun('run_vanish') === null
								&& app.cache.getWindow('run_vanish', 'palette/test') === null
								&& !app.runColorMap.has('run_vanish')
							""")));
				} finally {
					context.close();
				}
			} finally {
				browser.close();
			}
		}
	}

	@Test
	void convertingRunPollOnlyUpdatesExistingPercentageUntilReload() {
		final String baseUrl = "http://127.0.0.1:" + port;
		assumeTrue(isMicrosoftEdgeInstalled(), "Microsoft Edge is not installed.");
		final String nextGeneration = "00000000-0000-0000-0000-000000000002";
		final String newTag = "poll/new-tag";

		try (Playwright playwright = Playwright.create()) {
			final Browser browser = launchMicrosoftEdge(playwright);
			try {
				final BrowserContext context = browser.newContext(new Browser.NewContextOptions()
						.setViewportSize(1280, 720));
				try {
					final Page page = context.newPage();
					final AtomicInteger runsRequests = new AtomicInteger();
					final AtomicInteger metricsRequests = new AtomicInteger();
					final AtomicBoolean reloaded = new AtomicBoolean();
					page.route("**/api/runs.json", route -> {
						final int index = runsRequests.incrementAndGet();
						final String pollRun = index == 1
								? runJson("run_poll", 2, GENERATION, "converting", 10, TAG_KEY)
								: runJson(
										"run_poll",
										8,
										nextGeneration,
										index == 2 ? "converting" : "ready",
										index == 2 ? 50 : 100,
										TAG_KEY,
										newTag);
						fulfillJson(
								route,
								"{\"runs\":["
										+ pollRun
										+ (index == 1
												? ""
												: "," + runJson(
														"run_new",
														8,
														nextGeneration,
														"ready",
														100,
														TAG_KEY))
										+ "]}");
					});
					page.route("**/api/metrics.json", route -> {
						metricsRequests.incrementAndGet();
						final String generation = reloaded.get() ? nextGeneration : GENERATION;
						final String extraSeries = reloaded.get()
								? "," + rawSeriesJson(
										"run_poll",
										newTag,
										generation,
										new double[] {0, 1, 2},
										new float[] {4, 5, 6})
								: "";
						fulfillJson(route, "{\"data\":["
								+ rawSeriesJson(
										"run_poll",
										TAG_KEY,
										generation,
										new double[] {0, 1, 2},
										new float[] {1, 2, 3})
								+ extraSeries
								+ "]}");
					});
					page.route("**/api/runs/prioritize", PalettePlaywrightTest::fulfillNoContent);

					page.navigate(baseUrl + "/?ingestPollTest=" + System.nanoTime(),
							new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
					waitForGraph(page);
					page.evaluate("""
							() => {
								const plot = document.querySelector('.js-plotly-plot');
								window.__pollPlot = plot;
								window.__pollRunRow = document.querySelector(
									"#run-list .run-row[data-run-id='run_poll']");
								window.__pollRunName = document.querySelector(
									"#run-list .run-row[data-run-id='run_poll'] .run-name").textContent;
								window.__pollTagList = document.getElementById('tag-list').textContent;
								window.__pollHeader = document.querySelector('.graph-header').textContent;
								window.__pollGeneration = app.cache.getRun('run_poll').generation;
								window.__pollMainScrollTop = document.getElementById('main-area').scrollTop;
								window.__pollXRange = JSON.stringify(plot.layout.xaxis.range ?? null);
								window.__pollYRange = JSON.stringify(plot.layout.yaxis.range ?? null);
							}
							""");
					final int initialMetricsRequests = metricsRequests.get();
					page.waitForFunction(
							"document.querySelector(\"#run-list .run-row[data-run-id='run_poll'] .run-progress\")"
									+ "?.textContent === '50%'",
							null,
							new Page.WaitForFunctionOptions().setTimeout(7000));

					assertEquals(initialMetricsRequests, metricsRequests.get());
					assertEquals(1, page.locator("#run-list .run-row").count());
					assertEquals(1, page.locator("#tag-list li").count());
					assertTrue(Boolean.TRUE.equals(page.evaluate("""
							() => window.__pollPlot === document.querySelector('.js-plotly-plot')
								&& window.__pollRunRow === document.querySelector(
									"#run-list .run-row[data-run-id='run_poll']")
								&& window.__pollRunName === document.querySelector(
									"#run-list .run-row[data-run-id='run_poll'] .run-name").textContent
								&& window.__pollTagList === document.getElementById('tag-list').textContent
								&& window.__pollHeader === document.querySelector('.graph-header').textContent
								&& window.__pollGeneration === app.cache.getRun('run_poll').generation
								&& app.cache.getRunIds().join(',') === 'run_poll'
								&& app.cache.getTag('run_poll', 'palette/test').stats.count === 3
								&& window.__pollMainScrollTop === document.getElementById('main-area').scrollTop
								&& window.__pollXRange === JSON.stringify(
									document.querySelector('.js-plotly-plot').layout.xaxis.range ?? null)
								&& window.__pollYRange === JSON.stringify(
									document.querySelector('.js-plotly-plot').layout.yaxis.range ?? null)
							""")));

					assertEquals(
							"rgb(36, 87, 125)",
							page.evaluate("""
									() => getComputedStyle(
										document.querySelector(
											"#run-list .run-row[data-run-id='run_poll']"),
										'::before').backgroundColor
									"""));
					assertEquals(
							"rgba(190, 190, 190, 0.24)",
							page.evaluate("""
									() => {
										const row = document.querySelector(
											"#run-list .run-row[data-run-id='run_poll']");
										row.classList.remove('active');
										const color = getComputedStyle(row, '::before').backgroundColor;
										row.classList.add('active');
										return color;
									}
									"""));
					assertEquals("rgb(36, 87, 125)", page.evaluate("""
							() => getComputedStyle(document.querySelector('.run-row.active')).backgroundColor
							"""));
					assertEquals("rgb(36, 87, 125)", page.evaluate("""
							() => getComputedStyle(document.querySelector('#tag-list li.active')).backgroundColor
							"""));
					assertEquals("rgb(196, 196, 196)", page.evaluate("""
							() => getComputedStyle(document.querySelector('.run-progress')).color
							"""));
					assertEquals(
							"rgb(36, 87, 125)|rgb(36, 87, 125)|rgb(36, 87, 125)"
									+ "|rgb(36, 87, 125)|rgb(73, 201, 255)|rgb(243, 250, 255)",
							page.evaluate("""
									() => {
										const autoReload = document.getElementById('btn-auto-reload');
										const scrollLock = document.getElementById('btn-graph-scroll-lock');
										const log = document.querySelector('.graph-log-toggle');
										const checkbox = document.getElementById('chk-lock-tags');
										autoReload.classList.add('active');
										scrollLock.classList.add('active');
										log.classList.add('active');
										checkbox.checked = true;
										return [
											getComputedStyle(autoReload).backgroundColor,
											getComputedStyle(scrollLock).backgroundColor,
											getComputedStyle(log).backgroundColor,
											getComputedStyle(checkbox).backgroundColor,
											getComputedStyle(checkbox).borderColor,
											getComputedStyle(autoReload).color
										].join('|');
									}
									"""));

					page.waitForFunction(
							"document.querySelector(\"#run-list .run-row[data-run-id='run_poll'] .run-progress\")"
									+ " === null && app.ingestPollTimer === null",
							null,
							new Page.WaitForFunctionOptions().setTimeout(7000));
					assertTrue(runsRequests.get() >= 3);
					assertEquals(initialMetricsRequests, metricsRequests.get());
					assertEquals(0, page.locator(
							"#run-list .run-row[data-run-id='run_poll'] .run-progress").count());
					assertEquals("0%", page.evaluate("""
							() => document.querySelector(
								"#run-list .run-row[data-run-id='run_poll']")
								.style.getPropertyValue('--ingest-progress')
							"""));

					reloaded.set(true);
					page.click("#btn-reload");
					page.waitForFunction(
							"document.querySelectorAll('#run-list .run-row').length === 2"
									+ " && document.querySelectorAll('#tag-list li').length === 2"
									+ " && app.cache.getRun('run_poll').generation"
									+ " === '00000000-0000-0000-0000-000000000002'"
									+ " && app.cache.getTag('run_poll', 'palette/test').stats.count === 9");
					assertTrue(metricsRequests.get() > initialMetricsRequests);
				} finally {
					context.close();
				}
			} finally {
				browser.close();
			}
		}
	}

	@Test
	void zoomUsesTheCoarsePlotImmediatelyThenRequestsOneThreeViewportWindow() throws Exception {
		final String baseUrl = "http://127.0.0.1:" + port;
		assumeTrue(isMicrosoftEdgeInstalled(), "Microsoft Edge is not installed.");

		try (Playwright playwright = Playwright.create()) {
			final Browser browser = launchMicrosoftEdge(playwright);
			try {
				final BrowserContext context = browser.newContext(new Browser.NewContextOptions()
						.setViewportSize(1280, 720));
				try {
					final Page page = context.newPage();
					final List<String> requests = Collections.synchronizedList(new ArrayList<>());
					page.route("**/api/runs.json", route -> fulfillJson(
							route,
							"{\"runs\":[" + runJson("run_lod_ui", 31, TAG_KEY) + "]}"));
					page.route("**/api/metrics.json", route -> {
						requests.add(route.request().postData());
						fulfillJson(route, lodMetricsJson());
					});
					page.route("**/api/runs/prioritize", PalettePlaywrightTest::fulfillNoContent);

					page.navigate(baseUrl + "/?zoomRefinementTest=" + System.nanoTime(),
							new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
					waitForGraph(page);
					page.evaluate("""
							() => Plotly.relayout(
								document.querySelector('.js-plotly-plot'),
								{'xaxis.range': [8, 12]})
							""");
					page.waitForTimeout(700);

					assertTrue(requests.size() >= 2);
					final JsonNode second = new com.fasterxml.jackson.databind.ObjectMapper()
							.readTree(requests.get(1))
							.path("series")
							.get(0);
					assertEquals(4, second.path("fromStep").asLong());
					assertEquals(16, second.path("toStep").asLong());
				} finally {
					context.close();
				}
			} finally {
				browser.close();
			}
		}
	}

	@Test
	void staleGenerationResponseIsDiscardedUntilMetadataAndProjectionMatch() {
		final String baseUrl = "http://127.0.0.1:" + port;
		final String nextGeneration = "00000000-0000-0000-0000-000000000002";
		assumeTrue(isMicrosoftEdgeInstalled(), "Microsoft Edge is not installed.");

		try (Playwright playwright = Playwright.create()) {
			final Browser browser = launchMicrosoftEdge(playwright);
			try {
				final BrowserContext context = browser.newContext(new Browser.NewContextOptions()
						.setViewportSize(1280, 720));
				try {
					final Page page = context.newPage();
					final AtomicInteger runsRequests = new AtomicInteger();
					page.route("**/api/runs.json", route -> {
						final String generation = runsRequests.incrementAndGet() == 1
								? GENERATION
								: nextGeneration;
						fulfillJson(
								route,
								"{\"runs\":["
										+ runJsonWithGeneration(
												"run_generation",
												2,
												generation,
												TAG_KEY)
										+ "]}");
					});
					page.route("**/api/metrics.json", route -> fulfillJson(
							route,
							"{\"data\":["
									+ rawSeriesJson(
											"run_generation",
											TAG_KEY,
											nextGeneration,
											new double[] {0, 1, 2},
											new float[] {1, 2, 3})
									+ "]}"));
					page.route("**/api/runs/prioritize", PalettePlaywrightTest::fulfillNoContent);

					page.navigate(baseUrl + "/?staleGenerationTest=" + System.nanoTime(),
							new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
					page.waitForFunction("""
							() => app?.mode === 'normal'
								&& !document.getElementById('loading-spinner')?.classList.contains('active')
							""");
					assertTrue(Boolean.TRUE.equals(page.evaluate("""
							() => app.cache.getWindow('run_generation', 'palette/test') === null
								&& document.querySelector('.js-plotly-plot') === null
							""")));

					page.click("#btn-reload");
					waitForGraph(page);
					assertTrue(Boolean.TRUE.equals(page.evaluate("""
							() => app.cache.getWindow('run_generation', 'palette/test')?.generation
								=== '00000000-0000-0000-0000-000000000002'
							""")));
				} finally {
					context.close();
				}
			} finally {
				browser.close();
			}
		}
	}

	@Test
	void plotlyPngDownloadWorksForSlashTagKeys() {
		final String baseUrl = "http://127.0.0.1:" + port;

		assumeTrue(isMicrosoftEdgeInstalled(), "Microsoft Edge is not installed.");

		try (Playwright playwright = Playwright.create()) {
			final Browser browser = launchMicrosoftEdge(playwright);
			try {
				final BrowserContext context = browser.newContext(new Browser.NewContextOptions()
						.setViewportSize(1280, 720));
				try {
					final Page page = context.newPage();
					page.route("**/api/runs.json", route -> fulfillJson(route, runsJson()));
					page.route("**/api/metrics.json", route -> fulfillJson(route, metricsJson()));

					page.navigate(baseUrl + "/?plotlyPngDownloadTest=" + System.nanoTime(),
							new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
					waitForGraph(page);

					final String traceUid = readFirstTraceUid(page);
					assertTrue(traceUid.startsWith("mv_"));
					assertFalse(traceUid.contains("/"));

					final String imagePrefix = renderFirstPlotPngPrefix(page);
					assertEquals("data:image/png;base64,", imagePrefix);
				} finally {
					context.close();
				}
			} finally {
				browser.close();
			}
		}
	}

	@Test
	void plotlyModeBarShowsAutoscaleAndResetAxes() {
		final String baseUrl = "http://127.0.0.1:" + port;

		assumeTrue(isMicrosoftEdgeInstalled(), "Microsoft Edge is not installed.");

		try (Playwright playwright = Playwright.create()) {
			final Browser browser = launchMicrosoftEdge(playwright);
			try {
				final BrowserContext context = browser.newContext(new Browser.NewContextOptions()
						.setViewportSize(1280, 720));
				try {
					final Page page = context.newPage();
					page.route("**/api/runs.json", route -> fulfillJson(route, runsJson()));
					page.route("**/api/metrics.json", route -> fulfillJson(route, metricsJson()));

					page.navigate(baseUrl + "/?plotlyModeBarTest=" + System.nanoTime(),
							new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
					waitForGraph(page);

					final List<String> buttonTitles = readModeBarButtonTitles(page);
					assertTrue(buttonTitles.contains("Autoscale"));
					assertTrue(buttonTitles.contains("Reset axes"));
				} finally {
					context.close();
				}
			} finally {
				browser.close();
			}
		}
	}

	@Test
	void graphScrollLockDefaultsOffTogglesPersistsAndHidesInScreenshotMode() {
		final String baseUrl = "http://127.0.0.1:" + port;

		assumeTrue(isMicrosoftEdgeInstalled(), "Microsoft Edge is not installed.");

		try (Playwright playwright = Playwright.create()) {
			final Browser browser = launchMicrosoftEdge(playwright);
			try {
				final BrowserContext context = browser.newContext(new Browser.NewContextOptions()
						.setViewportSize(1280, 720));
				try {
					final Page page = context.newPage();
					page.route("**/api/runs.json", route -> fulfillJson(route, runsJson()));
					page.route("**/api/metrics.json", route -> fulfillJson(route, metricsJson()));

					page.navigate(baseUrl + "/?graphScrollLockTest=" + System.nanoTime(),
							new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
					waitForGraph(page);

					assertEquals("Scroll Lock: OFF", page.textContent("#btn-graph-scroll-lock"));
					assertEquals("false", page.getAttribute("#btn-graph-scroll-lock", "aria-pressed"));
					assertFalse(isGraphScrollLockButtonActive(page));
					assertTrue(isGraphScrollLockButtonVisible(page));
					assertTrue(areFloatingControlsSideBySide(page));
					assertTrue(hasTopClearanceForFloatingControls(page));
					assertFalse(isPlotlyDragModeFalse(page));
					assertEquals(null, readGraphScrollLockStorage(page));

					setPlotlyPanMode(page);
					waitForPlotlyDragMode(page, "pan");
					page.click("#btn-graph-scroll-lock");
					assertEquals("Scroll Lock: ON", page.textContent("#btn-graph-scroll-lock"));
					assertEquals("true", page.getAttribute("#btn-graph-scroll-lock", "aria-pressed"));
					assertTrue(isGraphScrollLockButtonActive(page));
					waitForPlotlyDragModeFalse(page);
					assertEquals("true", readGraphScrollLockStorage(page));

					page.click("#btn-graph-scroll-lock");
					waitForPlotlyDragMode(page, "pan");
					assertEquals("false", readGraphScrollLockStorage(page));

					page.click("#btn-graph-scroll-lock");
					waitForPlotlyDragModeFalse(page);
					page.reload(new Page.ReloadOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
					waitForGraph(page);
					assertEquals("Scroll Lock: ON", page.textContent("#btn-graph-scroll-lock"));
					assertTrue(isGraphScrollLockButtonActive(page));
					waitForPlotlyDragModeFalse(page);

					page.click("#btn-screenshot");
					page.waitForFunction("document.body.classList.contains('screenshot-mode')",
							null, new Page.WaitForFunctionOptions().setTimeout(30000));
					assertFalse(isGraphScrollLockButtonVisible(page));
					assertEquals("true", readGraphScrollLockStorage(page));
				} finally {
					context.close();
				}
			} finally {
				browser.close();
			}
		}
	}

	@Test
	void graphScrollLockIsAvailableForMultiRunGraphs() {
		final String baseUrl = "http://127.0.0.1:" + port;

		assumeTrue(isMicrosoftEdgeInstalled(), "Microsoft Edge is not installed.");

		try (Playwright playwright = Playwright.create()) {
			final Browser browser = launchMicrosoftEdge(playwright);
			try {
				final BrowserContext context = browser.newContext(new Browser.NewContextOptions()
						.setViewportSize(1280, 720));
				try {
					final Page page = context.newPage();
					page.route("**/api/runs.json", route -> fulfillJson(route, runsJson()));
					page.route("**/api/metrics.json", route -> fulfillJson(route, metricsJson()));

					page.navigate(baseUrl + "/?multiRunGraphScrollLockTest=" + System.nanoTime(),
							new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
					waitForGraph(page);

					page.click("#btn-select-all-runs");
					waitForMultiRunGraph(page);
					assertTrue(isGraphScrollLockButtonVisible(page));
					assertFalse(isPlotlyDragModeFalse(page));

					page.click("#btn-graph-scroll-lock");
					assertTrue(isGraphScrollLockButtonActive(page));
					waitForPlotlyDragModeFalse(page);
					hoverFirstTraceMiddlePoint(page);
					waitForPlotlyHoverText(page);
					clickFirstLegendItem(page);
					waitForLegendOnlyTrace(page);
				} finally {
					context.close();
				}
			} finally {
				browser.close();
			}
		}
	}

	@Test
	void graphScrollLockAllowsVerticalTouchScrollingOnMobileGraph() {
		final String baseUrl = "http://127.0.0.1:" + port;

		assumeTrue(isMicrosoftEdgeInstalled(), "Microsoft Edge is not installed.");

		try (Playwright playwright = Playwright.create()) {
			final Browser browser = launchMicrosoftEdge(playwright);
			try {
				final BrowserContext context = browser.newContext(new Browser.NewContextOptions()
						.setViewportSize(1280, 720)
						.setHasTouch(true));
				try {
					final Page page = context.newPage();
					page.route("**/api/runs.json", route -> fulfillJson(route, manyGraphRunsJson(5)));
					page.route("**/api/metrics.json", route -> fulfillJson(route, manyGraphMetricsJson(5)));

					page.navigate(baseUrl + "/?mobileGraphScrollLockTest=" + System.nanoTime(),
							new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
					waitForGraphCount(page, 5);
					assertTrue(isMainAreaScrollable(page));

					page.click("#btn-graph-scroll-lock");
					waitForPlotlyDragModeFalse(page);
					for (int attempt = 0; attempt < 2; attempt++) {
						setMainAreaScrollTop(page, 0);
						dispatchTouchSwipe(page, readFirstGraphCenterX(page), readFirstGraphCenterY(page) + 90,
								readFirstGraphCenterY(page) - 130);
						waitForMainAreaScrolled(page);
						waitForPlotlyDragCoverRemoved(page);
					}
				} finally {
					context.close();
				}
			} finally {
				browser.close();
			}
		}
	}

	@Test
	void graphScrollLockAllowsDragScrollingOnGraph() {
		final String baseUrl = "http://127.0.0.1:" + port;

		assumeTrue(isMicrosoftEdgeInstalled(), "Microsoft Edge is not installed.");

		try (Playwright playwright = Playwright.create()) {
			final Browser browser = launchMicrosoftEdge(playwright);
			try {
				final BrowserContext context = browser.newContext(new Browser.NewContextOptions()
						.setViewportSize(1280, 720));
				try {
					final Page page = context.newPage();
					page.route("**/api/runs.json", route -> fulfillJson(route, manyGraphRunsJson(5)));
					page.route("**/api/metrics.json", route -> fulfillJson(route, manyGraphMetricsJson(5)));

					page.navigate(baseUrl + "/?graphDragScrollLockTest=" + System.nanoTime(),
							new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
					waitForGraphCount(page, 5);
					assertTrue(isMainAreaScrollable(page));

					page.click("#btn-graph-scroll-lock");
					waitForPlotlyDragModeFalse(page);
					for (int attempt = 0; attempt < 2; attempt++) {
						setMainAreaScrollTop(page, 0);
						dragFromFirstPlotBody(page, 1500);
						waitForMainAreaScrolled(page);
						waitForPlotlyDragCoverRemoved(page);
					}
				} finally {
					context.close();
				}
			} finally {
				browser.close();
			}
		}
	}

	@Test
	void autoReloadButtonReflectsToggleState() {
		final String baseUrl = "http://127.0.0.1:" + port;

		assumeTrue(isMicrosoftEdgeInstalled(), "Microsoft Edge is not installed.");

		try (Playwright playwright = Playwright.create()) {
			final Browser browser = launchMicrosoftEdge(playwright);
			try {
				final BrowserContext context = browser.newContext(new Browser.NewContextOptions()
						.setViewportSize(1280, 720));
				try {
					final Page page = context.newPage();
					page.route("**/api/runs.json", route -> fulfillJson(route, runsJson()));
					page.route("**/api/metrics.json", route -> fulfillJson(route, metricsJson()));

					page.navigate(baseUrl + "/?autoReloadButtonTest=" + System.nanoTime(),
							new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
					waitForGraph(page);

					assertEquals("Auto Reload: OFF", page.textContent("#btn-auto-reload"));
					assertEquals("false", page.getAttribute("#btn-auto-reload", "aria-pressed"));
					assertFalse(isAutoReloadButtonActive(page));

					page.click("#btn-auto-reload");
					assertEquals("Auto Reload: ON", page.textContent("#btn-auto-reload"));
					assertEquals("true", page.getAttribute("#btn-auto-reload", "aria-pressed"));
					assertTrue(isAutoReloadButtonActive(page));

					page.click("#btn-auto-reload");
					assertEquals("Auto Reload: OFF", page.textContent("#btn-auto-reload"));
					assertEquals("false", page.getAttribute("#btn-auto-reload", "aria-pressed"));
					assertFalse(isAutoReloadButtonActive(page));
				} finally {
					context.close();
				}
			} finally {
				browser.close();
			}
		}
	}

	@Test
	void logScaleTogglePersistsAcrossReloadsButNotPageRefresh() {
		final String baseUrl = "http://127.0.0.1:" + port;

		assumeTrue(isMicrosoftEdgeInstalled(), "Microsoft Edge is not installed.");

		try (Playwright playwright = Playwright.create()) {
			final Browser browser = launchMicrosoftEdge(playwright);
			try {
				final BrowserContext context = browser.newContext(new Browser.NewContextOptions()
						.setViewportSize(1280, 720));
				try {
					final Page page = context.newPage();
					page.route("**/api/runs.json", route -> fulfillJson(route, runsJson()));
					page.route("**/api/metrics.json", route -> fulfillJson(route, metricsJson()));

					page.navigate(baseUrl + "/?logScaleTest=" + System.nanoTime(),
							new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
					waitForGraph(page);

					assertEquals("linear", readYAxisType(page));
					assertEquals("Toggle signed log scale", page.getAttribute(".graph-log-toggle", "title"));

					page.click(".graph-log-toggle");
					waitForSignedLogTrace(page);
					assertEquals("linear", readYAxisType(page));

					page.click("#btn-reload");
					waitForGraph(page);
					waitForSignedLogTrace(page);
					assertEquals("linear", readYAxisType(page));

					page.click("#btn-auto-reload");
					page.evaluate("app.onReload()");
					waitForGraph(page);
					waitForSignedLogTrace(page);
					assertEquals("linear", readYAxisType(page));

					page.reload(new Page.ReloadOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
					waitForGraph(page);
					assertEquals("linear", readYAxisType(page));
					assertEquals("false", page.getAttribute(".graph-log-toggle", "aria-pressed"));
				} finally {
					context.close();
				}
			} finally {
				browser.close();
			}
		}
	}

	@Test
	void signedLogScaleKeepsNegativeZeroAndPositiveValues() {
		final String baseUrl = "http://127.0.0.1:" + port;

		assumeTrue(isMicrosoftEdgeInstalled(), "Microsoft Edge is not installed.");

		try (Playwright playwright = Playwright.create()) {
			final Browser browser = launchMicrosoftEdge(playwright);
			try {
				final BrowserContext context = browser.newContext(new Browser.NewContextOptions()
						.setViewportSize(1280, 720));
				try {
					final Page page = context.newPage();
					page.route("**/api/runs.json", route -> fulfillJson(route, signedLogRunsJson()));
					page.route("**/api/metrics.json", route -> fulfillJson(route, signedLogMetricsJson()));

					page.navigate(baseUrl + "/?signedLogScaleTest=" + System.nanoTime(),
							new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
					waitForGraph(page);

					page.click(".graph-log-toggle");
					waitForSignedLogMixedSignTrace(page);
					assertEquals("linear", readYAxisType(page));
				} finally {
					context.close();
				}
			} finally {
				browser.close();
			}
		}
	}

	@Test
	void signedLogZoomKeepsYAxisTickLabelsVisible() {
		final String baseUrl = "http://127.0.0.1:" + port;

		assumeTrue(isMicrosoftEdgeInstalled(), "Microsoft Edge is not installed.");

		try (Playwright playwright = Playwright.create()) {
			final Browser browser = launchMicrosoftEdge(playwright);
			try {
				final BrowserContext context = browser.newContext(new Browser.NewContextOptions()
						.setViewportSize(1280, 720));
				try {
					final Page page = context.newPage();
					page.route("**/api/runs.json", route -> fulfillJson(route, signedLogRunsJson()));
					page.route("**/api/metrics.json", route -> fulfillJson(route, signedLogZoomMetricsJson()));

					page.navigate(baseUrl + "/?signedLogZoomTickTest=" + System.nanoTime(),
							new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
					waitForGraph(page);

					page.click(".graph-log-toggle");
					waitForSignedLogZoomSourceTrace(page);
					zoomToSignedLogRange(page, 1, 3, 20, 30);
					waitForSignedLogZoomTicks(page);
				} finally {
					context.close();
				}
			} finally {
				browser.close();
			}
		}
	}

	@Test
	void signedLogZoomDoesNotResetPlotlyPanMode() {
		final String baseUrl = "http://127.0.0.1:" + port;

		assumeTrue(isMicrosoftEdgeInstalled(), "Microsoft Edge is not installed.");

		try (Playwright playwright = Playwright.create()) {
			final Browser browser = launchMicrosoftEdge(playwright);
			try {
				final BrowserContext context = browser.newContext(new Browser.NewContextOptions()
						.setViewportSize(1280, 720));
				try {
					final Page page = context.newPage();
					page.route("**/api/runs.json", route -> fulfillJson(route, signedLogRunsJson()));
					page.route("**/api/metrics.json", route -> fulfillJson(route, signedLogZoomMetricsJson()));

					page.navigate(baseUrl + "/?signedLogPanModeTest=" + System.nanoTime(),
							new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
					waitForGraph(page);

					page.click(".graph-log-toggle");
					waitForSignedLogZoomSourceTrace(page);
					zoomToSignedLogRange(page, 1, 3, 20, 30);
					waitForSignedLogZoomTicks(page);
					setPlotlyPanMode(page);
					waitForPlotlyDragMode(page, "pan");
					zoomToSignedLogRange(page, 1, 4, 20, 100);
					waitForPlotlyDragMode(page, "pan");
				} finally {
					context.close();
				}
			} finally {
				browser.close();
			}
		}
	}

	@Test
	void doubleClickingGraphAreaReloads() {
		final String baseUrl = "http://127.0.0.1:" + port;

		assumeTrue(isMicrosoftEdgeInstalled(), "Microsoft Edge is not installed.");

		try (Playwright playwright = Playwright.create()) {
			final Browser browser = launchMicrosoftEdge(playwright);
			try {
				final BrowserContext context = browser.newContext(new Browser.NewContextOptions()
						.setViewportSize(1280, 720));
				try {
					final Page page = context.newPage();
					page.route("**/api/runs.json", route -> fulfillJson(route, runsJson()));
					page.route("**/api/metrics.json", route -> fulfillJson(route, metricsJson()));

					page.navigate(baseUrl + "/?graphDoubleClickReloadTest=" + System.nanoTime(),
							new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
					waitForGraph(page);
					page.waitForFunction("""
							() => app?.mode === 'normal'
								&& !!document.getElementById('main-area')?.__mvGraphDblClickReloadHandler
							""", null, new Page.WaitForFunctionOptions().setTimeout(30000));

					page.evaluate("""
							() => {
								const original = app.onReload.bind(app);
								window.__graphDblClickReloadCalls = 0;
								window.__graphDblClickReloadSettled = 0;
								app.onReload = async () => {
									window.__graphDblClickReloadCalls += 1;
									try {
										return await original();
									} finally {
										window.__graphDblClickReloadSettled += 1;
									}
								};
							}
							""");

					page.dblclick(".js-plotly-plot");
					page.waitForFunction("""
							() => window.__graphDblClickReloadCalls === 1
								&& window.__graphDblClickReloadSettled === 1
							""", null, new Page.WaitForFunctionOptions().setTimeout(30000));
					assertEquals(1, ((Number) page.evaluate("() => window.__graphDblClickReloadCalls")).intValue());
				} finally {
					context.close();
				}
			} finally {
				browser.close();
			}
		}
	}

	@Test
	void hiddenTagSelectionRestoresWhenRunContainsTagAgain() {
		final String baseUrl = "http://127.0.0.1:" + port;

		assumeTrue(isMicrosoftEdgeInstalled(), "Microsoft Edge is not installed.");

		try (Playwright playwright = Playwright.create()) {
			final Browser browser = launchMicrosoftEdge(playwright);
			try {
				final BrowserContext context = browser.newContext(new Browser.NewContextOptions()
						.setViewportSize(1280, 720));
				try {
					final Page page = context.newPage();
					page.route("**/api/runs.json", route -> fulfillJson(route, splitTagRunsJson()));
					page.route("**/api/metrics.json", route -> fulfillJson(route, splitTagMetricsJson()));

					page.navigate(baseUrl + "/?tagRestoreTest=" + System.nanoTime(),
							new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
					waitForTag(page, TAG_B);

					selectSingleRun(page, "run_a");
					waitForTag(page, TAG_A);
					assertTrue(isTagActive(page, TAG_A));
					assertTrue(isTagActive(page, TAG_C));
					assertEquals(List.of(TAG_A, TAG_C), readTagList(page));

					waitForGraphTitle(page, TAG_A);
					assertEquals(List.of(TAG_A, TAG_C), readGraphTitles(page));

					selectSingleRun(page, "run_b");
					waitForTag(page, TAG_B);
					assertFalse(isTagVisible(page, TAG_A));
					assertFalse(isGraphTitleVisible(page, TAG_A));

					page.click("#btn-clear-all");
					assertFalse(isTagActive(page, TAG_B));

					selectSingleRun(page, "run_a");
					waitForTag(page, TAG_A);
					assertTrue(isTagActive(page, TAG_A));
					assertTrue(isTagActive(page, TAG_C));
					waitForGraphTitle(page, TAG_A);
					assertEquals(List.of(TAG_A, TAG_C), readTagList(page));
					assertEquals(List.of(TAG_A, TAG_C), readGraphTitles(page));
				} finally {
					context.close();
				}
			} finally {
				browser.close();
			}
		}
	}

	@Test
	void reloadActivatesNewTagReturnedByRunsMetadata() {
		final String baseUrl = "http://127.0.0.1:" + port;

		assumeTrue(isMicrosoftEdgeInstalled(), "Microsoft Edge is not installed.");

		try (Playwright playwright = Playwright.create()) {
			final Browser browser = launchMicrosoftEdge(playwright);
			try {
				final BrowserContext context = browser.newContext(new Browser.NewContextOptions()
						.setViewportSize(1280, 720));
				try {
					final Page page = context.newPage();
					final AtomicInteger runsRequestCount = new AtomicInteger();
					final AtomicInteger metricsRequestCount = new AtomicInteger();
					final List<String> metricsRequests = Collections.synchronizedList(new ArrayList<>());
					page.route("**/api/runs.json", route -> fulfillJson(
							route,
							reloadTagRunsJson(runsRequestCount.incrementAndGet() > 1)));
					page.route("**/api/metrics.json", route -> {
						final int requestIndex = metricsRequestCount.incrementAndGet();
						metricsRequests.add(route.request().postData());
						fulfillJson(route,
								requestIndex == 1 ? reloadInitialMetricsJson() : reloadNewTagMetricsJson());
					});

					page.navigate(baseUrl + "/?reloadNewTagTest=" + System.nanoTime(),
							new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
					waitForGraphTitle(page, RELOAD_OLD_TAG);
					assertFalse(isTagVisible(page, RELOAD_NEW_TAG));

					page.click("#btn-reload");
					waitForTag(page, RELOAD_NEW_TAG);
					waitForGraphTitle(page, RELOAD_NEW_TAG);

					assertTrue(isTagActive(page, RELOAD_NEW_TAG));
					assertTrue(readTagList(page).contains(RELOAD_OLD_TAG));
					assertTrue(readTagList(page).contains(RELOAD_NEW_TAG));
					assertTrue(isGraphTitleVisible(page, RELOAD_OLD_TAG));
					assertTrue(isGraphTitleVisible(page, RELOAD_NEW_TAG));
					assertTrue(metricsRequests.size() >= 2);
					assertTrue(metricsRequests.get(1).contains(RELOAD_OLD_TAG));
					assertTrue(metricsRequests.get(1).contains(RELOAD_NEW_TAG));
				} finally {
					context.close();
				}
			} finally {
				browser.close();
			}
		}
	}

	@Test
	void tagListAllowsVerticalTouchScrollingOnMobile() {
		final String baseUrl = "http://127.0.0.1:" + port;

		assumeTrue(isMicrosoftEdgeInstalled(), "Microsoft Edge is not installed.");

		try (Playwright playwright = Playwright.create()) {
			final Browser browser = launchMicrosoftEdge(playwright);
			try {
				final BrowserContext context = browser.newContext(new Browser.NewContextOptions()
						.setViewportSize(390, 640)
						.setIsMobile(true)
						.setHasTouch(true));
				try {
					final Page page = context.newPage();
					page.route("**/api/runs.json", route -> fulfillJson(route, manyTagRunsJson(60)));
					page.route("**/api/metrics.json", route -> fulfillJson(route, "{\"data\":[]}"));

					page.navigate(baseUrl + "/?mobileTagScrollTest=" + System.nanoTime(),
							new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
					page.waitForFunction("document.querySelectorAll('#tag-list li').length === 60",
							null, new Page.WaitForFunctionOptions().setTimeout(30000));

					assertTrue(isTagListScrollable(page));
					assertTrue(canScrollTagList(page));

					final String touchAction = readTagListTouchAction(page);
					assertFalse("none".equals(touchAction));
					assertTrue(touchAction.contains("pan-y") || "auto".equals(touchAction)
							|| "manipulation".equals(touchAction));
				} finally {
					context.close();
				}
			} finally {
				browser.close();
			}
		}
	}

	private void assertServedPaletteScript(String baseUrl) throws IOException, InterruptedException {
		final HttpRequest request = HttpRequest.newBuilder(URI.create(baseUrl + "/metrics-viewer.js"))
				.GET()
				.build();
		final HttpResponse<String> response = HttpClient.newHttpClient()
				.send(request, HttpResponse.BodyHandlers.ofString());

		assertEquals(200, response.statusCode());
		final String script = response.body();
		for (String color : RUN_COLORS) {
			assertTrue(script.contains(color), "metrics-viewer.js should contain " + color);
		}
		assertFalse(script.contains("getPlotlyColors"));
		assertFalse(script.contains("RUN_COLORS_FALLBACK"));
		assertFalse(script.contains("Plotly.colors.qualitative"));
	}

	private static Browser launchMicrosoftEdge(Playwright playwright) {
		try {
			return playwright.chromium().launch(new BrowserType.LaunchOptions()
					.setChannel("msedge")
					.setHeadless(true));
		} catch (PlaywrightException e) {
			assumeTrue(false, "Microsoft Edge Playwright launch is not available: " + e.getMessage());
			throw e;
		}
	}

	private static boolean isMicrosoftEdgeInstalled() {
		return Files.exists(Path.of("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"))
				|| Files.exists(Path.of("C:/Program Files/Microsoft/Edge/Application/msedge.exe"));
	}

	private static void fulfillJson(Route route, String body) {
		route.fulfill(new Route.FulfillOptions()
				.setStatus(200)
				.setContentType("application/json")
				.setBody(body));
	}

	private static void fulfillNoContent(Route route) {
		route.fulfill(new Route.FulfillOptions().setStatus(204).setBody(""));
	}

	private static void waitForGraph(Page page) {
		page.waitForFunction("document.querySelectorAll('.js-plotly-plot path.js-line').length > 0",
				null, new Page.WaitForFunctionOptions().setTimeout(30000));
	}

	private static void waitForMultiRunGraph(Page page) {
		page.waitForFunction("""
				() => {
					const plot = document.querySelector('.js-plotly-plot');
					return (plot?.data?.length ?? 0) > 1;
				}
				""", null, new Page.WaitForFunctionOptions().setTimeout(30000));
	}

	private static void waitForGraphCount(Page page, int count) {
		page.waitForFunction("""
				count => document.querySelectorAll('.js-plotly-plot').length === count
					&& document.querySelectorAll('.js-plotly-plot path.js-line').length >= count
				""", count, new Page.WaitForFunctionOptions().setTimeout(30000));
	}

	private static boolean isAutoReloadButtonActive(Page page) {
		return Boolean.TRUE.equals(page.evaluate("""
				() => document.getElementById('btn-auto-reload').classList.contains('active')
				"""));
	}

	private static boolean isGraphScrollLockButtonActive(Page page) {
		return Boolean.TRUE.equals(page.evaluate("""
				() => document.getElementById('btn-graph-scroll-lock').classList.contains('active')
				"""));
	}

	private static boolean isGraphScrollLockButtonVisible(Page page) {
		return Boolean.TRUE.equals(page.evaluate("""
				() => {
					const el = document.getElementById('btn-graph-scroll-lock');
					if (!el) return false;
					const style = getComputedStyle(el);
					return style.display !== 'none'
						&& style.visibility !== 'hidden'
						&& el.getClientRects().length > 0;
				}
				"""));
	}

	private static boolean areFloatingControlsSideBySide(Page page) {
		return Boolean.TRUE.equals(page.evaluate("""
				() => {
					const lock = document.getElementById('btn-graph-scroll-lock')?.getBoundingClientRect();
					const shot = document.getElementById('btn-screenshot-toggle')?.getBoundingClientRect();
					if (!lock || !shot) return false;
					const verticallyOverlaps = lock.top < shot.bottom && shot.top < lock.bottom;
					return verticallyOverlaps && lock.right <= shot.left + 1;
				}
				"""));
	}

	private static boolean hasTopClearanceForFloatingControls(Page page) {
		return Boolean.TRUE.equals(page.evaluate("""
				() => {
					const controls = document.getElementById('floating-controls')?.getBoundingClientRect();
					const firstGraph = document.querySelector('.graph-block')?.getBoundingClientRect();
					if (!controls || !firstGraph) return false;
					return firstGraph.top >= controls.bottom + 8;
				}
				"""));
	}

	private static String readGraphScrollLockStorage(Page page) {
		return (String) page.evaluate("""
				() => localStorage.getItem('anet.metricsviewer.graphScrollLockEnabled')
				""");
	}

	private static boolean isPlotlyDragModeFalse(Page page) {
		return Boolean.TRUE.equals(page.evaluate("""
				() => {
					const plot = document.querySelector('.js-plotly-plot');
					return (plot?._fullLayout?.dragmode ?? plot?.layout?.dragmode) === false;
				}
				"""));
	}

	private static void waitForPlotlyDragModeFalse(Page page) {
		page.waitForFunction("""
				() => {
					const plot = document.querySelector('.js-plotly-plot');
					return (plot?._fullLayout?.dragmode ?? plot?.layout?.dragmode) === false;
				}
				""", null, new Page.WaitForFunctionOptions().setTimeout(30000));
	}

	@SuppressWarnings("unchecked")
	private static Map<String, Number> readFirstTraceMiddlePointScreenPosition(Page page) {
		return (Map<String, Number>) page.evaluate("""
				() => {
					const plot = document.querySelector('.js-plotly-plot');
					const full = plot?._fullLayout;
					const trace = plot?.data?.[0];
					if (!plot || !full || !trace || trace.x.length < 2 || trace.y.length < 2) {
						throw new Error('plot point not ready');
					}
					const rect = plot.getBoundingClientRect();
					return {
						x: rect.left + full._size.l + full.xaxis.l2p(Number(trace.x[1])),
						y: rect.top + full._size.t + full.yaxis.l2p(Number(trace.y[1]))
					};
				}
				""");
	}

	private static void hoverFirstTraceMiddlePoint(Page page) {
		final Map<String, Number> position = readFirstTraceMiddlePointScreenPosition(page);
		page.mouse().move(position.get("x").doubleValue(), position.get("y").doubleValue());
	}

	private static void waitForPlotlyHoverText(Page page) {
		page.waitForFunction("""
				() => Array.from(document.querySelectorAll('.hovertext'))
					.some(el => (el.textContent || '').trim().length > 0)
				""", null, new Page.WaitForFunctionOptions().setTimeout(30000));
	}

	private static void clickFirstLegendItem(Page page) {
		page.click(".legend .traces");
	}

	private static void waitForLegendOnlyTrace(Page page) {
		page.waitForFunction("""
				() => {
					const plot = document.querySelector('.js-plotly-plot');
					return Array.from(plot?.data ?? []).some(trace => trace.visible === 'legendonly');
				}
				""", null, new Page.WaitForFunctionOptions().setTimeout(30000));
	}

	private static boolean isMainAreaScrollable(Page page) {
		return Boolean.TRUE.equals(page.evaluate("""
				() => {
					const el = document.getElementById('main-area');
					return !!el && el.scrollHeight > el.clientHeight;
				}
				"""));
	}

	private static void setMainAreaScrollTop(Page page, int scrollTop) {
		page.evaluate("""
				scrollTop => {
					const el = document.getElementById('main-area');
					if (!el) throw new Error('main-area not found');
					el.scrollTop = scrollTop;
				}
				""", scrollTop);
	}

	private static void waitForMainAreaScrolled(Page page) {
		page.waitForFunction("""
				() => (document.getElementById('main-area')?.scrollTop ?? 0) > 0
				""", null, new Page.WaitForFunctionOptions().setTimeout(30000));
	}

	private static void waitForPlotlyDragCoverRemoved(Page page) {
		page.waitForFunction("""
				() => !document.querySelector('.dragcover')
				""", null, new Page.WaitForFunctionOptions().setTimeout(30000));
	}

	private static double readFirstGraphCenterX(Page page) {
		return ((Number) page.evaluate("""
				() => {
					const rect = document.querySelector('.js-plotly-plot')?.getBoundingClientRect();
					if (!rect) throw new Error('plot not found');
					return rect.left + rect.width / 2;
				}
				""")).doubleValue();
	}

	private static double readFirstGraphCenterY(Page page) {
		return ((Number) page.evaluate("""
				() => {
					const rect = document.querySelector('.js-plotly-plot')?.getBoundingClientRect();
					if (!rect) throw new Error('plot not found');
					return rect.top + rect.height / 2;
				}
				""")).doubleValue();
	}

	private static void dispatchTouchSwipe(Page page, double x, double startY, double endY) {
		final CDPSession cdp = page.context().newCDPSession(page);
		try {
			dispatchTouchEvent(cdp, "touchStart", x, startY);
			final int steps = 10;
			for (int i = 1; i <= steps; i++) {
				final double y = startY + (endY - startY) * i / steps;
				dispatchTouchEvent(cdp, "touchMove", x, y);
				page.waitForTimeout(16);
			}
			dispatchTouchEvent(cdp, "touchEnd", x, endY);
		} finally {
			cdp.detach();
		}
	}

	private static void dispatchTouchEvent(CDPSession cdp, String type, double x, double y) {
		final JsonObject params = new JsonObject();
		params.addProperty("type", type);
		final JsonArray touchPoints = new JsonArray();
		if (!"touchEnd".equals(type) && !"touchCancel".equals(type)) {
			final JsonObject touchPoint = new JsonObject();
			touchPoint.addProperty("x", x);
			touchPoint.addProperty("y", y);
			touchPoint.addProperty("id", 1);
			touchPoint.addProperty("radiusX", 2);
			touchPoint.addProperty("radiusY", 2);
			touchPoint.addProperty("force", 1);
			touchPoints.add(touchPoint);
		}
		params.add("touchPoints", touchPoints);
		cdp.send("Input.dispatchTouchEvent", params);
	}

	@SuppressWarnings("unchecked")
	private static void dragFromFirstPlotBody(Page page, int holdMs) {
		final Map<String, Number> position = (Map<String, Number>) page.evaluate("""
				() => {
					const rect = document.querySelector('.js-plotly-plot .nsewdrag')?.getBoundingClientRect();
					if (!rect) throw new Error('plot body not found');
					return {
						x: rect.left + rect.width / 2,
						y: rect.top + rect.height * 0.7
					};
				}
				""");
		final double x = position.get("x").doubleValue();
		final double startY = position.get("y").doubleValue();
		final double endY = startY - 80;
		page.mouse().move(x, startY);
		assertTrue(Boolean.TRUE.equals(page.evaluate("""
				([x, y]) => document.elementFromPoint(x, y)?.classList.contains('nsewdrag') === true
				""", List.of(x, startY))), "Drag must start on Plotly's nsewdrag layer");
		page.mouse().down();
		page.waitForTimeout(holdMs);
		page.mouse().move(x, endY, new com.microsoft.playwright.Mouse.MoveOptions().setSteps(20));
		page.mouse().up();
	}

	private static String runsJson() {
		final StringBuilder sb = new StringBuilder();
		sb.append("{\"runs\":[");
		for (int i = 0; i < RUN_IDS.size(); i++) {
			if (i > 0) sb.append(',');
			sb.append(runJson(RUN_IDS.get(i), 2, TAG_KEY));
		}
		sb.append("]}");
		return sb.toString();
	}

	private static String manyTagRunsJson(int tagCount) {
		final List<String> tags = new ArrayList<>();
		for (int i = 0; i < tagCount; i++) tags.add("mobile/tag_" + String.format("%02d", i));
		return "{\"runs\":[" + runJson("run_mobile", 2, tags.toArray(String[]::new)) + "]}";
	}

	private static String manyGraphRunsJson(int graphCount) {
		final List<String> tags = new ArrayList<>();
		for (int i = 0; i < graphCount; i++) tags.add("mobile/graph_" + String.format("%02d", i));
		return "{\"runs\":[" + runJson("run_mobile_graph", 2, tags.toArray(String[]::new)) + "]}";
	}

	private static String metricsJson() {
		final StringBuilder sb = new StringBuilder();
		sb.append("{\"data\":[");
		for (int i = 0; i < RUN_IDS.size(); i++) {
			if (i > 0) sb.append(',');
			final float baseValue = i + 1;
			sb.append(rawSeriesJson(
					RUN_IDS.get(i),
					TAG_KEY,
					new double[] {0, 1, 2},
					new float[] {baseValue, baseValue + 1, baseValue + 2}));
		}
		sb.append("]}");
		return sb.toString();
	}

	private static String manyGraphMetricsJson(int graphCount) {
		final StringBuilder sb = new StringBuilder();
		sb.append("{\"data\":[");
		for (int i = 0; i < graphCount; i++) {
			if (i > 0) sb.append(',');
			sb.append(rawSeriesJson(
					"run_mobile_graph",
					"mobile/graph_" + String.format("%02d", i),
					new double[] {0, 1, 2},
					new float[] {i + 1, i + 2, i + 3}));
		}
		sb.append("]}");
		return sb.toString();
	}

	private static String signedLogRunsJson() {
		return "{\"runs\":[" + runJson("run_signed", 4, SIGNED_LOG_TAG) + "]}";
	}

	private static String signedLogMetricsJson() {
		return "{\"data\":[" + rawSeriesJson(
				"run_signed",
				SIGNED_LOG_TAG,
				new double[] {0, 1, 2, 3, 4},
				new float[] {-100, -9, 0, 9, 100}) + "]}";
	}

	private static String signedLogZoomMetricsJson() {
		return "{\"data\":[" + rawSeriesJson(
				"run_signed",
				SIGNED_LOG_TAG,
				new double[] {0, 1, 2, 3, 4},
				new float[] {-100, 20, 25, 30, 100}) + "]}";
	}

	private static String splitTagRunsJson() {
		return "{\"runs\":["
				+ runJson("run_a", 2, TAG_C, TAG_A)
				+ ","
				+ runJson("run_b", 2, TAG_B)
				+ "]}";
	}

	private static String splitTagMetricsJson() {
		return "{\"data\":["
				+ rawSeriesJson("run_a", TAG_A, new double[] {0, 1, 2}, new float[] {1, 2, 3})
				+ ","
				+ rawSeriesJson("run_a", TAG_C, new double[] {0, 1, 2}, new float[] {7, 8, 9})
				+ ","
				+ rawSeriesJson("run_b", TAG_B, new double[] {0, 1, 2}, new float[] {4, 5, 6})
				+ "]}";
	}

	private static String reloadTagRunsJson(boolean includeNewTag) {
		return "{\"runs\":["
				+ runJson(
						RELOAD_RUN,
						3,
						includeNewTag
								? new String[] {RELOAD_OLD_TAG, RELOAD_NEW_TAG}
								: new String[] {RELOAD_OLD_TAG})
				+ "]}";
	}

	private static String reloadInitialMetricsJson() {
		return "{\"data\":["
				+ rawSeriesJson(RELOAD_RUN, RELOAD_OLD_TAG, new double[] {0, 1, 2}, new float[] {1, 2, 3})
				+ "]}";
	}

	private static String reloadNewTagMetricsJson() {
		return "{\"data\":["
				+ rawSeriesJson(
						RELOAD_RUN,
						RELOAD_OLD_TAG,
						new double[] {0, 1, 2, 3},
						new float[] {1, 2, 3, 4})
				+ ","
				+ rawSeriesJson(
						RELOAD_RUN,
						RELOAD_NEW_TAG,
						new double[] {0, 1},
						new float[] {10, 11})
				+ "]}";
	}

	private static String runJson(String runId, long maxStep, String... tagKeys) {
		return runJsonWithState(runId, maxStep, "ready", 100, tagKeys);
	}

	private static String runJsonWithGeneration(
			String runId,
			long maxStep,
			String generation,
			String... tagKeys) {
		return runJson(runId, maxStep, generation, "ready", 100, tagKeys);
	}

	private static String runJsonWithState(
			String runId,
			long maxStep,
			String state,
			int percentage,
			String... tagKeys) {
		return runJson(runId, maxStep, GENERATION, state, percentage, tagKeys);
	}

	private static String runJson(
			String runId,
			long maxStep,
			String generation,
			String state,
			int percentage,
			String... tagKeys) {
		final StringBuilder tags = new StringBuilder();
		for (int i = 0; i < tagKeys.length; i++) {
			if (i > 0) tags.append(',');
			tags.append(tagJson(tagKeys[i], maxStep));
		}
		return "{\"id\":\"" + runId + "\","
				+ "\"generation\":\"" + generation + "\","
				+ "\"stats\":{\"maxStep\":" + maxStep + "},"
				+ "\"ingest\":{\"state\":\"" + state + "\",\"percentage\":" + percentage + "},"
				+ "\"tags\":[" + tags + "]}";
	}

	private static String tagJson(String tagKey, long maxStep) {
		final long count = maxStep + 1;
		return "{\"key\":\"" + tagKey + "\","
				+ "\"type\":\"scalar\",\"status\":\"ok\","
				+ "\"stats\":{"
				+ "\"minStep\":0,\"maxStep\":" + maxStep + ","
				+ "\"count\":" + count + ",\"lastValue\":1,"
				+ "\"minValue\":0,\"maxValue\":1,\"mean\":0.5,"
				+ "\"variance\":0.25,\"stdDev\":0.5}}";
	}

	private static String rawSeriesJson(
			String runId,
			String tagKey,
			double[] steps,
			float[] values) {
		return rawSeriesJson(runId, tagKey, GENERATION, steps, values);
	}

	private static String rawSeriesJson(
			String runId,
			String tagKey,
			String generation,
			double[] steps,
			float[] values) {
		final double fromStep = steps.length == 0 ? 0 : steps[0];
		final double toStep = steps.length == 0 ? 0 : steps[steps.length - 1];
		return "{\"runId\":\"" + runId + "\","
				+ "\"tagKey\":\"" + tagKey + "\","
				+ "\"generation\":\"" + generation + "\","
				+ "\"fromStep\":" + fromStep + ",\"toStep\":" + toStep + ","
				+ "\"availability\":\"ok\",\"pointBudget\":" + steps.length + ","
				+ "\"level\":0,\"bucketWidth\":1,\"issues\":[],"
				+ "\"projection\":{\"kind\":\"raw\","
				+ "\"steps\":\"" + encodeFloat64(steps) + "\","
				+ "\"values\":\"" + encodeFloat32(values) + "\"}}";
	}

	private static String lodMetricsJson() {
		return "{\"data\":[{"
				+ "\"runId\":\"run_lod_ui\",\"tagKey\":\"" + TAG_KEY + "\","
				+ "\"generation\":\"" + GENERATION + "\","
				+ "\"fromStep\":-31,\"toStep\":62,\"availability\":\"ok\","
				+ "\"pointBudget\":6,\"level\":1,\"bucketWidth\":16,\"issues\":[],"
				+ "\"projection\":{\"kind\":\"lod\","
				+ "\"minMax\":{\"steps\":\""
				+ encodeFloat64(new double[] {2, 8, 15, 18, 25, 31})
				+ "\",\"values\":\""
				+ encodeFloat32(new float[] {-2, 8, 1, -3, 9, 2})
				+ "\"},"
				+ "\"summary\":{\"steps\":\""
				+ encodeFloat64(new double[] {0, 16})
				+ "\",\"mins\":\""
				+ encodeFloat32(new float[] {-2, -3})
				+ "\",\"maxs\":\""
				+ encodeFloat32(new float[] {8, 9})
				+ "\",\"means\":\""
				+ encodeFloat32(new float[] {3, 4})
				+ "\",\"minSteps\":\""
				+ encodeFloat64(new double[] {2, 18})
				+ "\",\"maxSteps\":\""
				+ encodeFloat64(new double[] {8, 25})
				+ "\"}}}]}";
	}

	private static String statsWarningRunsJson() {
		return """
				{"runs":[
				  {
				    "id":"run_stats_a",
				    "generation":"%s",
				    "stats":{"maxStep":1},
				    "ingest":{"state":"ready","percentage":100},
				    "tags":[{
				      "key":"%s","type":"scalar","status":"ok",
				      "stats":{"minStep":0,"maxStep":1,"count":2,"lastValue":1,
				        "minValue":-1,"maxValue":1,"mean":0,"variance":1,"stdDev":1}
				    }]
				  },
				  {
				    "id":"run_stats_b",
				    "generation":"%s",
				    "stats":{"maxStep":1},
				    "ingest":{"state":"error","percentage":80,
				      "error":{"code":"invalid_json","message":"bad source line"}},
				    "tags":[{
				      "key":"%s","type":"scalar","status":"error",
				      "stats":{"minStep":0,"maxStep":1,"count":2,"lastValue":11,
				        "minValue":9,"maxValue":11,"mean":10,"variance":1,"stdDev":1},
				      "error":{"code":"tag_step_regression","message":"step regressed"}
				    }]
				  }
				]}
				""".formatted(GENERATION, TAG_KEY, GENERATION, TAG_KEY);
	}

	private static String statsWarningMetricsJson() {
		return "{\"data\":["
				+ rawSeriesJson(
						"run_stats_a",
						TAG_KEY,
						new double[] {0, 1},
						new float[] {-1, 1})
				+ ","
				+ rawSeriesJson(
						"run_stats_b",
						TAG_KEY,
						new double[] {0, 1},
						new float[] {9, 11})
				+ "]}";
	}

	private static String encodeFloat64(double[] values) {
		final ByteBuffer buffer = ByteBuffer.allocate(values.length * Double.BYTES)
				.order(ByteOrder.LITTLE_ENDIAN);
		for (double value : values) buffer.putDouble(value);
		return Base64.getEncoder().encodeToString(buffer.array());
	}

	private static String encodeFloat32(float[] values) {
		final ByteBuffer buffer = ByteBuffer.allocate(values.length * Float.BYTES)
				.order(ByteOrder.LITTLE_ENDIAN);
		for (float value : values) buffer.putFloat(value);
		return Base64.getEncoder().encodeToString(buffer.array());
	}

	private static List<Map<String, String>> expectedDisplayedChips() {
		final List<Map<String, String>> expected = new ArrayList<>();
		for (int i = 0; i < RUN_IDS.size(); i++) {
			final Map<String, String> chip = new LinkedHashMap<>();
			chip.put("runId", RUN_IDS.get(i));
			chip.put("color", hexToRgb(RUN_COLORS.get(i)));
			expected.add(chip);
		}
		Collections.reverse(expected);
		return expected;
	}

	@SuppressWarnings("unchecked")
	private static List<Map<String, String>> readRunChips(Page page) {
		return (List<Map<String, String>>) page.evaluate("""
				Array.from(document.querySelectorAll('#run-list .run-color')).map((el) => ({
					runId: el.parentElement.dataset.runId,
					color: getComputedStyle(el).backgroundColor
				}))
				""");
	}

	@SuppressWarnings("unchecked")
	private static List<String> readActiveRunIds(Page page) {
		return (List<String>) page.evaluate("""
				Array.from(document.querySelectorAll('#run-list .run-row.active'))
					.map(row => row.dataset.runId)
				""");
	}

	private static String readFirstLineStroke(Page page) {
		return (String) page.evaluate("""
				(() => {
					const line = document.querySelector('.js-plotly-plot path.js-line');
					return line ? (line.getAttribute('stroke') || getComputedStyle(line).stroke) : '';
				})()
				""");
	}

	private static String readFirstTraceUid(Page page) {
		return (String) page.evaluate("""
				(() => {
					const plot = document.querySelector('.js-plotly-plot');
					return plot?.data?.[0]?.uid ?? '';
				})()
				""");
	}

	private static String renderFirstPlotPngPrefix(Page page) {
		return (String) page.evaluate("""
				async () => {
					const plot = document.querySelector('.js-plotly-plot');
					const imageUrl = await Plotly.toImage(plot, {
						format: 'png',
						width: plot._fullLayout.width,
						height: plot._fullLayout.height
					});
					return imageUrl.slice(0, 'data:image/png;base64,'.length);
				}
				""");
	}

	@SuppressWarnings("unchecked")
	private static List<String> readModeBarButtonTitles(Page page) {
		return (List<String>) page.evaluate("""
				Array.from(document.querySelectorAll('.modebar-btn[data-title]'))
					.map(el => el.getAttribute('data-title'))
				""");
	}

	private static String readYAxisType(Page page) {
		return (String) page.evaluate("""
				(() => {
					const plot = document.querySelector('.js-plotly-plot');
					return plot ? (plot._fullLayout?.yaxis?.type || plot.layout?.yaxis?.type || 'linear') : '';
				})()
				""");
	}

	private static void waitForSignedLogTrace(Page page) {
		page.waitForFunction("""
				() => {
					const plot = document.querySelector('.js-plotly-plot');
					const button = document.querySelector('.graph-log-toggle');
					if (!plot || !button || button.getAttribute('aria-pressed') !== 'true') return false;
					if ((plot._fullLayout?.yaxis?.type || plot.layout?.yaxis?.type) !== 'linear') return false;

					const y = Array.from(plot.data?.[0]?.y ?? []);
					const customdata = Array.from(plot.data?.[0]?.customdata ?? []);
					const ticktext = Array.from(plot._fullLayout?.yaxis?.ticktext ?? plot.layout?.yaxis?.ticktext ?? []);
					return y.length === 3
						&& customdata.length === 3
						&& customdata.every((raw, index) => {
							const value = Number(raw);
							const expectedY = value === 0 ? 0 : Math.sign(value) * Math.log10(1 + Math.abs(value));
							return Number.isFinite(value)
								&& Number.isFinite(Number(y[index]))
								&& Math.abs(Number(y[index]) - expectedY) < 0.00001;
						})
						&& ticktext.length > 0;
				}
				""", null, new Page.WaitForFunctionOptions().setTimeout(30000));
	}

	private static void waitForSignedLogMixedSignTrace(Page page) {
		page.waitForFunction("""
				() => {
					const plot = document.querySelector('.js-plotly-plot');
					const button = document.querySelector('.graph-log-toggle');
					if (!plot || !button || button.getAttribute('aria-pressed') !== 'true') return false;
					if ((plot._fullLayout?.yaxis?.type || plot.layout?.yaxis?.type) !== 'linear') return false;

					const expectedRaw = [-100, -9, 0, 9, 100];
					const y = Array.from(plot.data?.[0]?.y ?? []);
					const customdata = Array.from(plot.data?.[0]?.customdata ?? []);
					const ticktext = Array.from(plot._fullLayout?.yaxis?.ticktext ?? plot.layout?.yaxis?.ticktext ?? []);
					const requiredTicks = ['-100', '-10', '-1', '0', '1', '10', '100'];
					return y.length === expectedRaw.length
						&& customdata.length === expectedRaw.length
						&& expectedRaw.every((raw, index) => {
							const expectedY = raw === 0 ? 0 : Math.sign(raw) * Math.log10(1 + Math.abs(raw));
							return Number(customdata[index]) === raw
								&& Number.isFinite(Number(y[index]))
								&& Math.abs(Number(y[index]) - expectedY) < 0.00001;
						})
						&& requiredTicks.every(tick => ticktext.includes(tick));
				}
				""", null, new Page.WaitForFunctionOptions().setTimeout(30000));
	}

	private static void waitForSignedLogZoomSourceTrace(Page page) {
		page.waitForFunction("""
				() => {
					const plot = document.querySelector('.js-plotly-plot');
					const button = document.querySelector('.graph-log-toggle');
					if (!plot || !button || button.getAttribute('aria-pressed') !== 'true') return false;

					const expectedRaw = [-100, 20, 25, 30, 100];
					const customdata = Array.from(plot.data?.[0]?.customdata ?? []);
					return expectedRaw.length === customdata.length
						&& expectedRaw.every((raw, index) => Number(customdata[index]) === raw);
				}
				""", null, new Page.WaitForFunctionOptions().setTimeout(30000));
	}

	private static void zoomToSignedLogRange(Page page, int minStep, int maxStep, double minValue, double maxValue) {
		page.evaluate("""
				({ minStep, maxStep, minValue, maxValue }) => {
					const plot = document.querySelector('.js-plotly-plot');
					const signedLog = value => Math.sign(value) * Math.log10(1 + Math.abs(value));
					return Plotly.relayout(plot, {
						'xaxis.range[0]': minStep,
						'xaxis.range[1]': maxStep,
						'yaxis.range[0]': signedLog(minValue),
						'yaxis.range[1]': signedLog(maxValue)
					});
				}
				""", Map.of(
				"minStep", minStep,
				"maxStep", maxStep,
				"minValue", minValue,
				"maxValue", maxValue));
	}

	private static void waitForSignedLogZoomTicks(Page page) {
		page.waitForFunction("""
				() => {
					const plot = document.querySelector('.js-plotly-plot');
					const yaxis = plot?._fullLayout?.yaxis ?? plot?.layout?.yaxis;
					const tickvals = Array.from(yaxis?.tickvals ?? []);
					const ticktext = Array.from(yaxis?.ticktext ?? []);
					const range = Array.from(yaxis?.range ?? []);
					if (range.length !== 2 || tickvals.length !== ticktext.length) return false;

					const visibleTicktext = ticktext.filter((_, index) => {
						const value = Number(tickvals[index]);
						return Number.isFinite(value)
							&& range[0] - 0.000001 <= value
							&& value <= range[1] + 0.000001;
					});
					return visibleTicktext.includes('20') && visibleTicktext.includes('30');
				}
				""", null, new Page.WaitForFunctionOptions().setTimeout(30000));
	}

	private static void setPlotlyPanMode(Page page) {
		page.evaluate("""
				() => {
					const plot = document.querySelector('.js-plotly-plot');
					return Plotly.relayout(plot, { dragmode: 'pan' });
				}
				""");
	}

	private static void waitForPlotlyDragMode(Page page, String dragMode) {
		page.waitForFunction("""
				dragMode => {
					const plot = document.querySelector('.js-plotly-plot');
					return (plot?._fullLayout?.dragmode ?? plot?.layout?.dragmode) === dragMode;
				}
				""", dragMode, new Page.WaitForFunctionOptions().setTimeout(30000));
	}

	@SuppressWarnings("unchecked")
	private static List<String> readTagList(Page page) {
		return (List<String>) page.evaluate("""
				Array.from(document.querySelectorAll('#tag-list li')).map(el => el.textContent)
				""");
	}

	@SuppressWarnings("unchecked")
	private static List<String> readGraphTitles(Page page) {
		return (List<String>) page.evaluate("""
				Array.from(document.querySelectorAll('.graph-title')).map(el => el.textContent)
				""");
	}

	private static boolean isTagListScrollable(Page page) {
		return Boolean.TRUE.equals(page.evaluate("""
				() => {
					const el = document.querySelector('#tag-list');
					return !!el && el.scrollHeight > el.clientHeight;
				}
				"""));
	}

	private static boolean canScrollTagList(Page page) {
		return Boolean.TRUE.equals(page.evaluate("""
				() => {
					const el = document.querySelector('#tag-list');
					if (!el) return false;
					el.scrollTop = 0;
					el.scrollBy(0, 120);
					return el.scrollTop > 0;
				}
				"""));
	}

	private static String readTagListTouchAction(Page page) {
		return (String) page.evaluate("""
				() => {
					const el = document.querySelector('#tag-list');
					return el ? getComputedStyle(el).touchAction : '';
				}
				""");
	}

	private static void selectSingleRun(Page page, String runId) {
		page.evaluate("""
				runId => {
					const row = Array.from(document.querySelectorAll('#run-list .run-row'))
						.find(el => el.dataset.runId === runId);
					if (!row) throw new Error('run not found: ' + runId);
					row.click();
					row.click();
				}
				""", runId);
	}

	private static void clickTag(Page page, String tagKey) {
		page.evaluate("""
				tagKey => {
					const tag = Array.from(document.querySelectorAll('#tag-list li'))
						.find(el => el.textContent === tagKey);
					if (!tag) throw new Error('tag not found: ' + tagKey);
					tag.click();
				}
				""", tagKey);
	}

	private static void waitForTag(Page page, String tagKey) {
		page.waitForFunction("""
				tagKey => Array.from(document.querySelectorAll('#tag-list li'))
					.some(el => el.textContent === tagKey)
				""", tagKey, new Page.WaitForFunctionOptions().setTimeout(30000));
	}

	private static void waitForGraphTitle(Page page, String tagKey) {
		page.waitForFunction("""
				tagKey => Array.from(document.querySelectorAll('.graph-title'))
					.some(el => el.textContent === tagKey)
				""", tagKey, new Page.WaitForFunctionOptions().setTimeout(30000));
	}

	private static boolean isTagVisible(Page page, String tagKey) {
		return Boolean.TRUE.equals(page.evaluate("""
				tagKey => Array.from(document.querySelectorAll('#tag-list li'))
					.some(el => el.textContent === tagKey)
				""", tagKey));
	}

	private static boolean isTagActive(Page page, String tagKey) {
		return Boolean.TRUE.equals(page.evaluate("""
				tagKey => {
					const tag = Array.from(document.querySelectorAll('#tag-list li'))
						.find(el => el.textContent === tagKey);
					return !!tag && tag.classList.contains('active');
				}
				""", tagKey));
	}

	private static boolean isGraphTitleVisible(Page page, String tagKey) {
		return Boolean.TRUE.equals(page.evaluate("""
				tagKey => Array.from(document.querySelectorAll('.graph-title'))
					.some(el => el.textContent === tagKey)
				""", tagKey));
	}

	private static String hexToRgb(String hex) {
		final int r = Integer.parseInt(hex.substring(1, 3), 16);
		final int g = Integer.parseInt(hex.substring(3, 5), 16);
		final int b = Integer.parseInt(hex.substring(5, 7), 16);
		return "rgb(" + r + ", " + g + ", " + b + ")";
	}
}
