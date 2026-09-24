package io.github.kazukin123.anetlab.metricsviewer.view;

import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.GENERATION;
import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.TAG_KEY;
import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.colorMetricsJson;
import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.colorRunIds;
import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.colorRunsJson;
import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.metricsJson;
import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.rawSeriesJson;
import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.runJson;
import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.runsJson;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Map;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

import com.microsoft.playwright.Page;
import com.microsoft.playwright.Route;
import com.microsoft.playwright.options.WaitUntilState;

@SpringBootTest(
		webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT,
		properties = "metricsviewer.workspaces-dir=target/playwright-test-empty-workspaces")
class RunListPlaywrightTest extends MetricsViewerPlaywrightTestSupport {

	// RUN_COLORSのうち、Run色テストで期待値に使う色。コメントはpalette上のindex。
	private static final String BLUE = "rgb(47, 125, 225)"; // #2F7DE1 [0]
	private static final String PURPLE = "rgb(122, 92, 255)"; // #7A5CFF [2]
	private static final String RED = "rgb(226, 59, 79)"; // #E23B4F [9]
	private static final String LIME = "rgb(209, 216, 59)"; // #D1D83B [12]
	private static final String MAGENTA = "rgb(184, 50, 128)"; // #B83280 [13]
	private static final String GREEN = "rgb(0, 179, 107)"; // #00B36B [14]

	@Test
	void metricsRequestsKeepOneChannelAndIncreaseTheirSequence() throws Exception {
		final List<Map<String, String>> headers = new CopyOnWriteArrayList<>();
		final CountDownLatch twoRequests = new CountDownLatch(2);
		page.route("**/api/runs.json", route -> fulfillJson(route, runsJson()));
		page.route("**/api/metrics.json", route -> {
			headers.add(route.request().headers());
			twoRequests.countDown();
			fulfillJson(route, metricsJson());
		});
		page.route("**/api/runs/prioritize", MetricsViewerPlaywrightTestSupport::fulfillNoContent);

		page.navigate(baseUrl + "/?queryChannelTest=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		waitForGraph(page);
		page.evaluate("() => app.requestVisibleData({ force: true })");
		assertTrue(twoRequests.await(5, TimeUnit.SECONDS));

		final String channel = headers.get(0).get("x-query-channel");
		assertTrue(channel != null && !channel.isBlank());
		assertEquals(channel, headers.get(1).get("x-query-channel"));
		assertEquals("0", headers.get(0).get("x-query-sequence"));
		assertEquals("1", headers.get(1).get("x-query-sequence"));
	}

	@Test
	void metricsRequestsUseFallbackChannelWhenRandomUuidIsUnavailable() throws Exception {
		final List<Map<String, String>> headers = new CopyOnWriteArrayList<>();
		final CountDownLatch twoRequests = new CountDownLatch(2);
		page.addInitScript("""
				Object.defineProperty(globalThis.crypto, 'randomUUID', {
					configurable: true,
					value: undefined
				});
				""");
		page.route("**/api/runs.json", route -> fulfillJson(route, runsJson()));
		page.route("**/api/metrics.json", route -> {
			headers.add(route.request().headers());
			twoRequests.countDown();
			fulfillJson(route, metricsJson());
		});
		page.route("**/api/runs/prioritize", MetricsViewerPlaywrightTestSupport::fulfillNoContent);

		page.navigate(baseUrl + "/?queryChannelFallbackTest=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		waitForGraph(page);
		assertEquals("undefined", page.evaluate("() => typeof crypto.randomUUID"));
		page.evaluate("() => app.requestVisibleData({ force: true })");
		assertTrue(twoRequests.await(5, TimeUnit.SECONDS));

		final String channel = headers.get(0).get("x-query-channel");
		assertTrue(channel != null && !channel.isBlank());
		assertTrue(channel.length() <= 128);
		assertEquals(channel, headers.get(1).get("x-query-channel"));
		assertEquals("0", headers.get(0).get("x-query-sequence"));
		assertEquals("1", headers.get(1).get("x-query-sequence"));
	}

	@Test
	void supersededMetricsResponseIsNotShownOrLoggedAsAnUpdateFailure() {
		final AtomicInteger metricsRequests = new AtomicInteger();
		final List<String> consoleErrors = new CopyOnWriteArrayList<>();
		page.onConsoleMessage(message -> {
			if ("error".equals(message.type())
					&& !message.text().startsWith("Failed to load resource:")) {
				consoleErrors.add(message.text());
			}
		});
		page.route("**/api/runs.json", route -> fulfillJson(route, runsJson()));
		page.route("**/api/metrics.json", route -> {
			if (metricsRequests.incrementAndGet() == 1) {
				fulfillJson(route, metricsJson());
				return;
			}
			route.fulfill(new Route.FulfillOptions()
					.setStatus(409)
					.setContentType("application/json")
					.setBody("{\"code\":\"superseded\",\"message\":\"newer query\"}"));
		});
		page.route("**/api/runs/prioritize", MetricsViewerPlaywrightTestSupport::fulfillNoContent);

		page.navigate(baseUrl + "/?supersededTest=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		waitForGraph(page);
		page.waitForResponse("**/api/metrics.json", () -> page.evaluate("""
				() => {
					app.fetcher.fetchMetrics([]).catch(error => {
						app._setUpdateFailure('metrics', error);
						app._handleQueryError(error);
					});
				}
				"""));
		page.waitForFunction("""
				() => !document.getElementById('loading-spinner')?.classList.contains('active')
				""");
		page.waitForTimeout(100);

		assertFalse(page.locator("#update-status").isVisible());
		assertEquals(List.of(), consoleErrors);
	}

	@Test
	void runRowsToggleImmediatelyAllowEmptySelectionAndSoloOnTheSecondClick() {
		page.route("**/api/runs.json", route -> fulfillJson(route, runsJson()));
		page.route("**/api/metrics.json", route -> fulfillJson(route, metricsJson()));
		page.route("**/api/runs/prioritize", MetricsViewerPlaywrightTestSupport::fulfillNoContent);

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
	}

	@Test
	void initialLoadErrorAllowsReloadRecoveryButNotScreenshotEscape() {
		final AtomicInteger runsRequests = new AtomicInteger();
		page.route("**/api/runs.json", route -> {
			if (runsRequests.incrementAndGet() == 1) route.abort();
			else fulfillJson(route, runsJson());
		});
		page.route("**/api/metrics.json", route -> fulfillJson(route, metricsJson()));
		page.route("**/api/runs/prioritize", MetricsViewerPlaywrightTestSupport::fulfillNoContent);

		page.navigate(baseUrl + "/?errorRecoveryTest=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		page.waitForFunction("document.body.classList.contains('error') && app?.mode === 'error'");

		assertEquals("none", page.evaluate(
				"() => getComputedStyle(document.body, '::before').pointerEvents"));
		page.click("#btn-screenshot");
		assertTrue(Boolean.TRUE.equals(page.evaluate("""
				() => document.body.classList.contains('error')
					&& !document.body.classList.contains('screenshot-mode')
					&& app?.mode === 'error'
				""")));

		page.click("#btn-reload");
		page.waitForFunction("""
				() => app?.mode === 'normal'
					&& document.querySelectorAll('.js-plotly-plot').length > 0
				""");
		assertFalse(Boolean.TRUE.equals(page.evaluate(
				"() => document.body.classList.contains('error')")));
	}

	@Test
	void updateFailuresRemainVisibleUntilEachRequestTypeSucceeds() {
		final AtomicInteger runsRequests = new AtomicInteger();
		final AtomicInteger metricsRequests = new AtomicInteger();
		page.route("**/api/runs.json", route -> {
			if (runsRequests.incrementAndGet() == 4) {
				route.fulfill(new Route.FulfillOptions()
						.setStatus(503)
						.setContentType("text/plain")
						.setBody("metadata unavailable"));
			} else {
				fulfillJson(route, runsJson());
			}
		});
		page.route("**/api/metrics.json", route -> {
			if (metricsRequests.incrementAndGet() == 2) {
				route.fulfill(new Route.FulfillOptions()
						.setStatus(502)
						.setContentType("text/plain")
						.setBody("metrics unavailable"));
			} else {
				fulfillJson(route, metricsJson());
			}
		});
		page.route("**/api/runs/prioritize", MetricsViewerPlaywrightTestSupport::fulfillNoContent);

		page.navigate(baseUrl + "/?updateFailureTest=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		waitForGraph(page);
		assertEquals(1, page.locator("#update-status").count());
		assertFalse(page.locator("#update-status").isVisible());

		page.click("#btn-reload");
		page.waitForFunction("""
				() => document.getElementById('update-status')?.title
					=== 'Metrics: Failed metrics.json: 502 metrics unavailable'
				""");
		assertEquals("Update failed", page.textContent("#update-status"));
		page.evaluate("() => app.refreshMetadata({ requestData: false })");
		assertEquals(
				"Metrics: Failed metrics.json: 502 metrics unavailable",
				page.getAttribute("#update-status", "title"));

		page.click("#btn-reload");
		page.waitForFunction("""
				() => document.getElementById('update-status')?.title
					=== 'Metadata: Failed runs.json: 503\\n'
						+ 'Metrics: Failed metrics.json: 502 metrics unavailable'
				""");

		page.click("#run-list .run-row[data-run-id='run_10']");
		page.waitForFunction("""
				() => document.getElementById('update-status')?.title
					=== 'Metadata: Failed runs.json: 503'
				""");

		page.click("#btn-reload");
		page.waitForFunction("""
				() => {
					const status = document.getElementById('update-status');
					return status?.hidden && status.textContent === '' && status.title === '';
				}
				""");
	}

	@Test
	void disappearingRunIsRemovedFromSelectionWindowsAndColorCache() {
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
		page.route("**/api/runs/prioritize", MetricsViewerPlaywrightTestSupport::fulfillNoContent);

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
	}

	@Test
	void convertingRunPollOnlyUpdatesExistingPercentageUntilReload() {
		final String nextGeneration = "00000000-0000-0000-0000-000000000002";
		final String newTag = "poll/new-tag";
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
		page.route("**/api/runs/prioritize", MetricsViewerPlaywrightTestSupport::fulfillNoContent);

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
							const selectedOnly = document.getElementById('btn-selected-only');
							autoReload.classList.add('active');
							scrollLock.classList.add('active');
							log.classList.add('active');
							selectedOnly.classList.add('active');
							return [
								getComputedStyle(autoReload).backgroundColor,
								getComputedStyle(scrollLock).backgroundColor,
								getComputedStyle(log).backgroundColor,
								getComputedStyle(selectedOnly).backgroundColor,
								getComputedStyle(selectedOnly).borderColor,
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
		// metadataの更新後にmetrics再取得が非同期で走るので、カウンタ側も待ってから見る。
		page.waitForCondition(() -> metricsRequests.get() > initialMetricsRequests);
		assertTrue(metricsRequests.get() > initialMetricsRequests);
	}

	@Test
	void autoReloadButtonReflectsToggleState() {
		page.route("**/api/runs.json", route -> fulfillJson(route, runsJson()));
		page.route("**/api/metrics.json", route -> fulfillJson(route, metricsJson()));

		page.navigate(baseUrl + "/?autoReloadButtonTest=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		waitForGraph(page);

		assertEquals("Auto Reload", page.textContent("#btn-auto-reload"));
		assertFalse(isAutoReloadButtonActive(page));

		page.click("#btn-auto-reload");
		assertTrue(isAutoReloadButtonActive(page));

		page.click("#btn-auto-reload");
		assertEquals("Auto Reload", page.textContent("#btn-auto-reload"));
		assertFalse(isAutoReloadButtonActive(page));
	}

	@Test
	void recolorButtonGivesSelectedRunsTheMostDistinguishableColors() {
		openColorFixture("recolorButtonTest", 14);

		setAutoRecolor(page, false);
		selectSingleRun(page, "run_01");
		clickRunRow(page, "run_03");
		clickRunRow(page, "run_10");
		clickRunRow(page, "run_14");
		page.click("#btn-recolor-runs");
		waitForSelectedTraceCount(page, 4);

		assertEquals(
				List.of(BLUE, LIME, RED, GREEN),
				readChipColors(page, List.of("run_01", "run_03", "run_10", "run_14")));
		assertEquals(readChipColor(page, "run_01"), readTraceColor(page, "run_01"));
		assertEquals(readChipColor(page, "run_14"), readTraceColor(page, "run_14"));
	}

	@Test
	void recolorButtonIsIdempotentForTheSameSelection() {
		openColorFixture("recolorIdempotentTest", 14);

		setAutoRecolor(page, false);
		selectSingleRun(page, "run_01");
		clickRunRow(page, "run_03");
		clickRunRow(page, "run_10");
		page.click("#btn-recolor-runs");
		final List<String> first = readChipColors(page, List.of("run_01", "run_03", "run_10"));

		page.click("#btn-recolor-runs");

		assertEquals(List.of(BLUE, LIME, RED), first);
		assertEquals(first, readChipColors(page, List.of("run_01", "run_03", "run_10")));
	}

	@Test
	void autoRecolorMovesTheAddedRunOffAConfusableBaseColor() {
		// 既定ONのまま操作する。run_10=#E23B4F と run_14=#B83280 の分離距離は0.1204で、しきい値0.16を下回る。
		openColorFixture("autoRecolorTest", 14);

		assertTrue(isToggleOn(page, "#btn-auto-recolor"));
		clickRunRow(page, "run_14"); // 初期選択の最新Runを外し、空選択から選び直す
		clickRunRow(page, "run_10");
		clickRunRow(page, "run_14");
		waitForSelectedTraceCount(page, 2);

		assertEquals(RED, readChipColor(page, "run_10"));
		assertEquals(LIME, readChipColor(page, "run_14"));
		assertEquals(LIME, readTraceColor(page, "run_14"));
	}

	@Test
	void autoRecolorKeepsEarlierRunsWhenTheSelectionGrowsOrShrinks() {
		openColorFixture("autoRecolorStabilityTest", 14);

		clickRunRow(page, "run_14");
		clickRunRow(page, "run_10");
		clickRunRow(page, "run_14");
		// run_03 の基本色 #7A5CFF は先行2本から十分離れているので、足しても動かない。
		clickRunRow(page, "run_03");
		waitForSelectedTraceCount(page, 3);

		assertEquals(
				List.of(RED, LIME, PURPLE),
				readChipColors(page, List.of("run_10", "run_14", "run_03")));

		clickRunRow(page, "run_14");

		assertEquals(List.of(RED, PURPLE), readChipColors(page, List.of("run_10", "run_03")));
	}

	@Test
	void autoRecolorDefaultsToOnAndSurvivesReload() {
		openColorFixture("autoRecolorStorageTest", 14);

		assertTrue(isToggleOn(page, "#btn-auto-recolor"));
		assertNull(readAutoRecolorStorage(page));

		setAutoRecolor(page, false);
		page.reload(new Page.ReloadOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		waitForRunRows(page, 14);

		assertFalse(isToggleOn(page, "#btn-auto-recolor"));
		assertEquals("false", readAutoRecolorStorage(page));
	}

	@Test
	void recolorLeavesUnselectedRunsOnTheirBaseColors() {
		openColorFixture("unselectedRunColorTest", 14);

		setAutoRecolor(page, false);
		selectSingleRun(page, "run_01");
		clickRunRow(page, "run_03");
		clickRunRow(page, "run_10");
		clickRunRow(page, "run_14");
		page.click("#btn-recolor-runs");

		// run_03 が #D1D83B を取っても、それを基本色に持つ未選択の run_13 は動かない。
		assertEquals(LIME, readChipColor(page, "run_03"));
		assertEquals(LIME, readChipColor(page, "run_13"));
	}

	@Test
	void recolorRepeatsThePaletteAfterTwentySelectedRuns() {
		openColorFixture("recolorRoundTest", 25);

		setAutoRecolor(page, false);
		page.click("#btn-select-all-runs");
		page.click("#btn-recolor-runs");

		final List<String> colors = readChipColors(page, colorRunIds(25));
		assertEquals(20, colors.subList(0, 20).stream().distinct().count());
		assertEquals(BLUE, colors.get(0));
		assertEquals(BLUE, colors.get(20));
	}

	@Test
	void autoRecolorKeepsTheColorsAssignedByTheRecolorButton() {
		openColorFixture("recolorWithAutoOnTest", 14);

		clickRunRow(page, "run_14");
		clickRunRow(page, "run_01");
		clickRunRow(page, "run_03");
		clickRunRow(page, "run_10");
		page.click("#btn-recolor-runs");
		final List<String> afterButton = readChipColors(page, List.of("run_01", "run_03", "run_10"));

		clickRunRow(page, "run_14"); // Autoが走る選択変更
		waitForSelectedTraceCount(page, 4);

		assertEquals(List.of(BLUE, LIME, RED), afterButton);
		assertEquals(afterButton, readChipColors(page, List.of("run_01", "run_03", "run_10")));
		assertEquals(GREEN, readChipColor(page, "run_14"));
	}

	@Test
	void recolorButtonDoesNothingForASingleSelectedRun() {
		openColorFixture("recolorSingleRunTest", 14);

		setAutoRecolor(page, false);
		selectSingleRun(page, "run_10");

		page.click("#btn-recolor-runs");

		assertEquals(RED, readChipColor(page, "run_10"));
		assertEquals(MAGENTA, readChipColor(page, "run_14"));
	}

	@Test
	void sidePanelControlRowsFitOnOneLineWithoutShrinkingButtons() {
		openColorFixture("controlRowLayoutTest", 14);

		// 行が横へ溢れていないか（overflow）と、中身が押し潰されていないか（shrunk）を見る。
		// 見出し行はラベルとボタンで高さが違うので、topの一致では判定できない。
		assertEquals(
				"run-section/header:false:false|run-section/controls:false:false"
						+ "|tag-section/header:false:false",
				page.evaluate("""
						() => [...document.querySelectorAll('.section-header, .section-controls')]
							.map(row => {
								const overflow = row.scrollWidth > row.clientWidth + 1;
								const shrunk = [...row.children].some(
									el => el.scrollWidth > el.clientWidth + 1);
								const kind = row.classList.contains('section-header')
									? 'header' : 'controls';
								return row.closest('.section').id + '/' + kind
									+ ':' + overflow + ':' + shrunk;
							})
							.join('|')
						"""));
	}

	@Test
	void selectedOnlyHidesUnselectedRowsInBothSections() {
		openColorFixture("selectedOnlyTest", 14);
		selectSingleRun(page, "run_10");

		assertEquals("run:14/14|tag:1/1", readVisibleListCounts(page));

		setToggle(page, "#btn-selected-only-runs", true);
		setToggle(page, "#btn-selected-only", true);
		assertEquals("run:1/14|tag:1/1", readVisibleListCounts(page));

		// タグを未選択にすると、選択中しか残さないTag listからも消える。
		page.click("#tag-list li");
		assertEquals("run:1/14|tag:0/1", readVisibleListCounts(page));

		// OFFに戻せば、未選択のままのタグもまた見える。
		setToggle(page, "#btn-selected-only-runs", false);
		setToggle(page, "#btn-selected-only", false);
		assertEquals("run:14/14|tag:1/1", readVisibleListCounts(page));
	}

	@Test
	void bulkSelectionButtonsTurnOffSelectedOnly() {
		openColorFixture("bulkTurnsOffSelectedOnlyTest", 14);

		setToggle(page, "#btn-selected-only-runs", true);
		page.click("#btn-latest-only");
		assertFalse(isToggleOn(page, "#btn-selected-only-runs"), "Select Latest");

		setToggle(page, "#btn-selected-only", true);
		page.click("#btn-clear-all");
		assertFalse(isToggleOn(page, "#btn-selected-only"), "Clear All");

		setToggle(page, "#btn-selected-only", true);
		page.click("#btn-select-all");
		assertFalse(isToggleOn(page, "#btn-selected-only"), "Tags Select All");

		setToggle(page, "#btn-selected-only-runs", true);
		page.click("#btn-select-all-runs");
		assertFalse(isToggleOn(page, "#btn-selected-only-runs"), "Runs Select All");
	}

	/**
	 * 「可視行/全行」をsectionごとに連結して返す。
	 * hidden属性の有無ではなく実際に描画されているかを見る（display持ちの要素ではhiddenが効かないことがある）。
	 */
	private static String readVisibleListCounts(Page page) {
		return (String) page.evaluate("""
				() => {
					const count = selector => {
						const all = [...document.querySelectorAll(selector)];
						return all.filter(el => el.getClientRects().length > 0).length
							+ '/' + all.length;
					};
					return 'run:' + count('#run-list .run-row')
						+ '|tag:' + count('#tag-list li');
				}
				""");
	}

	private void openColorFixture(String testName, int runCount) {
		page.route("**/api/runs.json", route -> fulfillJson(route, colorRunsJson(runCount)));
		page.route("**/api/metrics.json", route -> fulfillJson(route, colorMetricsJson(runCount)));
		page.route("**/api/runs/prioritize", MetricsViewerPlaywrightTestSupport::fulfillNoContent);

		page.navigate(baseUrl + "/?" + testName + "=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		waitForGraph(page);
		waitForRunRows(page, runCount);
	}
}
