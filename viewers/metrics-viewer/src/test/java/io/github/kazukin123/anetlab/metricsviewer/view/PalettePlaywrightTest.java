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
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.server.LocalServerPort;

import com.microsoft.playwright.Browser;
import com.microsoft.playwright.BrowserContext;
import com.microsoft.playwright.BrowserType;
import com.microsoft.playwright.Page;
import com.microsoft.playwright.Playwright;
import com.microsoft.playwright.PlaywrightException;
import com.microsoft.playwright.Route;
import com.microsoft.playwright.options.WaitUntilState;

@SpringBootTest(
		webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT,
		properties = "metricsviewer.runs-dir=target/playwright-test-empty-runs")
class PalettePlaywrightTest {

	private static final String TAG_KEY = "palette/test";
	private static final String TAG_A = "tag/a";
	private static final String TAG_B = "tag/b";
	private static final String TAG_C = "tag/c";
	private static final String SIGNED_LOG_TAG = "signed/log";
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
					assertFalse(isTagActive(page, TAG_A));
					assertEquals(List.of(TAG_A, TAG_C), readTagList(page));

					clickTag(page, TAG_A);
					clickTag(page, TAG_C);
					assertTrue(isTagActive(page, TAG_A));
					assertTrue(isTagActive(page, TAG_C));
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

	private static void waitForGraph(Page page) {
		page.waitForFunction("document.querySelectorAll('.js-plotly-plot path.js-line').length > 0",
				null, new Page.WaitForFunctionOptions().setTimeout(30000));
	}

	private static boolean isAutoReloadButtonActive(Page page) {
		return Boolean.TRUE.equals(page.evaluate("""
				() => document.getElementById('btn-auto-reload').classList.contains('active')
				"""));
	}

	private static String runsJson() {
		final StringBuilder sb = new StringBuilder();
		sb.append("{\"runs\":[");
		for (int i = 0; i < RUN_IDS.size(); i++) {
			if (i > 0) sb.append(',');
			sb.append("{\"id\":\"").append(RUN_IDS.get(i)).append("\",")
					.append("\"stats\":{\"maxStep\":2},")
					.append("\"tags\":[{\"key\":\"").append(TAG_KEY).append("\",\"type\":\"scalar\"}]}");
		}
		sb.append("]}");
		return sb.toString();
	}

	private static String manyTagRunsJson(int tagCount) {
		final StringBuilder sb = new StringBuilder();
		sb.append("{\"runs\":[{\"id\":\"run_mobile\",\"stats\":{\"maxStep\":2},\"tags\":[");
		for (int i = 0; i < tagCount; i++) {
			if (i > 0) sb.append(',');
			sb.append("{\"key\":\"mobile/tag_").append(String.format("%02d", i)).append("\",\"type\":\"scalar\"}");
		}
		sb.append("]}]}");
		return sb.toString();
	}

	private static String metricsJson() {
		final StringBuilder sb = new StringBuilder();
		sb.append("{\"data\":[");
		for (int i = 0; i < RUN_IDS.size(); i++) {
			if (i > 0) sb.append(',');
			final int baseValue = i + 1;
			sb.append("{\"runId\":\"").append(RUN_IDS.get(i)).append("\",")
					.append("\"tagKey\":\"").append(TAG_KEY).append("\",")
					.append("\"type\":\"scalar\",")
					.append("\"beginStep\":0,")
					.append("\"endStep\":2,")
					.append("\"steps\":[0,1,2],")
					.append("\"values\":[").append(baseValue).append(',')
					.append(baseValue + 1).append(',')
					.append(baseValue + 2).append("]}");
		}
		sb.append("]}");
		return sb.toString();
	}

	private static String signedLogRunsJson() {
		return """
				{"runs":[
					{"id":"run_signed","stats":{"maxStep":4},
						"tags":[{"key":"%s","type":"scalar"}]}
				]}
				""".formatted(SIGNED_LOG_TAG);
	}

	private static String signedLogMetricsJson() {
		return """
				{"data":[
					{"runId":"run_signed","tagKey":"%s","type":"scalar","beginStep":0,"endStep":4,
						"steps":[0,1,2,3,4],"values":[-100,-9,0,9,100]}
				]}
				""".formatted(SIGNED_LOG_TAG);
	}

	private static String signedLogZoomMetricsJson() {
		return """
				{"data":[
					{"runId":"run_signed","tagKey":"%s","type":"scalar","beginStep":0,"endStep":4,
						"steps":[0,1,2,3,4],"values":[-100,20,25,30,100]}
				]}
				""".formatted(SIGNED_LOG_TAG);
	}

	private static String splitTagRunsJson() {
		return """
				{"runs":[
					{"id":"run_a","stats":{"maxStep":2},"tags":[
						{"key":"tag/c","type":"scalar"},
						{"key":"tag/a","type":"scalar"}
					]},
					{"id":"run_b","stats":{"maxStep":2},"tags":[{"key":"tag/b","type":"scalar"}]}
				]}
				""";
	}

	private static String splitTagMetricsJson() {
		return """
				{"data":[
					{"runId":"run_a","tagKey":"tag/a","type":"scalar","beginStep":0,"endStep":2,
						"steps":[0,1,2],"values":[1,2,3]},
					{"runId":"run_a","tagKey":"tag/c","type":"scalar","beginStep":0,"endStep":2,
						"steps":[0,1,2],"values":[7,8,9]},
					{"runId":"run_b","tagKey":"tag/b","type":"scalar","beginStep":0,"endStep":2,
						"steps":[0,1,2],"values":[4,5,6]}
				]}
				""";
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
					runId: el.parentElement.querySelector('.run-check').value,
					color: getComputedStyle(el).backgroundColor
				}))
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
						.find(el => el.textContent.includes(runId));
					if (!row) throw new Error('run not found: ' + runId);
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
