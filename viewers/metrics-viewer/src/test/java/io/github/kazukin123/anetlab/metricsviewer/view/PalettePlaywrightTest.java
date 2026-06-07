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

					page.click(".graph-log-toggle");
					page.waitForFunction("document.querySelector('.js-plotly-plot')._fullLayout.yaxis.type === 'log'",
							null, new Page.WaitForFunctionOptions().setTimeout(30000));
					assertEquals("log", readYAxisType(page));

					page.click("#btn-reload");
					waitForGraph(page);
					page.waitForFunction("document.querySelector('.js-plotly-plot')._fullLayout.yaxis.type === 'log'",
							null, new Page.WaitForFunctionOptions().setTimeout(30000));
					assertEquals("log", readYAxisType(page));

					page.click("#btn-auto-reload");
					page.evaluate("app.onReload()");
					waitForGraph(page);
					page.waitForFunction("document.querySelector('.js-plotly-plot')._fullLayout.yaxis.type === 'log'",
							null, new Page.WaitForFunctionOptions().setTimeout(30000));
					assertEquals("log", readYAxisType(page));

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

	private static String readYAxisType(Page page) {
		return (String) page.evaluate("""
				(() => {
					const plot = document.querySelector('.js-plotly-plot');
					return plot ? (plot._fullLayout?.yaxis?.type || plot.layout?.yaxis?.type || 'linear') : '';
				})()
				""");
	}

	private static String hexToRgb(String hex) {
		final int r = Integer.parseInt(hex.substring(1, 3), 16);
		final int g = Integer.parseInt(hex.substring(3, 5), 16);
		final int b = Integer.parseInt(hex.substring(5, 7), 16);
		return "rgb(" + r + ", " + g + ", " + b + ")";
	}
}
