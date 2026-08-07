package io.github.kazukin123.anetlab.metricsviewer.view;

import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.metricsJson;
import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.runsJson;
import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.signedLogMetricsJson;
import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.signedLogRunsJson;
import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.signedLogZoomMetricsJson;
import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

import com.microsoft.playwright.Page;
import com.microsoft.playwright.options.WaitUntilState;

@SpringBootTest(
		webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT,
		properties = "metricsviewer.runs-dir=target/playwright-test-empty-runs")
class SignedLogPlaywrightTest extends MetricsViewerPlaywrightTestSupport {

	@Test
	void logScaleTogglePersistsAcrossReloadsButNotPageRefresh() {
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
	}

	@Test
	void signedLogScaleKeepsNegativeZeroAndPositiveValues() {
		page.route("**/api/runs.json", route -> fulfillJson(route, signedLogRunsJson()));
		page.route("**/api/metrics.json", route -> fulfillJson(route, signedLogMetricsJson()));

		page.navigate(baseUrl + "/?signedLogScaleTest=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		waitForGraph(page);

		page.click(".graph-log-toggle");
		waitForSignedLogMixedSignTrace(page);
		assertEquals("linear", readYAxisType(page));
	}

	@Test
	void signedLogZoomKeepsYAxisTickLabelsVisible() {
		page.route("**/api/runs.json", route -> fulfillJson(route, signedLogRunsJson()));
		page.route("**/api/metrics.json", route -> fulfillJson(route, signedLogZoomMetricsJson()));

		page.navigate(baseUrl + "/?signedLogZoomTickTest=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		waitForGraph(page);

		page.click(".graph-log-toggle");
		waitForSignedLogZoomSourceTrace(page);
		zoomToSignedLogRange(page, 1, 3, 20, 30);
		waitForSignedLogZoomTicks(page);
	}

	@Test
	void signedLogZoomDoesNotResetPlotlyPanMode() {
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
	}
}
