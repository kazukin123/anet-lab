package io.github.kazukin123.anetlab.metricsviewer.view;

import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.constantOutlierMetricsJson;
import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.interleavedOutlierMetricsJson;
import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.metricsJson;
import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.multiRunOutlierMetricsJson;
import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.multiRunOutlierRunsJson;
import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.outlierLodMetricsJson;
import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.outlierLodRunsJson;
import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.outlierMetricsJson;
import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.outlierRunsJson;
import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.runsJson;
import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.signedOutlierMetricsJson;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

import com.microsoft.playwright.Page;
import com.microsoft.playwright.options.WaitUntilState;

@SpringBootTest(
		webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT,
		properties = "metricsviewer.workspaces-dir=target/playwright-test-empty-workspaces")
class OutlierRangePlaywrightTest extends MetricsViewerPlaywrightTestSupport {
	@Test
	void p5P95KeepsContinuousTraceAndClipsItAtYAxis() {
		page.route("**/api/runs.json", route -> fulfillJson(route, outlierRunsJson()));
		page.route("**/api/metrics.json", route -> fulfillJson(route, interleavedOutlierMetricsJson()));

		page.navigate(baseUrl + "/?outlierContinuityTest=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		waitForGraph(page);
		page.click(".graph-outlier-toggle");
		waitForP5P95YAxisRange(page, 6, 14, 101);

		assertEquals(1, page.locator(".js-plotly-plot path.js-line").count());
		assertTrue(Boolean.TRUE.equals(page.evaluate(
				"() => Array.from(document.querySelector('.js-plotly-plot').data[0].y)"
						+ ".includes(30)")));
	}

	@Test
	void p5P95KeepsOutliersInTraceAndLimitsYAxisRange() {
		page.route("**/api/runs.json", route -> fulfillJson(route, outlierRunsJson()));
		page.route("**/api/metrics.json", route -> fulfillJson(route, outlierMetricsJson()));

		page.navigate(baseUrl + "/?outlierRangeTest=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		waitForGraph(page);

		assertEquals(
				List.of("Log", "p5–p95", "p1–p99"),
				page.locator(".graph-header button").allTextContents());
		assertEquals("false", page.getAttribute(".graph-outlier-toggle", "aria-pressed"));
		assertEquals("false", page.getAttribute(".graph-wide-outlier-toggle", "aria-pressed"));
		page.click(".graph-outlier-toggle");
		waitForP5P95YAxisRange(page, 5, 95, 101);

		assertEquals("true", page.getAttribute(".graph-outlier-toggle", "aria-pressed"));
		assertTrue(Boolean.TRUE.equals(page.evaluate(
				"() => Array.from(document.querySelector('.js-plotly-plot').data[0].y).includes(1000)")));
	}

	@Test
	void p1P99IsMutuallyExclusiveWithP5P95AndPersists() {
		page.route("**/api/runs.json", route -> fulfillJson(route, outlierRunsJson()));
		page.route("**/api/metrics.json", route -> fulfillJson(route, outlierMetricsJson()));

		page.navigate(baseUrl + "/?wideOutlierRangeTest=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		waitForGraph(page);

		page.click(".graph-outlier-toggle");
		waitForP5P95YAxisRange(page, 5, 95, 101);
		page.click(".graph-wide-outlier-toggle");
		waitForP5P95YAxisRange(page, 1, 99, 101);
		assertEquals("false", page.getAttribute(".graph-outlier-toggle", "aria-pressed"));
		assertEquals("true", page.getAttribute(".graph-wide-outlier-toggle", "aria-pressed"));
		assertEquals(
				"Display each Run's p1–p99 points (99/101 visible points)",
				page.getAttribute(".graph-wide-outlier-toggle", "title"));
		assertEquals("[]", page.evaluate(
				"() => localStorage.getItem('anet.metricsviewer.ignoreOutlierTags')"));
		assertEquals("[\"outlier/range\"]", page.evaluate(
				"() => localStorage.getItem('anet.metricsviewer.p1P99Tags')"));

		page.reload(new Page.ReloadOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		waitForGraph(page);
		waitForP5P95YAxisRange(page, 1, 99, 101);
		assertEquals("false", page.getAttribute(".graph-outlier-toggle", "aria-pressed"));
		assertEquals("true", page.getAttribute(".graph-wide-outlier-toggle", "aria-pressed"));

		page.click(".graph-outlier-toggle");
		waitForP5P95YAxisRange(page, 5, 95, 101);
		assertEquals("true", page.getAttribute(".graph-outlier-toggle", "aria-pressed"));
		assertEquals("false", page.getAttribute(".graph-wide-outlier-toggle", "aria-pressed"));
		assertEquals("[]", page.evaluate(
				"() => localStorage.getItem('anet.metricsviewer.p1P99Tags')"));
	}

	@Test
	void p5P95FiltersSmallSeriesAndStaysEnabled() {
		page.route("**/api/runs.json", route -> fulfillJson(route, runsJson()));
		page.route("**/api/metrics.json", route -> fulfillJson(route, metricsJson()));

		page.navigate(baseUrl + "/?outlierThresholdTest=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		waitForGraph(page);
		page.click(".graph-outlier-toggle");

		waitForP5P95YAxisRange(page, 11.1, 12.9, 3);
		assertEquals("true", page.getAttribute(".graph-outlier-toggle", "aria-pressed"));
		assertEquals(
				"Display each Run's p5–p95 points (1/3 visible points)",
				page.getAttribute(".graph-outlier-toggle", "title"));
	}

	@Test
	void hiddenLegendRunIsExcludedAndResetViewRestoresTheCombinedRange() {
		page.route("**/api/runs.json", route -> fulfillJson(route, multiRunOutlierRunsJson()));
		page.route("**/api/metrics.json", route -> fulfillJson(route, multiRunOutlierMetricsJson()));

		page.navigate(baseUrl + "/?outlierLegendTest=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		waitForGraph(page);
		page.click("#btn-select-all-runs");
		waitForMultiRunGraph(page);
		page.click(".graph-outlier-toggle");
		waitForP5P95YAxisRange(page, 5, 195, 202);

		clickLegendSeries(page, MetricsViewerPlaywrightTestData.OUTLIER_TAG, "run_outlier_b");
		waitForP5P95YAxisRange(page, 5, 95, 101);

		page.click("#btn-reset-view");
		waitForSeriesTrace(page, MetricsViewerPlaywrightTestData.OUTLIER_TAG, "run_outlier_b", true);
		waitForP5P95YAxisRange(page, 5, 195, 202);
	}

	@Test
	void logAndP5P95PersistIndependentlyAndComposeAfterPageRefresh() {
		page.route("**/api/runs.json", route -> fulfillJson(route, outlierRunsJson()));
		page.route("**/api/metrics.json", route -> fulfillJson(route, outlierMetricsJson()));

		page.navigate(baseUrl + "/?outlierStorageTest=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		waitForGraph(page);
		page.click(".graph-outlier-toggle");
		waitForP5P95YAxisRange(page, 5, 95, 101);
		assertEquals("[]", page.evaluate(
				"() => localStorage.getItem('anet.metricsviewer.logScaleTags')"));
		assertEquals("[\"outlier/range\"]", page.evaluate(
				"() => localStorage.getItem('anet.metricsviewer.ignoreOutlierTags')"));

		page.reload(new Page.ReloadOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		waitForGraph(page);
		waitForP5P95YAxisRange(page, 5, 95, 101);
		assertEquals("false", page.getAttribute(".graph-log-toggle", "aria-pressed"));
		assertEquals("true", page.getAttribute(".graph-outlier-toggle", "aria-pressed"));

		page.click(".graph-log-toggle");
		waitForP5P95YAxisRange(page, Math.log10(6), Math.log10(96), 101);
		assertEquals("[\"outlier/range\"]", page.evaluate(
				"() => localStorage.getItem('anet.metricsviewer.logScaleTags')"));

		page.reload(new Page.ReloadOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		waitForGraph(page);
		waitForP5P95YAxisRange(page, Math.log10(6), Math.log10(96), 101);
		assertEquals("true", page.getAttribute(".graph-log-toggle", "aria-pressed"));
		assertEquals("true", page.getAttribute(".graph-outlier-toggle", "aria-pressed"));
	}

	@Test
	void signedLogP5P95HandlesNegativeValuesAndZero() {
		page.route("**/api/runs.json", route -> fulfillJson(route, outlierRunsJson()));
		page.route("**/api/metrics.json", route -> fulfillJson(route, signedOutlierMetricsJson()));

		page.navigate(baseUrl + "/?signedOutlierRangeTest=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		waitForGraph(page);
		page.click(".graph-outlier-toggle");
		page.click(".graph-log-toggle");

		waitForP5P95YAxisRange(page, -Math.log10(46), Math.log10(46), 101);
		assertTrue(Boolean.TRUE.equals(page.evaluate(
				"() => Array.from(document.querySelector('.js-plotly-plot').data[0].customdata)"
						+ ".includes(0)")));
	}

	@Test
	void equalPercentilesUsePlotlyAutorange() {
		page.route("**/api/runs.json", route -> fulfillJson(route, outlierRunsJson()));
		page.route("**/api/metrics.json", route -> fulfillJson(route, constantOutlierMetricsJson()));

		page.navigate(baseUrl + "/?constantOutlierRangeTest=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		waitForGraph(page);
		page.click(".graph-outlier-toggle");

		waitForP5P95YAxisRange(page, 7, 7, 101);
	}

	@Test
	void invalidGraphDisplayStorageFallsBackToOff() {
		context.addInitScript("""
				localStorage.setItem('anet.metricsviewer.logScaleTags', '{}');
				localStorage.setItem('anet.metricsviewer.ignoreOutlierTags', '[1]');
				localStorage.setItem('anet.metricsviewer.p1P99Tags', '[false]');
				""");
		page.route("**/api/runs.json", route -> fulfillJson(route, outlierRunsJson()));
		page.route("**/api/metrics.json", route -> fulfillJson(route, outlierMetricsJson()));

		page.navigate(baseUrl + "/?invalidGraphStorageTest=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		waitForGraph(page);

		assertEquals("false", page.getAttribute(".graph-log-toggle", "aria-pressed"));
		assertEquals("false", page.getAttribute(".graph-outlier-toggle", "aria-pressed"));
		assertEquals("false", page.getAttribute(".graph-wide-outlier-toggle", "aria-pressed"));
	}

	@Test
	void p5P95UsesOnlyPointsInsideTheCurrentXRange() {
		page.route("**/api/runs.json", route -> fulfillJson(route, outlierRunsJson()));
		page.route("**/api/metrics.json", route -> fulfillJson(route, outlierMetricsJson()));

		page.navigate(baseUrl + "/?outlierViewportTest=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		waitForGraph(page);
		page.click(".graph-outlier-toggle");
		waitForP5P95YAxisRange(page, 5, 95, 101);

		page.evaluate("""
				() => Plotly.relayout(document.querySelector('.js-plotly-plot'), {
					'xaxis.range': [0, 49]
				})
				""");
		waitForP5P95YAxisRange(page, 2.45, 46.55, 101);
		assertEquals(
				"Display each Run's p5–p95 points (44/50 visible points)",
				page.getAttribute(".graph-outlier-toggle", "title"));
	}

	@Test
	void manualYZoomSurvivesRedrawAndAxisResetReturnsToP5P95() {
		page.route("**/api/runs.json", route -> fulfillJson(route, outlierRunsJson()));
		page.route("**/api/metrics.json", route -> fulfillJson(route, outlierMetricsJson()));

		page.navigate(baseUrl + "/?outlierManualZoomTest=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		waitForGraph(page);
		page.click(".graph-outlier-toggle");
		waitForP5P95YAxisRange(page, 5, 95, 101);

		page.evaluate("""
				() => Plotly.relayout(document.querySelector('.js-plotly-plot'), {
					'yaxis.range': [20, 30]
				})
				""");
		waitForYAxisRange(page, 20, 30);
		page.selectOption("#lod-display-mode", "Mean");
		waitForYAxisRange(page, 20, 30);

		page.evaluate("""
				() => Plotly.relayout(document.querySelector('.js-plotly-plot'), {
					'yaxis.autorange': true
				})
				""");
		waitForP5P95YAxisRange(page, 5, 95, 101);
	}

	@Test
	void p5P95FollowsTheCurrentLodDisplayValues() {
		page.route("**/api/runs.json", route -> fulfillJson(route, outlierLodRunsJson()));
		page.route("**/api/metrics.json", route -> fulfillJson(route, outlierLodMetricsJson()));

		page.navigate(baseUrl + "/?outlierLodTest=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		waitForGraph(page);
		page.click(".graph-outlier-toggle");
		waitForP5P95YAxisRange(page, 5, 95, 101);
		assertEquals(
				"Display each Run's p5–p95 points (91/101 visible points)",
				page.getAttribute(".graph-outlier-toggle", "title"));

		page.selectOption("#lod-display-mode", "Mean");
		waitForP5P95YAxisRange(page, 9.95, 99.05, 100);
		assertEquals(
				"Display each Run's p5–p95 points (90/100 visible points)",
				page.getAttribute(".graph-outlier-toggle", "title"));

		page.selectOption("#lod-display-mode", "Band");
		waitForP5P95YAxisRange(page, 9.95, 99.05, 300);
		assertEquals(
				"Display each Run's p5–p95 points (270/300 visible points)",
				page.getAttribute(".graph-outlier-toggle", "title"));
	}

	private static void waitForP5P95YAxisRange(Page page, double min, double max, int count) {
		page.waitForFunction("""
				([min, max, count]) => {
					const plot = document.querySelector('.js-plotly-plot');
					const values = (plot?.data ?? [])
						.filter(trace => trace.visible !== 'legendonly' && trace.visible !== false)
						.flatMap(trace => Array.from(trace.y ?? []).filter(Number.isFinite));
					const range = plot?._fullLayout?.yaxis?.range;
					const rangeMatches = min === max
							? range[0] < min && range[1] > max
							: Math.abs(range[0] - min) < 1e-5 && Math.abs(range[1] - max) < 1e-5;
					return plot?._fullLayout?.yaxis?.autorange === false
						&& values.length === count
						&& rangeMatches;
				}
				""", List.of(min, max, count));
	}

	private static void waitForYAxisRange(Page page, double min, double max) {
		page.waitForFunction("""
				([min, max]) => {
					const range = document.querySelector('.js-plotly-plot')?._fullLayout?.yaxis?.range;
					return Array.isArray(range)
						&& Math.abs(range[0] - min) < 1e-6
						&& Math.abs(range[1] - max) < 1e-6;
				}
				""", List.of(min, max));
	}
}
