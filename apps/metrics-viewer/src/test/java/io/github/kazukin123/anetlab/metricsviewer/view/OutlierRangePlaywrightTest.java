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
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

import com.microsoft.playwright.Browser;
import com.microsoft.playwright.Page;
import com.microsoft.playwright.options.WaitUntilState;

@SpringBootTest(
		webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT,
		properties = "metricsviewer.workspaces-dir=target/playwright-test-empty-workspaces")
class OutlierRangePlaywrightTest extends MetricsViewerPlaywrightTestSupport {
	private static final String LOWER = ".graph-lower-percentile";
	private static final String UPPER = ".graph-upper-percentile";
	private static final List<String> LOWER_LABELS = List.of("p0–", "p1–", "p5–");
	private static final List<String> UPPER_LABELS = List.of("–p100", "–p99", "–p95");
	private static final String PERCENTILE_STORAGE = "anet.metricsviewer.percentileBounds";

	@Test
	void graphHeaderTogglesKeepLabelsOnOneLineWhenHeaderIsNarrow() {
		// headerが狭いとボタンが押し潰され、–p100のdashでラベルが割れる。
		reopenPage(new Browser.NewContextOptions().setViewportSize(620, 720));
		page.route("**/api/runs.json", route -> fulfillJson(route, outlierRunsJson()));
		page.route("**/api/metrics.json", route -> fulfillJson(route, outlierMetricsJson()));

		page.navigate(baseUrl + "/?narrowGraphHeaderTest=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		waitForGraph(page);

		assertEquals("Log:ok|p0–:ok|–p100:ok", page.evaluate("""
				() => [...document.querySelectorAll('.graph-header button')]
					.map(el => el.textContent + ':'
						+ (el.scrollHeight > el.clientHeight + 1
							|| el.scrollWidth > el.clientWidth + 1 ? 'wrapped' : 'ok'))
					.join('|')
				"""));
	}

	@Test
	void percentileButtonsKeepTheirSizeAndPositionWhileCycling() {
		// ラベルの文字数が変わってもボタン幅が動くと、後ろの要素と次に押す位置がずれる。
		page.route("**/api/runs.json", route -> fulfillJson(route, outlierRunsJson()));
		page.route("**/api/metrics.json", route -> fulfillJson(route, outlierMetricsJson()));

		page.navigate(baseUrl + "/?percentileButtonGeometryTest=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		waitForGraph(page);

		final String initial = headerGeometry(page);
		for (int i = 1; i <= LOWER_LABELS.size(); i++) {
			clickAndWaitForLabel(page, LOWER, LOWER_LABELS.get(i % LOWER_LABELS.size()));
			assertEquals(initial, headerGeometry(page));
		}
		for (int i = 1; i <= UPPER_LABELS.size(); i++) {
			clickAndWaitForLabel(page, UPPER, UPPER_LABELS.get(i % UPPER_LABELS.size()));
			assertEquals(initial, headerGeometry(page));
		}
	}

	@Test
	void p5P95KeepsContinuousTraceAndClipsItAtYAxis() {
		page.route("**/api/runs.json", route -> fulfillJson(route, outlierRunsJson()));
		page.route("**/api/metrics.json", route -> fulfillJson(route, interleavedOutlierMetricsJson()));

		page.navigate(baseUrl + "/?outlierContinuityTest=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		waitForGraph(page);
		selectPercentileBounds(page, 5, 95);
		waitForPercentileYAxisRange(page, 6, 14, 101);

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
				List.of("Log", "p0–", "–p100"),
				page.locator(".graph-header button").allTextContents());
		assertEquals("false", page.getAttribute(LOWER, "aria-pressed"));
		assertEquals("false", page.getAttribute(UPPER, "aria-pressed"));
		selectPercentileBounds(page, 5, 95);
		waitForPercentileYAxisRange(page, 5, 95, 101);

		assertEquals("true", page.getAttribute(LOWER, "aria-pressed"));
		assertEquals("true", page.getAttribute(UPPER, "aria-pressed"));
		assertTrue(Boolean.TRUE.equals(page.evaluate(
				"() => Array.from(document.querySelector('.js-plotly-plot').data[0].y).includes(1000)")));
	}

	@Test
	void lowerAndUpperBoundsCycleIndependentlyAndPersist() {
		page.route("**/api/runs.json", route -> fulfillJson(route, outlierRunsJson()));
		page.route("**/api/metrics.json", route -> fulfillJson(route, outlierMetricsJson()));

		page.navigate(baseUrl + "/?percentileCycleTest=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		waitForGraph(page);

		// 下限だけを上げると、最大値1000はY軸に残る。
		clickAndWaitForLabel(page, LOWER, "p1–");
		waitForPercentileYAxisRange(page, 1, 1000, 101);
		assertEquals("true", page.getAttribute(LOWER, "aria-pressed"));
		assertEquals("false", page.getAttribute(UPPER, "aria-pressed"));
		assertEquals(
				"Display each Run's p1–p100 points (100/101 visible points)",
				page.getAttribute(LOWER, "title"));
		assertEquals(
				"Limit the Y-axis upper bound to each Run's percentile (p100 → p99 → p95)",
				page.getAttribute(UPPER, "title"));
		clickAndWaitForLabel(page, LOWER, "p5–");
		waitForPercentileYAxisRange(page, 5, 1000, 101);
		assertEquals("{\"outlier/range\":[5,100]}", percentileStorage(page));

		page.reload(new Page.ReloadOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		waitForGraph(page);
		waitForPercentileYAxisRange(page, 5, 1000, 101);
		assertEquals(List.of("Log", "p5–", "–p100"), page.locator(".graph-header button").allTextContents());
		assertEquals("true", page.getAttribute(LOWER, "aria-pressed"));
		assertEquals("false", page.getAttribute(UPPER, "aria-pressed"));

		clickAndWaitForLabel(page, UPPER, "–p99");
		waitForPercentileYAxisRange(page, 5, 99, 101);
		clickAndWaitForLabel(page, UPPER, "–p95");
		waitForPercentileYAxisRange(page, 5, 95, 101);
		assertEquals("{\"outlier/range\":[5,95]}", percentileStorage(page));

		// 下限を一周させて制限なしへ戻すと、上限だけが残り最小値0がY軸に入る。
		clickAndWaitForLabel(page, LOWER, "p0–");
		waitForPercentileYAxisRange(page, 0, 95, 101);
		assertEquals("false", page.getAttribute(LOWER, "aria-pressed"));
		assertEquals("true", page.getAttribute(UPPER, "aria-pressed"));
		assertEquals(
				"Display each Run's p0–p95 points (96/101 visible points)",
				page.getAttribute(UPPER, "title"));
		assertEquals(
				"Limit the Y-axis lower bound to each Run's percentile (p0 → p1 → p5)",
				page.getAttribute(LOWER, "title"));
		assertEquals("{\"outlier/range\":[0,95]}", percentileStorage(page));

		clickAndWaitForLabel(page, UPPER, "–p100");
		page.waitForFunction(
				"() => document.querySelector('.js-plotly-plot')?._fullLayout?.yaxis?.autorange === true");
		assertEquals("false", page.getAttribute(UPPER, "aria-pressed"));
		assertEquals("{}", percentileStorage(page));
	}

	@Test
	void p5P95FiltersSmallSeriesAndStaysEnabled() {
		page.route("**/api/runs.json", route -> fulfillJson(route, runsJson()));
		page.route("**/api/metrics.json", route -> fulfillJson(route, metricsJson()));

		page.navigate(baseUrl + "/?outlierThresholdTest=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		waitForGraph(page);
		selectPercentileBounds(page, 5, 95);

		waitForPercentileYAxisRange(page, 11.1, 12.9, 3);
		assertEquals("true", page.getAttribute(LOWER, "aria-pressed"));
		assertEquals(
				"Display each Run's p5–p95 points (1/3 visible points)",
				page.getAttribute(LOWER, "title"));
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
		selectPercentileBounds(page, 5, 95);
		waitForPercentileYAxisRange(page, 5, 195, 202);

		clickLegendSeries(page, MetricsViewerPlaywrightTestData.OUTLIER_TAG, "run_outlier_b");
		waitForPercentileYAxisRange(page, 5, 95, 101);

		page.click("#btn-reset-view");
		waitForSeriesTrace(page, MetricsViewerPlaywrightTestData.OUTLIER_TAG, "run_outlier_b", true);
		waitForPercentileYAxisRange(page, 5, 195, 202);
	}

	@Test
	void logAndPercentileBoundsPersistIndependentlyAndComposeAfterPageRefresh() {
		page.route("**/api/runs.json", route -> fulfillJson(route, outlierRunsJson()));
		page.route("**/api/metrics.json", route -> fulfillJson(route, outlierMetricsJson()));

		page.navigate(baseUrl + "/?outlierStorageTest=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		waitForGraph(page);
		selectPercentileBounds(page, 5, 95);
		waitForPercentileYAxisRange(page, 5, 95, 101);
		assertEquals("[]", page.evaluate(
				"() => localStorage.getItem('anet.metricsviewer.logScaleTags')"));
		assertEquals("{\"outlier/range\":[5,95]}", percentileStorage(page));

		page.reload(new Page.ReloadOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		waitForGraph(page);
		waitForPercentileYAxisRange(page, 5, 95, 101);
		assertEquals("false", page.getAttribute(".graph-log-toggle", "aria-pressed"));
		assertEquals("true", page.getAttribute(LOWER, "aria-pressed"));
		assertEquals("true", page.getAttribute(UPPER, "aria-pressed"));

		page.click(".graph-log-toggle");
		waitForPercentileYAxisRange(page, Math.log10(6), Math.log10(96), 101);
		assertEquals("[\"outlier/range\"]", page.evaluate(
				"() => localStorage.getItem('anet.metricsviewer.logScaleTags')"));

		page.reload(new Page.ReloadOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		waitForGraph(page);
		waitForPercentileYAxisRange(page, Math.log10(6), Math.log10(96), 101);
		assertEquals("true", page.getAttribute(".graph-log-toggle", "aria-pressed"));
		assertEquals("true", page.getAttribute(LOWER, "aria-pressed"));
		assertEquals("true", page.getAttribute(UPPER, "aria-pressed"));
	}

	@Test
	void signedLogP5P95HandlesNegativeValuesAndZero() {
		page.route("**/api/runs.json", route -> fulfillJson(route, outlierRunsJson()));
		page.route("**/api/metrics.json", route -> fulfillJson(route, signedOutlierMetricsJson()));

		page.navigate(baseUrl + "/?signedOutlierRangeTest=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		waitForGraph(page);
		selectPercentileBounds(page, 5, 95);
		page.click(".graph-log-toggle");

		waitForPercentileYAxisRange(page, -Math.log10(46), Math.log10(46), 101);
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
		selectPercentileBounds(page, 5, 95);

		waitForPercentileYAxisRange(page, 7, 7, 101);
	}

	@Test
	void invalidGraphDisplayStorageFallsBackToOff() {
		context.addInitScript("""
				localStorage.setItem('anet.metricsviewer.logScaleTags', '{}');
				localStorage.setItem('anet.metricsviewer.percentileBounds', '{"outlier/range":[2,100]}');
				""");
		page.route("**/api/runs.json", route -> fulfillJson(route, outlierRunsJson()));
		page.route("**/api/metrics.json", route -> fulfillJson(route, outlierMetricsJson()));

		page.navigate(baseUrl + "/?invalidGraphStorageTest=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		waitForGraph(page);

		assertEquals("false", page.getAttribute(".graph-log-toggle", "aria-pressed"));
		assertEquals("false", page.getAttribute(LOWER, "aria-pressed"));
		assertEquals("false", page.getAttribute(UPPER, "aria-pressed"));
	}

	@Test
	void discardedOutlierStorageKeysAreRemovedWithoutBeingRead() {
		// p5–p95 / p1–p99の択一トグルだった頃の保存値は引き継がない。
		context.addInitScript("""
				localStorage.setItem('anet.metricsviewer.ignoreOutlierTags', '["outlier/range"]');
				localStorage.setItem('anet.metricsviewer.p1P99Tags', '["outlier/range"]');
				""");
		page.route("**/api/runs.json", route -> fulfillJson(route, outlierRunsJson()));
		page.route("**/api/metrics.json", route -> fulfillJson(route, outlierMetricsJson()));

		page.navigate(baseUrl + "/?discardedOutlierStorageTest=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		waitForGraph(page);

		assertEquals("false", page.getAttribute(LOWER, "aria-pressed"));
		assertEquals("false", page.getAttribute(UPPER, "aria-pressed"));
		assertNull(page.evaluate("() => localStorage.getItem('anet.metricsviewer.ignoreOutlierTags')"));
		assertNull(page.evaluate("() => localStorage.getItem('anet.metricsviewer.p1P99Tags')"));
	}

	@Test
	void p5P95UsesOnlyPointsInsideTheCurrentXRange() {
		page.route("**/api/runs.json", route -> fulfillJson(route, outlierRunsJson()));
		page.route("**/api/metrics.json", route -> fulfillJson(route, outlierMetricsJson()));

		page.navigate(baseUrl + "/?outlierViewportTest=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		waitForGraph(page);
		selectPercentileBounds(page, 5, 95);
		waitForPercentileYAxisRange(page, 5, 95, 101);

		page.evaluate("""
				() => Plotly.relayout(document.querySelector('.js-plotly-plot'), {
					'xaxis.range': [0, 49]
				})
				""");
		waitForPercentileYAxisRange(page, 2.45, 46.55, 101);
		assertEquals(
				"Display each Run's p5–p95 points (44/50 visible points)",
				page.getAttribute(LOWER, "title"));
	}

	@Test
	void manualYZoomSurvivesRedrawAndAxisResetReturnsToP5P95() {
		page.route("**/api/runs.json", route -> fulfillJson(route, outlierRunsJson()));
		page.route("**/api/metrics.json", route -> fulfillJson(route, outlierMetricsJson()));

		page.navigate(baseUrl + "/?outlierManualZoomTest=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		waitForGraph(page);
		selectPercentileBounds(page, 5, 95);
		waitForPercentileYAxisRange(page, 5, 95, 101);

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
		waitForPercentileYAxisRange(page, 5, 95, 101);
	}

	@Test
	void p5P95FollowsTheCurrentLodDisplayValues() {
		page.route("**/api/runs.json", route -> fulfillJson(route, outlierLodRunsJson()));
		page.route("**/api/metrics.json", route -> fulfillJson(route, outlierLodMetricsJson()));

		page.navigate(baseUrl + "/?outlierLodTest=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		waitForGraph(page);
		selectPercentileBounds(page, 5, 95);
		waitForPercentileYAxisRange(page, 5, 95, 101);
		assertEquals(
				"Display each Run's p5–p95 points (91/101 visible points)",
				page.getAttribute(LOWER, "title"));

		page.selectOption("#lod-display-mode", "Mean");
		waitForPercentileYAxisRange(page, 9.95, 99.05, 100);
		assertEquals(
				"Display each Run's p5–p95 points (90/100 visible points)",
				page.getAttribute(LOWER, "title"));

		page.selectOption("#lod-display-mode", "Band");
		waitForPercentileYAxisRange(page, 9.95, 99.05, 300);
		assertEquals(
				"Display each Run's p5–p95 points (270/300 visible points)",
				page.getAttribute(LOWER, "title"));
	}

	// 下限・上限ボタンを押し回して指定の段にする。押すたびに次の段のラベルへ変わるのを待つ。
	private static void selectPercentileBounds(Page page, int lower, int upper) {
		cycleTo(page, LOWER, LOWER_LABELS, "p" + lower + "–");
		cycleTo(page, UPPER, UPPER_LABELS, "–p" + upper);
	}

	private static void cycleTo(Page page, String selector, List<String> labels, String target) {
		assertTrue(labels.contains(target), target);
		int index = labels.indexOf(page.textContent(selector));
		while (!labels.get(index).equals(target)) {
			index = (index + 1) % labels.size();
			clickAndWaitForLabel(page, selector, labels.get(index));
		}
	}

	// 押すとgraph blockごと作り直されるので、新しいボタンのラベルで反映を待つ。
	private static void clickAndWaitForLabel(Page page, String selector, String label) {
		page.click(selector);
		page.waitForFunction(
				"([selector, label]) => document.querySelector(selector)?.textContent === label",
				List.of(selector, label));
	}

	private static String headerGeometry(Page page) {
		return (String) page.evaluate("""
				() => {
					const rect = selector => {
						const r = document.querySelector(selector)?.getBoundingClientRect();
						return r ? `${r.left},${r.top},${r.width},${r.height}` : 'none';
					};
					return [
						rect('.graph-lower-percentile'),
						rect('.graph-upper-percentile'),
						rect('.graph-header > .graph-stats')
					].join('|');
				}
				""");
	}

	private static Object percentileStorage(Page page) {
		return page.evaluate("key => localStorage.getItem(key)", PERCENTILE_STORAGE);
	}

	private static void waitForPercentileYAxisRange(Page page, double min, double max, int count) {
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
