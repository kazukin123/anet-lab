package io.github.kazukin123.anetlab.metricsviewer.view;

import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.GENERATION;
import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.RUN_IDS;
import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.TAG_A;
import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.TAG_B;
import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.TAG_KEY;
import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.legendStateLodMetricsJson;
import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.legendStateMetricsJson;
import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.legendStateRunsJson;
import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.lodMetricsJson;
import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.metricsJson;
import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.rawSeriesJson;
import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.runJson;
import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.runJsonWithGeneration;
import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.runsJson;
import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.statsWarningMetricsJson;
import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.statsWarningRunsJson;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

import com.fasterxml.jackson.databind.JsonNode;
import com.microsoft.playwright.Page;
import com.microsoft.playwright.options.WaitUntilState;

@SpringBootTest(
		webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT,
		properties = "metricsviewer.runs-dir=target/playwright-test-empty-runs")
class MetricsPlotPlaywrightTest extends MetricsViewerPlaywrightTestSupport {

	@Test
	void paletteColorsRenderForRunChipsAndInitialTrace() throws Exception {
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
		assertEquals(RUN_IDS.size(), chips.stream().map(chip -> chip.get("color")).distinct().count());
		assertTrue(chips.stream().allMatch(chip -> chip.get("color").startsWith("rgb(")));

		final String selectedRunId = readActiveRunIds(page).get(0);
		final String selectedChipColor = chips.stream()
				.filter(chip -> selectedRunId.equals(chip.get("runId")))
				.findFirst()
				.orElseThrow()
				.get("color");
		assertEquals(selectedChipColor, readTraceColor(page, selectedRunId));
	}

	@Test
	void plotWidthDoesNotCreateHorizontalMainAreaOverflow() throws Exception {
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
	}

	@Test
	void lodDisplayModesReuseOneProjectionAndPersistTheSelection() {
		final AtomicInteger metricsRequests = new AtomicInteger();
		page.route("**/api/runs.json", route -> fulfillJson(
				route,
				"{\"runs\":[" + runJson("run_lod_ui", 31, TAG_KEY) + "]}"));
		page.route("**/api/metrics.json", route -> {
			metricsRequests.incrementAndGet();
			fulfillJson(route, lodMetricsJson());
		});
		page.route("**/api/runs/prioritize", MetricsViewerPlaywrightTestSupport::fulfillNoContent);

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
	}

	@Test
	void legendVisibilitySurvivesRedrawsAndResetViewShowsAllAndAutoscales() {
		final AtomicInteger metricsRequests = new AtomicInteger();
		page.route("**/api/runs.json", route -> fulfillJson(route, legendStateRunsJson()));
		page.route("**/api/metrics.json", route -> {
			metricsRequests.incrementAndGet();
			fulfillJson(route, legendStateMetricsJson());
		});
		page.route("**/api/runs/prioritize", MetricsViewerPlaywrightTestSupport::fulfillNoContent);

		page.navigate(baseUrl + "/?legendStateTest=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		page.click("#btn-select-all-runs");
		waitForGraphCount(page, 2);
		waitForSeriesTrace(page, TAG_A, "run_a", true);

		clickLegendSeries(page, TAG_A, "run_a");
		waitForSeriesTrace(page, TAG_A, "run_a", false);
		assertTrue(isSeriesVisible(page, TAG_B, "run_a"));

		final int requestsBeforeRangeResponse = metricsRequests.get();
		page.evaluate("""
				() => Plotly.relayout(
					document.getElementById(graphId('tag/a')),
					{'xaxis.range': [0.5, 1.5]})
				""");
		waitForSeriesTrace(page, TAG_A, "run_a", false);
		page.waitForCondition(() -> metricsRequests.get() > requestsBeforeRangeResponse);
		waitForSeriesTrace(page, TAG_A, "run_a", false);

		final int requestsBeforeReload = metricsRequests.get();
		page.click("#btn-reload");
		page.waitForCondition(() -> metricsRequests.get() > requestsBeforeReload);
		waitForSeriesTrace(page, TAG_A, "run_a", false);

		page.evaluate("""
				() => document.getElementById(graphId('tag/a'))
					.closest('.graph-block').querySelector('.graph-log-toggle').click()
				""");
		waitForSeriesTrace(page, TAG_A, "run_a", false);
		page.selectOption("#lod-display-mode", "Mean");
		waitForSeriesTrace(page, TAG_A, "run_a", false);

		page.evaluate("""
				async () => {
					await app.refreshMetadata({requestData: false});
					await app.requestVisibleData({force: true, followOnly: true});
				}
				""");
		waitForSeriesTrace(page, TAG_A, "run_a", false);

		final int requestsBeforeAxes = metricsRequests.get();
		page.evaluate("""
				async () => Promise.all([
					Plotly.relayout(document.getElementById(graphId('tag/a')), {
						'xaxis.range': [0.5, 1.5],
						'yaxis.range': [0.5, 1.5],
						dragmode: 'pan'
					}),
					Plotly.relayout(document.getElementById(graphId('tag/b')), {
						'xaxis.range': [1, 1.5],
						'yaxis.range': [4.5, 5.5]
					})
				])
				""");
		page.waitForCondition(() -> metricsRequests.get() > requestsBeforeAxes);
		final int requestsBeforeReset = metricsRequests.get();
		page.evaluate("""
				() => {
					const main = document.getElementById('main-area');
					main.style.flex = 'none';
					main.style.height = '400px';
					main.scrollTop = 100;
				}
				""");
		page.click("#btn-reset-view");
		waitForSeriesTrace(page, TAG_A, "run_a", true);
		page.waitForCondition(() -> metricsRequests.get() >= requestsBeforeReset + 1);
		assertEquals(requestsBeforeReset + 1, metricsRequests.get());
		assertEquals(true, page.evaluate("""
				() => ['tag/a', 'tag/b'].every(tagKey => {
					const plot = document.getElementById(graphId(tagKey));
					return app.explicitViewport(tagKey) === null
						&& plot._fullLayout.xaxis.autorange === true
						&& plot._fullLayout.yaxis.autorange === true;
				})
				"""));
		assertEquals("pan", page.evaluate("""
				() => document.getElementById(graphId('tag/a'))._fullLayout.dragmode
				"""));
		assertEquals(100, ((Number) page.evaluate(
				"() => document.getElementById('main-area').scrollTop")).intValue());

		clickLegendSeries(page, TAG_A, "run_a");
		waitForSeriesTrace(page, TAG_A, "run_a", false);
		page.evaluate("() => app.setSelectedRuns(['run_b'])");
		page.evaluate("() => app.setSelectedRuns(['run_a', 'run_b'])");
		waitForSeriesTrace(page, TAG_A, "run_a", true);

		clickLegendSeries(page, TAG_A, "run_a");
		waitForSeriesTrace(page, TAG_A, "run_a", false);
		page.evaluate("""
				() => {
					app.activeTags.delete('tag/a');
					app.onTagSelectionChanged();
					app.activeTags.add('tag/a');
					app.onTagSelectionChanged();
				}
				""");
		waitForSeriesTrace(page, TAG_A, "run_a", true);
	}

	@Test
	void bandLegendTogglesAllRunTracesAndResetViewIsHiddenInScreenshotMode() {
		page.route("**/api/runs.json", route -> fulfillJson(route, legendStateRunsJson()));
		page.route("**/api/metrics.json", route -> fulfillJson(route, legendStateLodMetricsJson()));
		page.route("**/api/runs/prioritize", MetricsViewerPlaywrightTestSupport::fulfillNoContent);

		page.navigate(baseUrl + "/?bandLegendStateTest=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		page.click("#btn-select-all-runs");
		page.selectOption("#lod-display-mode", "Band");
		page.waitForFunction("""
				() => document.getElementById(graphId('tag/a'))?.data?.length === 6
				""");

		clickLegendSeries(page, TAG_A, "run_a");
		page.waitForFunction("""
				() => document.getElementById(graphId('tag/a')).data
					.filter(trace => trace.meta?.runId === 'run_a')
					.every(trace => trace.visible === 'legendonly')
				""");
		page.click("#btn-graph-scroll-lock");
		waitForPlotlyDragModeFalse(page);
		page.click("#btn-reset-view");
		page.waitForFunction("""
				() => Array.from(document.querySelectorAll('.js-plotly-plot'))
					.flatMap(plot => Array.from(plot.data ?? []))
					.every(trace => trace.visible !== 'legendonly')
				""");
		assertTrue(isGraphScrollLockButtonActive(page));
		waitForPlotlyDragModeFalse(page);

		page.setViewportSize(320, 720);
		assertEquals("Reset View", page.textContent("#btn-reset-view"));
		assertEquals(true, page.evaluate("""
				() => {
					const controls = document.getElementById('floating-controls');
					const bounds = controls.getBoundingClientRect();
					const tops = Array.from(controls.children).map(child => child.getBoundingClientRect().top);
					return bounds.left >= 0
						&& bounds.right <= window.innerWidth
						&& Math.max(...tops) - Math.min(...tops) < 1;
				}
				"""));

		page.click("#btn-screenshot-toggle");
		assertEquals("none", page.evaluate("""
				() => getComputedStyle(document.getElementById('btn-reset-view')).display
				"""));
	}

	@Test
	void graphHeaderCombinesTagStatsAndShowsRunAndTagIssues() {
		page.route("**/api/runs.json", route -> fulfillJson(route, statsWarningRunsJson()));
		page.route("**/api/metrics.json", route -> fulfillJson(route, statsWarningMetricsJson()));
		page.route("**/api/runs/prioritize", MetricsViewerPlaywrightTestSupport::fulfillNoContent);

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
	}

	@Test
	void zoomUsesTheCoarsePlotImmediatelyThenRequestsOneThreeViewportWindow() throws Exception {
		final List<String> requests = Collections.synchronizedList(new ArrayList<>());
		page.route("**/api/runs.json", route -> fulfillJson(
				route,
				"{\"runs\":[" + runJson("run_lod_ui", 31, TAG_KEY) + "]}"));
		page.route("**/api/metrics.json", route -> {
			requests.add(route.request().postData());
			fulfillJson(route, lodMetricsJson());
		});
		page.route("**/api/runs/prioritize", MetricsViewerPlaywrightTestSupport::fulfillNoContent);

		page.navigate(baseUrl + "/?zoomRefinementTest=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		waitForGraph(page);
		page.evaluate("""
				() => Plotly.relayout(
					document.querySelector('.js-plotly-plot'),
					{'xaxis.range': [8, 12]})
				""");
		page.waitForCondition(() -> requests.size() >= 2);

		assertTrue(requests.size() >= 2);
		final JsonNode second = new com.fasterxml.jackson.databind.ObjectMapper()
				.readTree(requests.get(1))
				.path("series")
				.get(0);
		assertEquals(4, second.path("fromStep").asLong());
		assertEquals(16, second.path("toStep").asLong());
	}

	@Test
	void staleGenerationResponseIsDiscardedUntilMetadataAndProjectionMatch() {
		final String nextGeneration = "00000000-0000-0000-0000-000000000002";
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
		page.route("**/api/runs/prioritize", MetricsViewerPlaywrightTestSupport::fulfillNoContent);

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
	}

	@Test
	void plotlyPngDownloadWorksForSlashTagKeys() {
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
	}

	@Test
	void plotlyModeBarShowsAutoscaleAndResetAxes() {
		page.route("**/api/runs.json", route -> fulfillJson(route, runsJson()));
		page.route("**/api/metrics.json", route -> fulfillJson(route, metricsJson()));

		page.navigate(baseUrl + "/?plotlyModeBarTest=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		waitForGraph(page);

		final List<String> buttonTitles = readModeBarButtonTitles(page);
		assertTrue(buttonTitles.contains("Autoscale"));
		assertTrue(buttonTitles.contains("Reset axes"));
	}
}
