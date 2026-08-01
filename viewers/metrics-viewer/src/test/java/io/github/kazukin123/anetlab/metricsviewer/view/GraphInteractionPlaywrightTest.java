package io.github.kazukin123.anetlab.metricsviewer.view;

import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.manyGraphMetricsJson;
import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.manyGraphRunsJson;
import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.metricsJson;
import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.runsJson;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

import com.microsoft.playwright.Browser;
import com.microsoft.playwright.Page;
import com.microsoft.playwright.options.WaitUntilState;

@SpringBootTest(
		webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT,
		properties = "metricsviewer.runs-dir=target/playwright-test-empty-runs")
class GraphInteractionPlaywrightTest extends MetricsViewerPlaywrightTestSupport {

	@Test
	void plotlyPanModeSurvivesDraggingTheVisibleRange() {
		page.route("**/api/runs.json", route -> fulfillJson(route, runsJson()));
		page.route("**/api/metrics.json", route -> fulfillJson(route, metricsJson()));
		page.route("**/api/runs/prioritize", MetricsViewerPlaywrightTestSupport::fulfillNoContent);

		page.navigate(baseUrl + "/?plotlyPanDragTest=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		waitForGraph(page);
		page.hover(".js-plotly-plot");
		page.locator(".modebar-btn[data-title='Pan']").first().click();
		waitForPlotlyDragMode(page, "pan");

		panFirstPlotHorizontally(page);
		page.waitForFunction("app.explicitViewport('palette/test') !== null");
		waitForPlotlyDragMode(page, "pan");
		assertEquals("pan", readPlotlyDragMode(page));
	}

	@Test
	void graphScrollLockDefaultsOffTogglesPersistsAndHidesInScreenshotMode() {
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
	}

	@Test
	void graphScrollLockIsAvailableForMultiRunGraphs() {
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
	}

	@Test
	void graphScrollLockAllowsVerticalTouchScrollingOnMobileGraph() {
		reopenPage(new Browser.NewContextOptions()
			.setViewportSize(1280, 720)
			.setHasTouch(true));

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
	}

	@Test
	void graphScrollLockAllowsDragScrollingOnGraph() {
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
	}

	@Test
	void doubleClickingGraphAreaReloads() {
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
	}
}
