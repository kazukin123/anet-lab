package io.github.kazukin123.anetlab.metricsviewer.view;

import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.*;
import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

import com.microsoft.playwright.Browser;
import com.microsoft.playwright.Page;
import com.microsoft.playwright.options.BoundingBox;
import com.microsoft.playwright.options.WaitUntilState;

@SpringBootTest(
		webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT,
		properties = "metricsviewer.workspaces-dir=target/playwright-test-empty-workspaces")
class GraphInteractionPlaywrightTest extends MetricsViewerPlaywrightTestSupport {

	@Test
	void resizeKeepsAnExistingWindowHandlerAndResizesMetricsViewer() {
		context.addInitScript("""
				window.__existingResizeCalls = 0;
				window.onresize = () => window.__existingResizeCalls += 1;
				""");
		page.route("**/api/runs.json", route -> fulfillJson(route, runsJson()));
		page.route("**/api/metrics.json", route -> fulfillJson(route, metricsJson()));

		page.navigate(baseUrl + "/?resizeListenerCoexistenceTest=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		waitForGraph(page);
		page.evaluate("""
				() => {
					const original = app.plotly.resizeAll.bind(app.plotly);
					window.__metricsViewerResizeCalls = 0;
					app.plotly.resizeAll = () => {
						window.__metricsViewerResizeCalls += 1;
						return original();
					};
					window.dispatchEvent(new Event('resize'));
				}
				""");

		assertEquals(1, ((Number) page.evaluate("window.__existingResizeCalls")).intValue());
		assertEquals(1, ((Number) page.evaluate("window.__metricsViewerResizeCalls")).intValue());
	}

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

		assertEquals("Scroll Lock", page.textContent("#btn-graph-scroll-lock"));
		assertFalse(isGraphScrollLockButtonActive(page));
		assertTrue(isGraphScrollLockButtonVisible(page));
		assertTrue(areFloatingControlsSideBySide(page));
		assertTrue(hasTopClearanceForFloatingControls(page));
		assertFalse(isPlotlyDragModeFalse(page));
		assertEquals(null, readGraphScrollLockStorage(page));

		setPlotlyPanMode(page);
		waitForPlotlyDragMode(page, "pan");
		page.click("#btn-graph-scroll-lock");
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
		assertEquals("Scroll Lock", page.textContent("#btn-graph-scroll-lock"));
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
	void hoveringNearASingleSampleShowsItsValue() {
		page.route("**/api/runs.json", route -> fulfillJson(route, singleSampleRunsJson()));
		page.route("**/api/metrics.json", route -> fulfillJson(route, singleSampleMetricsJson()));
		page.route("**/api/runs/prioritize", MetricsViewerPlaywrightTestSupport::fulfillNoContent);

		page.navigate(baseUrl + "/?singleSampleHoverTest=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		waitForDrawnMarkers(page, 1);

		// 実際のカーソルはそのstepへ正確には重ならないので、点から数px外して動かす。
		final BoundingBox marker = page.locator(".scatterlayer path.point").boundingBox();
		page.mouse().move(marker.x + marker.width / 2 + 3, marker.y + marker.height / 2);
		waitForPlotlyHoverText(page);

		assertEquals("run_single", page.textContent(".hover-tip-name"));
		assertEquals("step 0", page.textContent(".hover-tip-step"));
	}

	@Test
	void tappingTheGraphShowsTheHoverOverlayOnTouchDevice() {
		reopenPage(new Browser.NewContextOptions()
				.setViewportSize(1280, 720)
				.setHasTouch(true));

		page.route("**/api/runs.json", route -> fulfillJson(route, runsJson()));
		page.route("**/api/metrics.json", route -> fulfillJson(route, metricsJson()));

		page.navigate(baseUrl + "/?touchTapHoverTest=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		waitForGraph(page);

		tapFirstTraceMiddlePoint(page);
		waitForPlotlyHoverText(page);
	}

	@Test
	void tappingTheGraphShowsTheHoverOverlayWhileScrollLocked() {
		reopenPage(new Browser.NewContextOptions()
				.setViewportSize(1280, 720)
				.setHasTouch(true));

		page.route("**/api/runs.json", route -> fulfillJson(route, runsJson()));
		page.route("**/api/metrics.json", route -> fulfillJson(route, metricsJson()));

		page.navigate(baseUrl + "/?touchTapHoverScrollLockTest=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		waitForGraph(page);

		page.click("#btn-graph-scroll-lock");
		waitForPlotlyDragModeFalse(page);

		tapFirstTraceMiddlePoint(page);
		waitForPlotlyHoverText(page);
	}

	@Test
	void tappingOutsideTheGraphHidesTheHoverOverlay() {
		reopenPage(new Browser.NewContextOptions()
				.setViewportSize(1280, 720)
				.setHasTouch(true));

		page.route("**/api/runs.json", route -> fulfillJson(route, runsJson()));
		page.route("**/api/metrics.json", route -> fulfillJson(route, metricsJson()));

		page.navigate(baseUrl + "/?touchTapDismissTest=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		waitForGraph(page);

		tapFirstTraceMiddlePoint(page);
		waitForPlotlyHoverText(page);

		final BoundingBox outside = page.locator("#side-header").boundingBox();
		dispatchTouchTap(page, outside.x + outside.width / 2, outside.y + outside.height / 2);
		waitForHoverOverlayHidden(page);
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
	void graphScrollLockLeavesTouchSwipeToNativeScrolling() {
		reopenPage(new Browser.NewContextOptions()
				.setViewportSize(1280, 720)
				.setHasTouch(true));

		page.route("**/api/runs.json", route -> fulfillJson(route, manyGraphRunsJson(5)));
		page.route("**/api/metrics.json", route -> fulfillJson(route, manyGraphMetricsJson(5)));

		page.navigate(baseUrl + "/?graphScrollLockNativeTouchTest=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		waitForGraphCount(page, 5);
		assertTrue(isMainAreaScrollable(page));

		page.click("#btn-graph-scroll-lock");
		waitForPlotlyDragModeFalse(page);

		// window の capture で最初に受け、配り終えたあとの defaultPrevented を見る。
		// 途中の listener が stopPropagation しても、ここには必ず届く。
		page.evaluate("""
				() => {
					window.__graphTouchMoves = [];
					window.addEventListener('touchmove', event => window.__graphTouchMoves.push(event),
							{ capture: true, passive: true });
				}
				""");
		setMainAreaScrollTop(page, 0);
		dispatchTouchSwipe(page, readFirstGraphCenterX(page), readFirstGraphCenterY(page) + 90,
				readFirstGraphCenterY(page) - 130);
		waitForMainAreaScrolled(page);

		// 指を離したあとの慣性は browser の標準 scroll が付ける。ページが touchmove を
		// 取り消すと標準 scroll が止まり、慣性も出ない。
		assertTrue(((Number) page.evaluate("window.__graphTouchMoves.length")).intValue() > 0);
		assertEquals(0, ((Number) page.evaluate(
				"window.__graphTouchMoves.filter(event => event.defaultPrevented).length")).intValue());
	}

	@Test
	void screenshotModeAllowsVerticalTouchScrollingOnGraph() {
		reopenPage(new Browser.NewContextOptions()
				.setViewportSize(1280, 720)
				.setHasTouch(true));

		page.route("**/api/runs.json", route -> fulfillJson(route, manyGraphRunsJson(5)));
		page.route("**/api/metrics.json", route -> fulfillJson(route, manyGraphMetricsJson(5)));

		page.navigate(baseUrl + "/?screenshotTouchScrollTest=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		waitForGraphCount(page, 5);

		page.click("#btn-screenshot");
		page.waitForFunction("document.body.classList.contains('screenshot-mode')",
				null, new Page.WaitForFunctionOptions().setTimeout(30000));
		waitForPlotlyDragModeFalse(page);
		assertTrue(isDocumentScrollable(page));

		for (int attempt = 0; attempt < 2; attempt++) {
			setDocumentScrollTop(page, 0);
			dispatchTouchSwipe(page, readFirstGraphCenterX(page), readFirstGraphCenterY(page) + 90,
					readFirstGraphCenterY(page) - 130);
			waitForDocumentScrolled(page);
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
	void doubleClickingGraphAreaReloadsWithoutReplacingAnExistingClickHandler() {
		context.addInitScript("""
				window.__existingMainAreaDoubleClickCalls = 0;
				window.addEventListener('load', () => {
					document.getElementById('main-area').onclick = event => {
						if (event.detail === 2) window.__existingMainAreaDoubleClickCalls += 1;
					};
				});
				""");
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
		assertEquals(1, ((Number) page.evaluate(
				"() => window.__existingMainAreaDoubleClickCalls")).intValue());
	}
}
