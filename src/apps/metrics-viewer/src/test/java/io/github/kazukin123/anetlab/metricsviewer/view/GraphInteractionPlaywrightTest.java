package io.github.kazukin123.anetlab.metricsviewer.view;

import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.*;
import static org.junit.jupiter.api.Assertions.*;

import java.util.List;

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

	@Test
	void graphHeightSliderResizesGraphsAndIsRestoredAfterPageReload() {
		routeSingleTagGraph();
		page.navigate(baseUrl + "/?graphHeightTest=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		waitForGraph(page);
		waitForGraphHeight(page, 300);

		// type=rangeはfillできないので、値を入れてからユーザー操作と同じinputとchangeを送る。
		page.evaluate("""
				() => {
					const slider = document.getElementById('graph-height');
					slider.value = '450';
					slider.dispatchEvent(new Event('input'));
					slider.dispatchEvent(new Event('change'));
				}
				""");
		waitForGraphHeight(page, 450);
		assertEquals("450", page.evaluate("() => localStorage.getItem('anet.metricsviewer.graphHeight')"));

		page.reload(new Page.ReloadOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		waitForGraph(page);
		waitForGraphHeight(page, 450);
		assertEquals("450|450|150|800|50", page.evaluate("""
				() => {
					const slider = document.getElementById('graph-height');
					return [slider.value, document.getElementById('graph-height-value').textContent,
						slider.min, slider.max, slider.step].join('|');
				}
				"""));
	}

	@Test
	void invalidStoredGraphHeightFallsBackToTheDefault() {
		routeSingleTagGraph();
		page.navigate(baseUrl + "/?graphHeightInvalidTest=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		waitForGraph(page);
		// 範囲の外の値と、範囲の中でも50px刻みに乗らない値は、どちらも既定へ戻す。
		for (String stored : List.of("9999", "425")) {
			page.evaluate("value => localStorage.setItem('anet.metricsviewer.graphHeight', value)", stored);
			page.reload(new Page.ReloadOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
			waitForGraph(page);
			waitForGraphHeight(page, 300);
		}
	}

	/**
	 * 点の無い範囲へ拡大したままtagを選び直しても、軸だけのgraphが出て、ダブルクリックで全体へ戻れる。
	 * graphが無いと、全tag一括のReset View以外に戻す手段が無かった。
	 */
	@Test
	void zoomIntoARangeWithoutPointsKeepsAnEmptyGraphThatCanBeResetByDoubleClick() {
		// 点はstep 0〜2だけにある。実serverと同じく、要求の範囲に点が無ければemptyで返す。
		page.route("**/api/runs.json", route -> fulfillJson(route, runsJson()));
		page.route("**/api/metrics.json", route -> {
			final var series = com.google.gson.JsonParser.parseString(route.request().postData())
					.getAsJsonObject().getAsJsonArray("series");
			if (series.get(0).getAsJsonObject().get("fromStep").getAsLong() <= 2) {
				fulfillJson(route, metricsJson());
				return;
			}
			final StringBuilder empty = new StringBuilder("{\"data\":[");
			for (int i = 0; i < series.size(); i++) {
				final var request = series.get(i).getAsJsonObject();
				if (i > 0) empty.append(',');
				empty.append("{\"runId\":").append(request.get("runId"))
						.append(",\"tagKey\":").append(request.get("tagKey"))
						.append(",\"generation\":\"").append(GENERATION).append("\"")
						.append(",\"availability\":\"empty\",\"pointBudget\":0,\"projection\":null}");
			}
			fulfillJson(route, empty.append("]}").toString());
		});
		page.route("**/api/runs/prioritize", MetricsViewerPlaywrightTestSupport::fulfillNoContent);
		page.navigate(baseUrl + "/?emptyZoomTest=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		waitForGraph(page);

		// 拡大した範囲の前後1画面分を含めても点に届かない所へ拡大し、tagを外して選び直す。
		page.evaluate("""
				async () => {
					await Plotly.relayout(document.getElementById(graphId('palette/test')),
						{'xaxis.range': [1000, 1100]});
					await new Promise(resolve => setTimeout(resolve, 300));
				}
				""");
		page.click("#tag-list li");
		page.waitForFunction("() => !document.querySelector('#main-area > .graph-block')");
		page.click("#tag-list li");
		page.waitForFunction("""
				() => {
					const plot = document.getElementById(graphId('palette/test'));
					const range = plot?.layout?.xaxis?.range;
					return plot && !plot.dataset.renderStage && plot.data.length === 0
						&& range?.[0] === 1000 && range?.[1] === 1100;
				}
				""");

		page.dblclick("#main-area .js-plotly-plot");
		page.waitForFunction("""
				() => app.explicitViewport('palette/test') === null
					&& (document.getElementById(graphId('palette/test'))?.data?.length ?? 0) > 0
				""");
	}

	private void routeSingleTagGraph() {
		page.route("**/api/runs.json", route -> fulfillJson(route, runsJson()));
		page.route("**/api/metrics.json", route -> fulfillJson(route, metricsJson()));
		page.route("**/api/runs/prioritize", MetricsViewerPlaywrightTestSupport::fulfillNoContent);
	}

	/** 外枠(CSS)とPlotlyのlayoutの両方が、指定の高さになるまで待つ。 */
	private static void waitForGraphHeight(Page page, int height) {
		page.waitForFunction("""
				height => {
					const block = document.querySelector('#main-area > .graph-block');
					const plot = block?.querySelector('.js-plotly-plot:not([data-render-stage])');
					return Math.round(block?.getBoundingClientRect().height ?? 0) === height
						&& plot?._fullLayout?.height === height;
				}
				""", height);
	}
}
