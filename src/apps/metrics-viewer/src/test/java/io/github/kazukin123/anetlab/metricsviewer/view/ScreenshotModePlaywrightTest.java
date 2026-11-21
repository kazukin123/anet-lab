package io.github.kazukin123.anetlab.metricsviewer.view;

import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.*;
import static org.junit.jupiter.api.Assertions.*;

import java.util.Map;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

import com.microsoft.playwright.Page;
import com.microsoft.playwright.options.WaitUntilState;

@SpringBootTest(
		webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT,
		properties = "metricsviewer.workspaces-dir=target/playwright-test-empty-workspaces")
class ScreenshotModePlaywrightTest extends MetricsViewerPlaywrightTestSupport {

	private static final int GRAPH_COUNT = 5;
	private static final int SCROLLED_PIXELS = 400;
	private static final double ANCHOR_TOLERANCE_PX = 2.0;
	private static final int SCREENSHOT_RESIZE_SETTLE_MS = 400;

	@Test
	void enteringAndLeavingScreenshotModeKeepsTheGraphAtTheTopOfTheView() {
		openGraphs("screenshotEnterScrollTest");
		setMainAreaScrollTop(page, SCROLLED_PIXELS);
		final Map<String, Object> before = readTopGraphAnchor(page);

		enterScreenshotMode(page);
		assertTrue(isDocumentScrollable(page));
		assertSameTopGraph(before, readTopGraphAnchor(page));

		leaveScreenshotMode(page);
		assertSameTopGraph(before, readTopGraphAnchor(page));
	}

	/**
	 * 再描画で位置が飛ぶかどうかはbrowserのscroll anchoring次第で、headless Edgeでは
	 * 修正前でも飛ばない。ここは再現テストではなく契約の固定として置く。
	 */
	@Test
	void reloadKeepsScrollPositionInScreenshotMode() {
		openGraphs("screenshotReloadScrollTest");
		enterScreenshotMode(page);
		setDocumentScrollTop(page, SCROLLED_PIXELS);
		final Map<String, Object> before = readTopGraphAnchor(page);

		page.evaluate("() => app.onReload()");
		waitForGraphCount(page, GRAPH_COUNT);

		assertSameTopGraph(before, readTopGraphAnchor(page));
	}

	@Test
	void reloadKeepsScrollPositionInNormalMode() {
		openGraphs("normalReloadScrollTest");
		setMainAreaScrollTop(page, SCROLLED_PIXELS);
		final Map<String, Object> before = readTopGraphAnchor(page);

		page.evaluate("() => app.onReload()");
		waitForGraphCount(page, GRAPH_COUNT);

		assertSameTopGraph(before, readTopGraphAnchor(page));
	}

	@Test
	void doubleClickingGraphReloadsInScreenshotMode() {
		openGraphs("screenshotDoubleClickReloadTest");
		enterScreenshotMode(page);
		// 早期returnでも onReload 自体は settle するので、取り直しが起きたことを数える。
		page.evaluate("""
				() => {
					const original = app.refreshMetadata.bind(app);
					window.__screenshotMetadataRefreshes = 0;
					app.refreshMetadata = async options => {
						window.__screenshotMetadataRefreshes += 1;
						return original(options);
					};
				}
				""");

		page.dblclick(".js-plotly-plot");

		page.waitForFunction("() => window.__screenshotMetadataRefreshes === 1",
				null, new Page.WaitForFunctionOptions().setTimeout(30000));
	}

	@Test
	void autoReloadStaysEnabledInScreenshotMode() {
		openGraphs("screenshotAutoReloadTest");
		page.click("#btn-auto-reload");
		assertTrue(isAutoReloadButtonActive(page));

		enterScreenshotMode(page);

		assertTrue(isAutoReloadButtonActive(page));
		assertTrue(Boolean.TRUE.equals(page.evaluate(
				"() => app.autoReloadEnabled && app.autoReloadTimer !== null")));
	}

	/** 初期化の途中で切り替えても、初期化の完了がスクリーンショットモードを上書きしない。 */
	@Test
	void screenshotModeEnteredDuringInitializationSurvivesItsCompletion() {
		installRunsFetchGate();
		page.route("**/api/runs.json", route -> fulfillJson(route, manyGraphRunsJson(GRAPH_COUNT)));
		page.route("**/api/metrics.json",
				route -> fulfillJson(route, manyGraphMetricsJson(GRAPH_COUNT)));
		page.route("**/api/runs/prioritize", MetricsViewerPlaywrightTestSupport::fulfillNoContent);

		page.navigate(baseUrl + "/?screenshotDuringInitTest=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		page.waitForFunction("() => app?.mode === 'metaLoading' && window.__runsHeld === true");
		page.click("#btn-screenshot");
		page.evaluate("() => window.__releaseRuns()");
		waitForGraphCount(page, GRAPH_COUNT);

		assertEquals("normal", page.evaluate("() => app.mode"));
		assertTrue(Boolean.TRUE.equals(page.evaluate("""
				() => app.screenshotModeEnabled
					&& document.body.classList.contains('screenshot-mode')
					&& app.graphScrollLockActive()
					&& app.scrollElement() === (document.scrollingElement ?? document.documentElement)
				""")));
	}

	/** 初期化に失敗したあとも、スクリーンショットモードから抜けられる。 */
	@Test
	void screenshotModeCanBeLeftAfterInitializationFailed() {
		installRunsFetchGate();
		page.route("**/api/runs.json", route -> route.fulfill(
				new com.microsoft.playwright.Route.FulfillOptions().setStatus(500).setBody("")));

		page.navigate(baseUrl + "/?screenshotAfterInitErrorTest=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		page.waitForFunction("() => app?.mode === 'metaLoading' && window.__runsHeld === true");
		page.click("#btn-screenshot");
		page.evaluate("() => window.__releaseRuns()");
		page.waitForFunction("() => app.mode === 'error'");

		page.click("#btn-screenshot-toggle");
		page.waitForFunction("""
				() => !app.screenshotModeEnabled && !document.body.classList.contains('screenshot-mode')
					&& document.getElementById('side-panel').getClientRects().length > 0
				""");
	}

	/** runs.jsonの最初の1回を、テストが放すまで止める。初期化をMETA_LOADINGのまま保つため。 */
	private void installRunsFetchGate() {
		context.addInitScript("""
				(() => {
					const originalFetch = globalThis.fetch.bind(globalThis);
					let release = null;
					const gate = new Promise(resolve => release = resolve);
					window.__runsHeld = false;
					window.__releaseRuns = () => release();
					globalThis.fetch = (input, init) => {
						const url = typeof input === 'string' ? input : input.url;
						if (!window.__runsHeld && url.endsWith('/api/runs.json')) {
							window.__runsHeld = true;
							return gate.then(() => originalFetch(input, init));
						}
						return originalFetch(input, init);
					};
				})();
				""");
	}

	private void openGraphs(String cacheBuster) {
		page.route("**/api/runs.json", route -> fulfillJson(route, manyGraphRunsJson(GRAPH_COUNT)));
		page.route("**/api/metrics.json",
				route -> fulfillJson(route, manyGraphMetricsJson(GRAPH_COUNT)));
		page.route("**/api/runs/prioritize", MetricsViewerPlaywrightTestSupport::fulfillNoContent);

		page.navigate(baseUrl + "/?" + cacheBuster + "=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		waitForGraphCount(page, GRAPH_COUNT);
		assertTrue(isMainAreaScrollable(page));
	}

	private static void enterScreenshotMode(Page page) {
		page.click("#btn-screenshot");
		page.waitForFunction("""
				() => document.body.classList.contains('screenshot-mode') && app.screenshotModeEnabled
				""", null, new Page.WaitForFunctionOptions().setTimeout(30000));
		// 遅れて走るresizeAllが位置を崩さないところまでを見る。
		page.waitForTimeout(SCREENSHOT_RESIZE_SETTLE_MS);
	}

	private static void leaveScreenshotMode(Page page) {
		page.click("#btn-screenshot-toggle");
		page.waitForFunction("""
				() => !document.body.classList.contains('screenshot-mode') && !app.screenshotModeEnabled
				""", null, new Page.WaitForFunctionOptions().setTimeout(30000));
	}

	/** 画面の一番上に見えているgraphを、tagKeyとそのはみ出し量で読む。 */
	@SuppressWarnings("unchecked")
	private static Map<String, Object> readTopGraphAnchor(Page page) {
		return (Map<String, Object>) page.evaluate("""
				() => {
					// スクロール主体の判定はapp側に頼らず、CSSの状態から独立に決める。
					const screenshot = document.body.classList.contains('screenshot-mode');
					const viewportTop = screenshot
						? 0
						: document.getElementById('main-area').getBoundingClientRect().top;
					for (const block of document.querySelectorAll('#main-area .graph-block')) {
						const rectangle = block.getBoundingClientRect();
						if (rectangle.bottom <= viewportTop) continue;
						return { tagKey: block.dataset.tagKey, offset: rectangle.top - viewportTop };
					}
					throw new Error('no graph block is visible');
				}
				""");
	}

	private static void assertSameTopGraph(Map<String, Object> expected, Map<String, Object> actual) {
		assertEquals(expected.get("tagKey"), actual.get("tagKey"), "top graph");
		assertEquals(
				((Number) expected.get("offset")).doubleValue(),
				((Number) actual.get("offset")).doubleValue(),
				ANCHOR_TOLERANCE_PX,
				"top graph offset");
	}
}
