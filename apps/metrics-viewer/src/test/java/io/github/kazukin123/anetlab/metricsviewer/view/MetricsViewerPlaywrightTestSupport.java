package io.github.kazukin123.anetlab.metricsviewer.view;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.TestInstance;
import org.springframework.boot.test.web.server.LocalServerPort;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.microsoft.playwright.Browser;
import com.microsoft.playwright.BrowserContext;
import com.microsoft.playwright.BrowserType;
import com.microsoft.playwright.CDPSession;
import com.microsoft.playwright.Page;
import com.microsoft.playwright.Playwright;
import com.microsoft.playwright.PlaywrightException;
import com.microsoft.playwright.Route;

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
abstract class MetricsViewerPlaywrightTestSupport {

	private static final Path TEST_RUNS_DIR = Path.of("target/playwright-test-empty-runs");

	@LocalServerPort
	protected int port;

	protected String baseUrl;
	protected BrowserContext context;
	protected Page page;

	private Playwright playwright;
	private Browser browser;

	@BeforeAll
	void launchBrowser() throws IOException {
		// 全テストクラスで同じ前提を使い、ブラウザ起動だけをクラス単位へ縮約する。
		Files.createDirectories(TEST_RUNS_DIR);
		assumeTrue(isMicrosoftEdgeInstalled(), "Microsoft Edge is not installed.");
		playwright = Playwright.create();
		browser = launchMicrosoftEdge(playwright);
	}

	@BeforeEach
	void openIsolatedPage() {
		// route・localStorage・Plotly状態をテスト間で共有しない。
		baseUrl = "http://127.0.0.1:" + port;
		reopenPage(new Browser.NewContextOptions().setViewportSize(1280, 720));
	}

	@AfterEach
	void closeIsolatedPage() {
		if (context != null) {
			context.close();
			context = null;
			page = null;
		}
	}

	@AfterAll
	void closeBrowser() {
		if (browser != null) browser.close();
		if (playwright != null) playwright.close();
	}

	protected final void reopenPage(Browser.NewContextOptions options) {
		closeIsolatedPage();
		context = browser.newContext(options);
		page = context.newPage();
	}

	protected static Browser launchMicrosoftEdge(Playwright playwright) {
		try {
			return playwright.chromium().launch(new BrowserType.LaunchOptions()
					.setChannel("msedge")
					.setHeadless(true));
		} catch (PlaywrightException e) {
			assumeTrue(false, "Microsoft Edge Playwright launch is not available: " + e.getMessage());
			throw e;
		}
	}

	protected static boolean isMicrosoftEdgeInstalled() {
		return Files.exists(Path.of("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"))
				|| Files.exists(Path.of("C:/Program Files/Microsoft/Edge/Application/msedge.exe"));
	}

	protected static void fulfillJson(Route route, String body) {
		route.fulfill(new Route.FulfillOptions()
				.setStatus(200)
				.setContentType("application/json")
				.setBody(body));
	}

	protected static void fulfillNoContent(Route route) {
		route.fulfill(new Route.FulfillOptions().setStatus(204).setBody(""));
	}

	protected static void waitForGraph(Page page) {
		page.waitForFunction("document.querySelectorAll('.js-plotly-plot path.js-line').length > 0",
				null, new Page.WaitForFunctionOptions().setTimeout(30000));
	}

	protected static void waitForMultiRunGraph(Page page) {
		page.waitForFunction("""
				() => {
					const plot = document.querySelector('.js-plotly-plot');
					return (plot?.data?.length ?? 0) > 1;
				}
				""", null, new Page.WaitForFunctionOptions().setTimeout(30000));
	}

	protected static void waitForGraphCount(Page page, int count) {
		page.waitForFunction("""
				count => document.querySelectorAll('.js-plotly-plot').length === count
					&& document.querySelectorAll('.js-plotly-plot path.js-line').length >= count
				""", count, new Page.WaitForFunctionOptions().setTimeout(30000));
	}

	protected static boolean isAutoReloadButtonActive(Page page) {
		return Boolean.TRUE.equals(page.evaluate("""
				() => document.getElementById('btn-auto-reload').classList.contains('active')
				"""));
	}

	protected static boolean isGraphScrollLockButtonActive(Page page) {
		return Boolean.TRUE.equals(page.evaluate("""
				() => document.getElementById('btn-graph-scroll-lock').classList.contains('active')
				"""));
	}

	protected static boolean isGraphScrollLockButtonVisible(Page page) {
		return Boolean.TRUE.equals(page.evaluate("""
				() => {
					const el = document.getElementById('btn-graph-scroll-lock');
					if (!el) return false;
					const style = getComputedStyle(el);
					return style.display !== 'none'
						&& style.visibility !== 'hidden'
						&& el.getClientRects().length > 0;
				}
				"""));
	}

	protected static boolean areFloatingControlsSideBySide(Page page) {
		return Boolean.TRUE.equals(page.evaluate("""
				() => {
					const lock = document.getElementById('btn-graph-scroll-lock')?.getBoundingClientRect();
					const shot = document.getElementById('btn-screenshot-toggle')?.getBoundingClientRect();
					if (!lock || !shot) return false;
					const verticallyOverlaps = lock.top < shot.bottom && shot.top < lock.bottom;
					return verticallyOverlaps && lock.right <= shot.left + 1;
				}
				"""));
	}

	protected static boolean hasTopClearanceForFloatingControls(Page page) {
		return Boolean.TRUE.equals(page.evaluate("""
				() => {
					const controls = document.getElementById('floating-controls')?.getBoundingClientRect();
					const firstGraph = document.querySelector('.graph-block')?.getBoundingClientRect();
					if (!controls || !firstGraph) return false;
					return firstGraph.top >= controls.bottom + 8;
				}
				"""));
	}

	protected static String readGraphScrollLockStorage(Page page) {
		return (String) page.evaluate("""
				() => localStorage.getItem('anet.metricsviewer.graphScrollLockEnabled')
				""");
	}

	protected static boolean isPlotlyDragModeFalse(Page page) {
		return Boolean.TRUE.equals(page.evaluate("""
				() => {
					const plot = document.querySelector('.js-plotly-plot');
					return (plot?._fullLayout?.dragmode ?? plot?.layout?.dragmode) === false;
				}
				"""));
	}

	protected static void waitForPlotlyDragModeFalse(Page page) {
		page.waitForFunction("""
				() => {
					const plot = document.querySelector('.js-plotly-plot');
					return (plot?._fullLayout?.dragmode ?? plot?.layout?.dragmode) === false;
				}
				""", null, new Page.WaitForFunctionOptions().setTimeout(30000));
	}

	@SuppressWarnings("unchecked")
	protected static Map<String, Number> readFirstTraceMiddlePointScreenPosition(Page page) {
		return (Map<String, Number>) page.evaluate("""
				() => {
					const plot = document.querySelector('.js-plotly-plot');
					const full = plot?._fullLayout;
					const trace = plot?.data?.[0];
					if (!plot || !full || !trace || trace.x.length < 2 || trace.y.length < 2) {
						throw new Error('plot point not ready');
					}
					const rect = plot.getBoundingClientRect();
					return {
						x: rect.left + full._size.l + full.xaxis.l2p(Number(trace.x[1])),
						y: rect.top + full._size.t + full.yaxis.l2p(Number(trace.y[1]))
					};
				}
				""");
	}

	protected static void hoverFirstTraceMiddlePoint(Page page) {
		final Map<String, Number> position = readFirstTraceMiddlePointScreenPosition(page);
		page.mouse().move(position.get("x").doubleValue(), position.get("y").doubleValue());
	}

	protected static void waitForPlotlyHoverText(Page page) {
		page.waitForFunction("""
				() => Array.from(document.querySelectorAll('.hovertext'))
					.some(el => (el.textContent || '').trim().length > 0)
				""", null, new Page.WaitForFunctionOptions().setTimeout(30000));
	}

	protected static void clickFirstLegendItem(Page page) {
		page.click(".legend .traces");
	}

	protected static void waitForLegendOnlyTrace(Page page) {
		page.waitForFunction("""
				() => {
					const plot = document.querySelector('.js-plotly-plot');
					return Array.from(plot?.data ?? []).some(trace => trace.visible === 'legendonly');
				}
				""", null, new Page.WaitForFunctionOptions().setTimeout(30000));
	}

	protected static void clickLegendSeries(Page page, String tagKey, String runId) {
		final int legendIndex = ((Number) page.evaluate("""
				([tagKey, runId]) => {
					const plot = document.getElementById(graphId(tagKey));
					return Array.from(plot?.data ?? [])
						.filter(trace => trace.showlegend !== false)
						.findIndex(trace => trace.meta?.runId === runId);
				}
				""", List.of(tagKey, runId))).intValue();
		assertTrue(legendIndex >= 0, "Legend series should exist");
		page.locator("#" + graphDomId(tagKey) + " .legend .traces").nth(legendIndex).click();
	}

	protected static void waitForSeriesTrace(Page page, String tagKey, String runId, boolean visible) {
		page.waitForFunction("""
				([tagKey, runId, visible]) => {
					const trace = Array.from(document.getElementById(graphId(tagKey))?.data ?? [])
						.find(candidate => candidate.meta?.runId === runId && candidate.showlegend !== false);
					return !!trace && (trace.visible !== 'legendonly') === visible;
				}
				""", List.of(tagKey, runId, visible),
				new Page.WaitForFunctionOptions().setTimeout(30000));
	}

	protected static boolean isSeriesVisible(Page page, String tagKey, String runId) {
		return Boolean.TRUE.equals(page.evaluate("""
				([tagKey, runId]) => Array.from(document.getElementById(graphId(tagKey))?.data ?? [])
					.some(trace => trace.meta?.runId === runId
						&& trace.showlegend !== false
						&& trace.visible !== 'legendonly')
				""", List.of(tagKey, runId)));
	}

	protected static String graphDomId(String tagKey) {
		final StringBuilder encoded = new StringBuilder("graph-");
		for (byte value : tagKey.getBytes(StandardCharsets.UTF_8)) {
			encoded.append(String.format("%02x", value & 0xff));
		}
		return encoded.toString();
	}

	protected static boolean isMainAreaScrollable(Page page) {
		return Boolean.TRUE.equals(page.evaluate("""
				() => {
					const el = document.getElementById('main-area');
					return !!el && el.scrollHeight > el.clientHeight;
				}
				"""));
	}

	protected static void setMainAreaScrollTop(Page page, int scrollTop) {
		page.evaluate("""
				scrollTop => {
					const el = document.getElementById('main-area');
					if (!el) throw new Error('main-area not found');
					el.scrollTop = scrollTop;
				}
				""", scrollTop);
	}

	protected static void waitForMainAreaScrolled(Page page) {
		page.waitForFunction("""
				() => (document.getElementById('main-area')?.scrollTop ?? 0) > 0
				""", null, new Page.WaitForFunctionOptions().setTimeout(30000));
	}

	protected static void waitForPlotlyDragCoverRemoved(Page page) {
		page.waitForFunction("""
				() => !document.querySelector('.dragcover')
				""", null, new Page.WaitForFunctionOptions().setTimeout(30000));
	}

	protected static double readFirstGraphCenterX(Page page) {
		return ((Number) page.evaluate("""
				() => {
					const rect = document.querySelector('.js-plotly-plot')?.getBoundingClientRect();
					if (!rect) throw new Error('plot not found');
					return rect.left + rect.width / 2;
				}
				""")).doubleValue();
	}

	protected static double readFirstGraphCenterY(Page page) {
		return ((Number) page.evaluate("""
				() => {
					const rect = document.querySelector('.js-plotly-plot')?.getBoundingClientRect();
					if (!rect) throw new Error('plot not found');
					return rect.top + rect.height / 2;
				}
				""")).doubleValue();
	}

	protected static void dispatchTouchSwipe(Page page, double x, double startY, double endY) {
		final CDPSession cdp = page.context().newCDPSession(page);
		try {
			dispatchTouchEvent(cdp, "touchStart", x, startY);
			final int steps = 10;
			for (int i = 1; i <= steps; i++) {
				final double y = startY + (endY - startY) * i / steps;
				dispatchTouchEvent(cdp, "touchMove", x, y);
				page.waitForTimeout(16);
			}
			dispatchTouchEvent(cdp, "touchEnd", x, endY);
		} finally {
			cdp.detach();
		}
	}

	protected static void dispatchTouchEvent(CDPSession cdp, String type, double x, double y) {
		final JsonObject params = new JsonObject();
		params.addProperty("type", type);
		final JsonArray touchPoints = new JsonArray();
		if (!"touchEnd".equals(type) && !"touchCancel".equals(type)) {
			final JsonObject touchPoint = new JsonObject();
			touchPoint.addProperty("x", x);
			touchPoint.addProperty("y", y);
			touchPoint.addProperty("id", 1);
			touchPoint.addProperty("radiusX", 2);
			touchPoint.addProperty("radiusY", 2);
			touchPoint.addProperty("force", 1);
			touchPoints.add(touchPoint);
		}
		params.add("touchPoints", touchPoints);
		cdp.send("Input.dispatchTouchEvent", params);
	}

	@SuppressWarnings("unchecked")
	protected static void dragFromFirstPlotBody(Page page, int holdMs) {
		final Map<String, Number> position = (Map<String, Number>) page.evaluate("""
				() => {
					const rect = document.querySelector('.js-plotly-plot .nsewdrag')?.getBoundingClientRect();
					if (!rect) throw new Error('plot body not found');
					return {
						x: rect.left + rect.width / 2,
						y: rect.top + rect.height * 0.7
					};
				}
				""");
		final double x = position.get("x").doubleValue();
		final double startY = position.get("y").doubleValue();
		final double endY = startY - 80;
		page.mouse().move(x, startY);
		assertTrue(Boolean.TRUE.equals(page.evaluate("""
				([x, y]) => document.elementFromPoint(x, y)?.classList.contains('nsewdrag') === true
				""", List.of(x, startY))), "Drag must start on Plotly's nsewdrag layer");
		page.mouse().down();
		page.waitForTimeout(holdMs);
		page.mouse().move(x, endY, new com.microsoft.playwright.Mouse.MoveOptions().setSteps(20));
		page.mouse().up();
	}

	@SuppressWarnings("unchecked")
	protected static List<Map<String, String>> readRunChips(Page page) {
		return (List<Map<String, String>>) page.evaluate("""
				Array.from(document.querySelectorAll('#run-list .run-color')).map((el) => ({
					runId: el.parentElement.dataset.runId,
					color: getComputedStyle(el).backgroundColor
				}))
				""");
	}

	@SuppressWarnings("unchecked")
	protected static List<String> readActiveRunIds(Page page) {
		return (List<String>) page.evaluate("""
				Array.from(document.querySelectorAll('#run-list .run-row.active'))
					.map(row => row.dataset.runId)
				""");
	}

	protected static String readFirstLineStroke(Page page) {
		return (String) page.evaluate("""
				(() => {
					const line = document.querySelector('.js-plotly-plot path.js-line');
					return line ? (line.getAttribute('stroke') || getComputedStyle(line).stroke) : '';
				})()
				""");
	}

	protected static String readTraceColor(Page page, String runId) {
		return (String) page.evaluate("""
				runId => {
					const plot = document.querySelector('.js-plotly-plot');
					const trace = Array.from(plot?.data ?? [])
						.find(candidate => candidate?.meta?.runId === runId);
					const color = trace?.line?.color ?? trace?.marker?.color ?? '';
					if (!color) return '';
					const probe = document.createElement('span');
					probe.style.color = color;
					document.body.appendChild(probe);
					const computedColor = getComputedStyle(probe).color;
					probe.remove();
					return computedColor;
				}
				""", runId);
	}

	protected static String readFirstTraceUid(Page page) {
		return (String) page.evaluate("""
				(() => {
					const plot = document.querySelector('.js-plotly-plot');
					return plot?.data?.[0]?.uid ?? '';
				})()
				""");
	}

	protected static String renderFirstPlotPngPrefix(Page page) {
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

	@SuppressWarnings("unchecked")
	protected static List<String> readModeBarButtonTitles(Page page) {
		return (List<String>) page.evaluate("""
				Array.from(document.querySelectorAll('.modebar-btn[data-title]'))
					.map(el => el.getAttribute('data-title'))
				""");
	}

	protected static String readYAxisType(Page page) {
		return (String) page.evaluate("""
				(() => {
					const plot = document.querySelector('.js-plotly-plot');
					return plot ? (plot._fullLayout?.yaxis?.type || plot.layout?.yaxis?.type || 'linear') : '';
				})()
				""");
	}

	protected static void waitForSignedLogTrace(Page page) {
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

	protected static void waitForSignedLogMixedSignTrace(Page page) {
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

	protected static void waitForSignedLogZoomSourceTrace(Page page) {
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

	protected static void zoomToSignedLogRange(Page page, int minStep, int maxStep, double minValue, double maxValue) {
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

	protected static void waitForSignedLogZoomTicks(Page page) {
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

	protected static void setPlotlyPanMode(Page page) {
		page.evaluate("""
				() => {
					const plot = document.querySelector('.js-plotly-plot');
					return Plotly.relayout(plot, { dragmode: 'pan' });
				}
				""");
	}

	@SuppressWarnings("unchecked")
	protected static void panFirstPlotHorizontally(Page page) {
		final Map<String, Number> position = (Map<String, Number>) page.evaluate("""
				() => {
					const rect = document.querySelector('.js-plotly-plot .nsewdrag')?.getBoundingClientRect();
					if (!rect) throw new Error('plot body not found');
					return {x: rect.left + rect.width / 2, y: rect.top + rect.height / 2};
				}
				""");
		final double x = position.get("x").doubleValue();
		final double y = position.get("y").doubleValue();
		page.mouse().move(x, y);
		page.mouse().down();
		page.mouse().move(x - 120, y, new com.microsoft.playwright.Mouse.MoveOptions().setSteps(20));
		page.mouse().up();
	}

	protected static String readPlotlyDragMode(Page page) {
		return (String) page.evaluate("""
				() => {
					const plot = document.querySelector('.js-plotly-plot');
					return plot?._fullLayout?.dragmode ?? plot?.layout?.dragmode ?? '';
				}
				""");
	}

	protected static void waitForPlotlyDragMode(Page page, String dragMode) {
		page.waitForFunction("""
				dragMode => {
					const plot = document.querySelector('.js-plotly-plot');
					return (plot?._fullLayout?.dragmode ?? plot?.layout?.dragmode) === dragMode;
				}
				""", dragMode, new Page.WaitForFunctionOptions().setTimeout(30000));
	}

	@SuppressWarnings("unchecked")
	protected static List<String> readTagList(Page page) {
		return (List<String>) page.evaluate("""
				Array.from(document.querySelectorAll('#tag-list li')).map(el => el.textContent)
				""");
	}

	@SuppressWarnings("unchecked")
	protected static List<String> readGraphTitles(Page page) {
		return (List<String>) page.evaluate("""
				Array.from(document.querySelectorAll('.graph-title')).map(el => el.textContent)
				""");
	}

	protected static boolean isTagListScrollable(Page page) {
		return Boolean.TRUE.equals(page.evaluate("""
				() => {
					const el = document.querySelector('#tag-list');
					return !!el && el.scrollHeight > el.clientHeight;
				}
				"""));
	}

	protected static boolean canScrollTagList(Page page) {
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

	protected static String readTagListTouchAction(Page page) {
		return (String) page.evaluate("""
				() => {
					const el = document.querySelector('#tag-list');
					return el ? getComputedStyle(el).touchAction : '';
				}
				""");
	}

	protected static void selectSingleRun(Page page, String runId) {
		page.evaluate("""
				runId => {
					const row = Array.from(document.querySelectorAll('#run-list .run-row'))
						.find(el => el.dataset.runId === runId);
					if (!row) throw new Error('run not found: ' + runId);
					row.click();
					row.click();
				}
				""", runId);
	}

	protected static void waitForTag(Page page, String tagKey) {
		page.waitForFunction("""
				tagKey => Array.from(document.querySelectorAll('#tag-list li'))
					.some(el => el.textContent === tagKey)
				""", tagKey, new Page.WaitForFunctionOptions().setTimeout(30000));
	}

	protected static void waitForGraphTitle(Page page, String tagKey) {
		page.waitForFunction("""
				tagKey => Array.from(document.querySelectorAll('.graph-title'))
					.some(el => el.textContent === tagKey)
				""", tagKey, new Page.WaitForFunctionOptions().setTimeout(30000));
	}

	protected static boolean isTagVisible(Page page, String tagKey) {
		return Boolean.TRUE.equals(page.evaluate("""
				tagKey => Array.from(document.querySelectorAll('#tag-list li'))
					.some(el => el.textContent === tagKey)
				""", tagKey));
	}

	protected static boolean isTagActive(Page page, String tagKey) {
		return Boolean.TRUE.equals(page.evaluate("""
				tagKey => {
					const tag = Array.from(document.querySelectorAll('#tag-list li'))
						.find(el => el.textContent === tagKey);
					return !!tag && tag.classList.contains('active');
				}
				""", tagKey));
	}

	protected static boolean isGraphTitleVisible(Page page, String tagKey) {
		return Boolean.TRUE.equals(page.evaluate("""
				tagKey => Array.from(document.querySelectorAll('.graph-title'))
					.some(el => el.textContent === tagKey)
				""", tagKey));
	}
}
