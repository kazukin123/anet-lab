package io.github.kazukin123.anetlab.metricsviewer.view;

import static com.microsoft.playwright.assertions.PlaywrightAssertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;

import com.microsoft.playwright.Browser;
import com.microsoft.playwright.Locator;
import com.microsoft.playwright.Mouse;
import com.microsoft.playwright.Page;
import com.microsoft.playwright.options.AriaRole;

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
class RunComparisonPlaywrightTest extends MetricsViewerPlaywrightTestSupport {

	private static final Path RUNS = createRuns();

	@DynamicPropertySource
	static void configure(DynamicPropertyRegistry registry) {
		registry.add("metricsviewer.workspaces-dir", () -> RUNS.getParent().getParent().toString());
		registry.add("metricsviewer.initial-workspace", () -> "comparison");
	}

	@Test
	void configTabComparesSettingsFromSelectedRuns() {
		page.navigate(baseUrl);
		waitForRunRows(page, 2);
		page.click("#btn-select-all-runs");
		page.getByRole(AriaRole.TAB,
				new Page.GetByRoleOptions().setName("Config")).click();

		assertThat(page.locator("#config-table thead th")).hasText(
				new String[] {"Key", "run-b", "run-a"});
		assertThat(page.locator("#config-table tbody tr")).hasCount(3);
		assertThat(page.locator("#config-table tbody tr.config-different")).hasCount(2);
		assertThat(page.locator("#config-table tbody").getByText("(none)")).isVisible();
		page.click("#btn-config-diff");
		assertThat(page.locator("#config-table tbody tr")).hasCount(2);
		page.fill("#config-filter", "ALPHA beta");
		assertThat(page.locator("#config-table tbody th")).hasText("alpha.beta");
		final List<String> graphControls = List.of(
				"#graph-height-control", "#lod-display-mode-control", "#btn-reset-view", "#btn-graph-scroll-lock");
		for (String selector : graphControls) assertThat(page.locator(selector)).isHidden();

		page.click("#tab-graphs");
		for (String selector : graphControls) assertThat(page.locator(selector)).isVisible();
		waitForGraph(page);
		page.click("#tab-config");
		page.reload();
		assertThat(page.locator("#tab-config")).hasAttribute("aria-selected", "true");
		page.click("#btn-screenshot-toggle");
		assertThat(page.locator("#main-tabs")).isHidden();
		assertThat(page.locator("#main-area")).isHidden();
		assertThat(page.locator("#config-table")).isVisible();
		page.click("#btn-screenshot-toggle");
		assertThat(page.locator("#tab-config")).isVisible();
	}


	@Test
	void keywordFiltersScopeBulkActionsWithoutReplacingOutOfFilterSelections() {
		page.navigate(baseUrl);
		waitForRunRows(page, 2);
		assertThat(page.locator("#run-list .run-row.active")).hasCount(1);
		for (String whitespace : new String[] {" ", "\u3000", "\u00A0"}) {
			page.fill("#run-filter", "RUN" + whitespace + "-A");
			assertThat(page.locator("#run-list .run-row:visible .run-name")).hasText("run-a");
		}
		page.click("#btn-selected-only-runs");
		assertThat(page.locator("#run-list .run-row:visible")).hasCount(0);
		page.click("#btn-select-all-runs");
		assertThat(page.locator("#btn-selected-only-runs")).hasAttribute("aria-pressed", "false");
		assertThat(page.locator("#run-filter")).hasValue("RUN\u00A0-A");
		assertThat(page.locator("#run-list .run-row.active")).hasCount(2);
		page.click("#btn-latest-only");
		assertThat(page.locator("#run-list .run-row.active .run-name")).hasText("run-a");

		waitForTag(page, "score/train");
		page.fill("#tag-filter", "SCORE train");
		assertThat(page.locator("#tag-list li:visible .tag-label")).hasText("score/train");
		page.click("#btn-clear-all");
		assertThat(page.locator("#tag-list li.active:visible")).hasCount(0);
		page.fill("#tag-filter", "");
		assertThat(page.locator("#tag-list li.active .tag-label")).hasText("loss");
		page.fill("#tag-filter", "score");
		page.click("#btn-select-all");
		assertThat(page.locator("#tag-list li.active:visible .tag-label")).hasText("score/train");
	}


	@Test
	void tagPanelComparesAllPointsAndUpdatesViewAfterDraggingThePlot() {
		final List<String> graphChannels = new ArrayList<>();
		final List<String> rangeChannels = new ArrayList<>();
		page.onRequest(request -> {
			if (request.url().endsWith("/api/metrics.json")) {
				graphChannels.add(request.headers().get("x-query-channel"));
			}
			if (request.url().endsWith("/api/range-stats.json")) {
				rangeChannels.add(request.headers().get("x-query-channel"));
			}
		});
		context.grantPermissions(List.of("clipboard-read", "clipboard-write"));
		page.navigate(baseUrl);
		waitForRunRows(page, 2);
		page.click("#btn-select-all-runs");
		waitForMultiRunGraph(page);
		final var title = page.locator(".graph-block[data-tag-key='loss'] .graph-title");
		title.hover();

		final var panel = page.locator("#tag-info-panel");
		assertThat(panel.getByRole(AriaRole.BUTTON, new Locator.GetByRoleOptions().setName("Copy Name")))
				.isVisible();
		assertThat(panel.locator(".tag-info-name")).hasText("loss");
		assertThat(panel.locator("tbody th")).hasText(new String[] {"run-b", "run-a", "All runs"});
		assertThat(panel.locator("thead tr:first-child th")).hasText(new String[] {"Run", "All"});
		final var labels = panel.locator("thead tr:last-child th");
		assertThat(labels).hasText(new String[] {"Count", "Max", "Mean", "Min", "Std", "Last"});
		final var combined = panel.locator("tbody tr:last-child td");
		assertThat(combined).hasCount(6);
		assertThat(combined.nth(0)).hasText("6");
		assertThat(combined.nth(2)).hasText("2");
		final var runA = panel.locator("tr[data-run-id='run-a'] td");
		assertThat(runA.nth(5)).hasText("3");
		assertTrue(rangeChannels.isEmpty());
		assertThat(page.locator(".graph-stats")).hasCount(0);

		// Plotly が受け取るドラッグで、中央の step だけが入る区間まで狭める。
		final var dragLayer = page.locator(".graph-block[data-tag-key='loss'] .nsewdrag");
		final var box = dragLayer.boundingBox();
		final double x = box.x + box.width * 0.36;
		final double y = box.y + box.height * 0.84;
		page.mouse().move(x, y);
		assertEquals("nsewdrag drag", page.evaluate(
				"([x,y]) => document.elementFromPoint(x,y)?.getAttribute('class')", List.of(x, y)),
				"Drag must start on Plotly's nsewdrag layer");
		page.mouse().down();
		page.mouse().move(box.x + box.width * 0.64, box.y + box.height * 0.96,
				new Mouse.MoveOptions().setSteps(8));
		page.mouse().up();
		assertThat(panel.locator("thead .tag-view-range")).hasText("Viewing (1-1)");
		assertThat(labels).hasText(new String[] {
				"Count", "Max", "Mean", "Min", "Std", "Last",
				"Count", "Max", "Mean", "Min", "Std", "Last"
		});
		assertThat(combined.nth(6)).hasText("2");
		assertThat(combined.nth(8)).hasText("2");
		assertThat(runA.nth(11)).hasText("2");
		assertThat(combined.nth(0)).hasText("6");
		panel.getByText("Copy Name").click();
		assertThat(panel.getByText("Copied")).isVisible();
		assertEquals("loss", page.evaluate("navigator.clipboard.readText()"));

		// ズーム中に開き直した問い合わせも、同じ channel で送る。
		page.keyboard().press("Escape");
		assertThat(panel).isHidden();
		title.hover();
		assertThat(panel).isVisible();
		assertThat(runA.nth(11)).hasText("2");
		final int narrowedRequests = rangeChannels.size();
		assertTrue(narrowedRequests >= 2);

		// 手動指定の横軸でも全点を含む範囲に戻れば、範囲統計の列と要求は不要になる。
		page.evaluate("""
				() => Plotly.relayout(document.querySelector(
				    ".graph-block[data-tag-key='loss'] .js-plotly-plot:not([data-render-stage])"),
				    {"xaxis.range": [-1, 3]})
				""");
		assertThat(panel.locator("thead tr:first-child th")).hasText(new String[] {"Run", "All"});
		assertThat(combined).hasCount(6);
		page.click("#btn-reset-view");
		title.hover();
		assertThat(panel).isVisible();
		assertThat(panel.locator("thead tr:first-child th")).hasText(new String[] {"Run", "All"});
		assertEquals(narrowedRequests, rangeChannels.size());

		page.keyboard().press("Escape");
		assertThat(panel).isHidden();
		page.locator(".graph-block[data-tag-key='score/train'] .graph-title").hover();
		assertThat(panel.locator(".tag-info-name")).hasText("score/train");
		final var missingRun = panel.locator("tr[data-run-id='run-b'] td");
		assertThat(missingRun).hasText(new String[] {"-", "-", "-", "-", "-", "-"});
		assertEquals(narrowedRequests, rangeChannels.size());
		assertTrue(!graphChannels.isEmpty());
		assertTrue(rangeChannels.get(0) != null && !rangeChannels.get(0).isBlank());
		assertTrue(rangeChannels.stream().allMatch(rangeChannels.get(0)::equals));
		assertTrue(graphChannels.stream().noneMatch(rangeChannels.get(0)::equals));
	}


	@Test
	void tappingTheTagTitleOpensThePanelWithoutSelectingTheTitle() {
		reopenPage(new Browser.NewContextOptions().setViewportSize(390, 844)
				.setIsMobile(true).setHasTouch(true));
		page.navigate(baseUrl);
		waitForRunRows(page, 2);
		page.click("#btn-screenshot-toggle");
		final var title = page.locator(".graph-block[data-tag-key='loss'] .graph-title");
		final var panel = page.locator("#tag-info-panel");
		title.tap();
		assertThat(panel).isVisible();
		assertThat(panel.locator(".tag-info-name")).hasText("loss");
		assertThat(title).hasCSS("user-select", "none");
		assertThat(panel.locator(".tag-info-name")).hasCSS("user-select", "all");
		assertEquals("", page.evaluate("window.getSelection().toString()"));

		// パネルの外側をタップして閉じ、tag名から始めたスワイプでは再表示しない。
		page.locator("#screenshot-header").tap();
		assertThat(panel).isHidden();
		final var bounds = title.boundingBox();
		final double x = bounds.x + bounds.width / 2;
		final double y = bounds.y + bounds.height / 2;
		dispatchTouchSwipe(page, x, y, y + 90);
		assertThat(panel).isHidden();

		// 長押し中にもブラウザがグラフの tag 名を選択しないことを確かめる。
		final var hold = context.newCDPSession(page);
		try {
			dispatchTouchEvent(hold, "touchStart", x, y);
			page.waitForTimeout(700);
			assertEquals("", page.evaluate("window.getSelection().toString()"));
		} finally {
			dispatchTouchEvent(hold, "touchEnd", x, y);
			hold.detach();
		}
	}

	@Test
	void graphRendersWithoutAnyRequestOutsideTheViewer() {
		final List<String> external = new ArrayList<>();
		page.route("**/*", route -> {
			if (route.request().url().startsWith(baseUrl + "/")) route.resume();
			else {
				external.add(route.request().url());
				route.abort();
			}
		});
		page.navigate(baseUrl);
		waitForRunRows(page, 2);
		assertEquals(List.of(), external);
		waitForGraph(page);
		assertEquals("2.35.3", page.evaluate("Plotly.version"));
		assertEquals(List.of(), external);
	}


	@Test
	void narrowConfigKeepsTabsSearchAndDiffOperableWhileTheTableScrolls() {
		page.setViewportSize(390, 844);
		page.navigate(baseUrl);
		waitForRunRows(page, 2);
		page.click("#btn-select-all-runs");
		page.click("#tab-config", new Page.ClickOptions().setTimeout(2000));
		assertThat(page.locator("#config-table tbody tr")).hasCount(3);
		final var search = page.locator("#config-filter");
		final var searchBox = search.boundingBox();
		assertTrue(searchBox.width >= 100, "The search field needs space to display the query");
		final int scrollLeft = ((Number) page.locator("#config-table-scroll").evaluate(
				"el => { el.scrollLeft = el.scrollWidth; return el.scrollLeft; }")).intValue();
		assertTrue(scrollLeft > 0);
		assertEquals(searchBox.x, search.boundingBox().x);
		search.fill("beta");
		page.click("#btn-config-diff", new Page.ClickOptions().setTimeout(2000));
		assertThat(page.locator("#config-table tbody th")).hasText("alpha.beta");
		page.click("#tab-graphs", new Page.ClickOptions().setTimeout(2000));
		assertThat(page.locator("#main-area")).isVisible();
	}


	@Test
	void tagWithoutCommittedPointsDistinguishesEmptyAndPendingRuns() {
		page.route("**/api/runs.json", route -> fulfillJson(route, """
				{"runs":[
				  {"id":"run-a","ingest":{"state":"ready","percentage":100},
				   "tags":[{"key":"empty","type":"scalar","status":"error"}]},
				  {"id":"run-b","ingest":{"state":"pending","percentage":0},"tags":[]}
				]}
				"""));
		page.navigate(baseUrl);
		waitForRunRows(page, 2);
		page.click("#btn-select-all-runs");
		page.locator(".graph-block[data-tag-key='empty'] .graph-title").hover();
		final var panel = page.locator("#tag-info-panel");
		assertThat(panel.locator("thead tr:first-child th")).hasText(new String[] {"Run", "All"});
		assertThat(panel.locator("tr[data-run-id='run-a'] td")).hasText(
				new String[] {"-", "-", "-", "-", "-", "-"});
		assertThat(panel.locator("tr[data-run-id='run-b'] td")).hasText(
				new String[] {"", "", "", "", "", ""});
		assertThat(panel.locator("tbody tr:last-child td")).hasText(
				new String[] {"", "", "", "", "", ""});
	}


	@Test
	void configWrapsLongTextAndOnlyTheTableScrollsHorizontally() {
		page.setViewportSize(950, 720);
		page.route("**/api/config.json", route -> fulfillJson(route, """
				{"runs":[
				  {"runId":"run-b","status":"ok","entries":[{"key":"%s","value":"%s"}]},
				  {"runId":"run-a","status":"ok","entries":[{"key":"%s","value":"short"}]}
				]}
				""".formatted("config.".repeat(52), "long-value".repeat(80), "config.".repeat(52))));
		page.navigate(baseUrl);
		waitForRunRows(page, 2);
		page.click("#btn-select-all-runs");
		page.click("#tab-config");
		assertThat(page.locator("#config-table tbody tr")).hasCount(1);
		assertTrue(Boolean.TRUE.equals(page.locator("#config-table-scroll").evaluate(
				"el => el.scrollWidth <= el.clientWidth")));

		page.setViewportSize(660, 720);
		final var search = page.locator("#config-filter");
		final var diff = page.locator("#btn-config-diff");
		final double searchLeft = search.boundingBox().x;
		final double diffLeft = diff.boundingBox().x;
		final int scrollLeft = ((Number) page.locator("#config-table-scroll").evaluate(
				"el => { el.scrollLeft = el.scrollWidth; return el.scrollLeft; }")).intValue();
		assertTrue(scrollLeft > 0);
		assertEquals(searchLeft, search.boundingBox().x);
		assertEquals(diffLeft, diff.boundingBox().x);
		search.fill("config");
		diff.click();
		assertThat(page.locator("#config-table tbody tr")).hasCount(1);
	}

	private static Path createRuns() {
		try {
			final Path root = Path.of("target", "run-comparison-" + System.nanoTime())
					.toAbsolutePath().resolve("comparison").resolve("runs");
			writeRun(root.resolve("run-a"), "alpha.beta = 1.00\nmissing =\nsame = hello\n");
			writeRun(root.resolve("run-b"), "alpha.beta = 1.0\nsame = hello\n");
			return root;
		} catch (IOException e) {
			throw new ExceptionInInitializerError(e);
		}
	}

	private static void writeRun(Path run, String config) throws IOException {
		Files.createDirectories(run.resolve("config"));
		Files.writeString(run.resolve("config/config_data.txt"), config);
		final String training = run.endsWith("run-a")
				? "{\"type\":\"scalar\",\"tag\":\"score/train\",\"step\":1,\"value\":4}\n" : "";
		Files.writeString(run.resolve("metrics.jsonl"), """
				{"type":"scalar","tag":"loss","step":0,"value":1}
				{"type":"scalar","tag":"loss","step":1,"value":2}
				{"type":"scalar","tag":"loss","step":2,"value":3}
				""" + training);
	}
}
