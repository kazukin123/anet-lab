package io.github.kazukin123.anetlab.metricsviewer.view;

import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.*;
import static org.junit.jupiter.api.Assertions.*;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import com.microsoft.playwright.options.RequestOptions;
import com.google.gson.JsonParser;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
class TagSetPlaywrightTest extends MetricsViewerPlaywrightTestSupport {

	private static final Path WORKSPACES = createWorkspaces();

	@DynamicPropertySource
	static void properties(DynamicPropertyRegistry registry) {
		registry.add("metricsviewer.workspaces-dir", () -> WORKSPACES.toString());
		registry.add("metricsviewer.cache-memory-mb", () -> "0");
	}

	private static Path createWorkspaces() {
		try {
			Path root = Files.createTempDirectory("prd111-workspaces-");
			Files.createDirectories(root.resolve("_default/config"));
			return root;
		} catch (IOException exception) {
			throw new IllegalStateException(exception);
		}
	}

	@BeforeEach
	void clearTagSets() {
		for (var info : JsonParser.parseString(context.request().get(baseUrl + "/api/tag-sets.json").text())
				.getAsJsonObject().getAsJsonArray("tagSets")) {
			context.request().post(baseUrl + "/api/tag-sets/delete", RequestOptions.create()
					.setData(Map.of("name", info.getAsJsonObject().get("name").getAsString())));
		}
		page.setDefaultTimeout(10000);
	}

	private void openFixture() {
		page.route("**/api/runs.json", route -> fulfillJson(route, splitTagRunsJson()));
		page.route("**/api/metrics.json", route -> fulfillJson(route, "{\"data\":[]}"));
		page.navigate(baseUrl);
		waitForTag(page, TAG_B);
		selectSingleRun(page, "run_a");
		waitForTag(page, TAG_A);
	}

	private void createSet(String name, List<String> tags) {
		assertEquals(200, context.request().post(baseUrl + "/api/tag-sets/create",
				RequestOptions.create().setData(Map.of("name", name, "tags", tags))).status());
	}

	private Object activeTags() {
		return page.evaluate("JSON.stringify(JSON.parse(localStorage.getItem('anet.metricsviewer.activeTags')).sort())");
	}

	@Test
	void editingASetCanBeSavedRevertedRenamedDeletedAndDeselected() {
		// 見えないtagを含むセットを呼び、保存直後には差が無い状態を用意する。
		createSet("A", List.of(TAG_A, "hidden/tag"));
		openFixture();
		page.click("#btn-tag-set");
		page.locator("#tag-set-menu .tag-set-name").getByText("A", new com.microsoft.playwright.Locator.GetByTextOptions().setExact(true)).click();
		assertEquals("[\"hidden/tag\",\"" + TAG_A + "\"]", activeTags());
		assertFalse(page.locator("#btn-tag-set-save").isEnabled());
		// 可視tagを外した差を確認し、Revertで元へ戻してから同じ差をSaveで確定する。
		page.locator("#tag-list li").filter(new com.microsoft.playwright.Locator.FilterOptions().setHasText(TAG_A)).click();
		assertTrue(page.locator("#btn-tag-set").textContent().contains("*"));
		assertEquals("Modified: +0 −1", page.locator("#btn-tag-set").getAttribute("title"));
		page.click("#btn-tag-set-revert");
		page.waitForFunction("!document.querySelector('#btn-tag-set').textContent.includes('*')");
		page.locator("#tag-list li").filter(new com.microsoft.playwright.Locator.FilterOptions().setHasText(TAG_A)).click();
		page.click("#btn-tag-set-save");
		page.waitForFunction("!document.querySelector('#btn-tag-set').textContent.includes('*')");
		assertEquals("[\"hidden/tag\"]", activeTags());
		// 再読み込みで名前を復元し、自分の改名では消失通知が出ないことを確かめる。
		page.reload();
		waitForTag(page, TAG_B);
		assertEquals("\"A\"", page.evaluate("localStorage.getItem('anet.metricsviewer.tagSet')"));
		page.click("#btn-tag-set");
		page.locator("#tag-set-menu button[title='Rename tag set']").click();
		page.locator("#tag-set-menu input").fill(" B ");
		page.locator("#tag-set-menu input").press("Enter");
		page.waitForFunction("localStorage.getItem('anet.metricsviewer.tagSet') === '\"B\"'");
		assertEquals(List.of(), page.locator(".toast").allTextContents());
		// Deselectは名前だけを外し、不可視tagを含む選択を維持する。
		page.getByText("Deselect tag set", new com.microsoft.playwright.Page.GetByTextOptions().setExact(true)).click();
		assertNull(page.evaluate("localStorage.getItem('anet.metricsviewer.tagSet')"));
		assertEquals("[\"hidden/tag\"]", activeTags());
		// 選び直して確認付きで削除し、選択維持と自分の削除での通知抑止を確かめる。
		page.click("#btn-tag-set");
		page.locator("#tag-set-menu .tag-set-name").click();
		page.click("#btn-tag-set");
		page.locator("#tag-set-menu button[title='Delete tag set']").click();
		page.locator("#tag-set-menu").getByRole(com.microsoft.playwright.options.AriaRole.BUTTON,
				new com.microsoft.playwright.Locator.GetByRoleOptions().setName("Delete").setExact(true)).click();
		page.waitForFunction("localStorage.getItem('anet.metricsviewer.tagSet') === null");
		assertEquals("[\"hidden/tag\"]", activeTags());
		assertEquals(List.of(), page.locator(".toast").allTextContents());
	}

	@Test
	void saveAsPersistsSelectionAndCallingTheSetRestoresIt() throws IOException {
		// Run metadataだけを置き換え、タグセットは実サーバーと実ファイルを通す。
		page.route("**/api/runs.json", route -> fulfillJson(route, splitTagRunsJson()));
		page.route("**/api/metrics.json", route -> fulfillJson(route, "{\"data\":[]}"));
		page.navigate(baseUrl);
		waitForTag(page, TAG_B);
		selectSingleRun(page, "run_a");
		waitForTag(page, TAG_A);
		final Object saved = page.evaluate("JSON.stringify(JSON.parse(localStorage.getItem('anet.metricsviewer.activeTags')).sort())");

		page.click("#btn-tag-set");
		page.getByText("+ Save current selection as…", new com.microsoft.playwright.Page.GetByTextOptions().setExact(true)).click();
		page.locator("#tag-set-menu input").fill("Selection A");
		page.locator("#tag-set-menu").getByRole(com.microsoft.playwright.options.AriaRole.BUTTON,
				new com.microsoft.playwright.Locator.GetByRoleOptions().setName("Save").setExact(true)).click();
		page.waitForFunction("localStorage.getItem('anet.metricsviewer.tagSet') === '\"Selection A\"'");
		assertTrue(Files.readString(WORKSPACES.resolve("metrics_viewer_tag_sets.json")).contains("Selection A"));

		// 選択を崩してから行を呼び直すと、保存した全体へ戻る。
		page.click("#btn-clear-all");
		page.click("#btn-tag-set");
		page.locator("#tag-set-menu .tag-set-name").filter(
				new com.microsoft.playwright.Locator.FilterOptions().setHasText("Selection A")).click();
		assertEquals(saved, page.evaluate("JSON.stringify(JSON.parse(localStorage.getItem('anet.metricsviewer.activeTags')).sort())"));
	}

	@Test
	void keyboardCallsRowsAndEscapeCancelsOnlyTheCurrentEdit() {
		// 行の移動先を区別できるよう、中身の違う2セットを用意する。
		createSet("A", List.of(TAG_A));
		createSet("B", List.of(TAG_C));
		openFixture();
		// 一覧の取得完了後にキーで先頭行を呼び、続けて2行目への移動も確かめる。
		page.click("#btn-tag-set");
		page.locator("#tag-set-menu").waitFor();
		page.keyboard().press("ArrowDown");
		page.keyboard().press("Enter");
		page.waitForFunction("localStorage.getItem('anet.metricsviewer.tagSet') === '\"A\"'");
		assertEquals("[\"" + TAG_A + "\"]", activeTags());
		page.click("#btn-tag-set");
		page.locator("#tag-set-menu").waitFor();
		page.keyboard().press("ArrowDown");
		page.keyboard().press("ArrowDown");
		page.keyboard().press("Enter");
		page.waitForFunction("localStorage.getItem('anet.metricsviewer.tagSet') === '\"B\"'");
		// 入力欄のEscは編集だけを取り消し、次のEscで一覧全体を閉じる。
		page.click("#btn-tag-set");
		page.getByText("+ Save current selection as…", new com.microsoft.playwright.Page.GetByTextOptions().setExact(true)).click();
		page.locator("#tag-set-menu input").fill("cancelled");
		page.locator("#tag-set-menu input").press("Escape");
		assertTrue(page.locator("#tag-set-menu").isVisible());
		assertEquals(0, page.locator("#tag-set-menu input").count());
		page.keyboard().press("Escape");
		assertFalse(page.locator("#tag-set-menu").isVisible());
	}

	@org.junit.jupiter.params.ParameterizedTest
	@org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
	void longNamesKeepControlsFixedAndSaveAsShowsHiddenCounts(boolean mobile) {
		// PCとタッチ端末の幅で、長い名前と可視・不可視tagを含む同じ条件を用意する。
		if (mobile) reopenPage(new com.microsoft.playwright.Browser.NewContextOptions()
				.setViewportSize(390, 640).setIsMobile(true).setHasTouch(true));
		String name = "name/".repeat(12) + "長い名前";
		createSet(name, List.of(TAG_A, TAG_B, TAG_C, "hidden/1", "hidden/2"));
		openFixture();
		page.click("#btn-select-all-runs");
		page.click("#btn-tag-set");
		page.locator("#tag-set-menu .tag-set-name").click();
		Object fixed = page.locator("#btn-tag-set-save").boundingBox().x;
		String disabledColor = page.locator("#btn-tag-set-save").evaluate("el => getComputedStyle(el).color").toString();
		// 一覧の見える数とSave Asの全体数を、同じ選択のまま確認する。
		page.click("#btn-tag-set");
		assertEquals("3/5", page.locator(".tag-set-count").textContent());
		page.getByText("+ Save current selection as…", new com.microsoft.playwright.Page.GetByTextOptions().setExact(true)).click();
		assertEquals("5 tags (2 not in the list)", page.locator(".tag-set-save-count").textContent());
		page.locator("#tag-set-menu input").press("Escape");
		page.keyboard().press("Escape");
		// 差を作っても位置を保ち、押せる配色と名前だけの省略を確認する。
		page.locator("#tag-list li").filter(new com.microsoft.playwright.Locator.FilterOptions().setHasText(TAG_A)).click();
		assertTrue(page.locator("#btn-tag-set").textContent().contains("*"));
		assertEquals(fixed, page.locator("#btn-tag-set-save").boundingBox().x);
		assertNotEquals(disabledColor, page.locator("#btn-tag-set-save").evaluate("el => getComputedStyle(el).color").toString());
		assertEquals(Boolean.TRUE, page.evaluate("""
				() => {
					const button = document.querySelector('#btn-tag-set');
					const label = button.querySelector('.tag-set-label');
					return button.scrollWidth <= button.clientWidth + 1
						&& label.scrollWidth > label.clientWidth
						&& getComputedStyle(label).textOverflow === 'ellipsis';
				}
				"""));
		// body直下の一覧が画面内に収まることを確認し、目視用の画像も保存する。
		page.click("#btn-tag-set");
		page.locator("#tag-set-menu").waitFor();
		assertEquals(Boolean.TRUE, page.evaluate("""
				() => {
					const menu = document.querySelector('#tag-set-menu');
					const r = menu.getBoundingClientRect();
					return menu.parentElement === document.body && r.left >= 0 && r.right <= innerWidth
						&& r.top >= 0 && r.bottom <= innerHeight && menu.scrollWidth <= menu.clientWidth + 1;
				}
				"""));
		page.screenshot(new com.microsoft.playwright.Page.ScreenshotOptions()
				.setPath(Path.of("target/prd111-" + (mobile ? "mobile" : "desktop") + ".png")));
	}

	@Test
	void failedInitialAndMenuFetchKeepTheNameAndNeverShowAnEmptyList() {
		// 復元する名前と選択を用意し、初回と開くときの両方の取得を失敗させる。
		context.addInitScript("""
				localStorage.setItem('anet.metricsviewer.tagSet', '"A"');
				localStorage.setItem('anet.metricsviewer.activeTags', '["hidden"]');
				localStorage.setItem('anet.metricsviewer.knownTags', '["tag/a","tag/b","tag/c"]');
				""");
		page.route("**/api/tag-sets.json", route -> route.fulfill(new com.microsoft.playwright.Route.FulfillOptions()
				.setStatus(503).setContentType("application/json").setBody("{}")));
		openFixture();
		// 一覧が一度も取れていない場合は開かず、操作を無効にして復元した状態を保つ。
		page.click("#btn-tag-set");
		page.waitForFunction("document.querySelectorAll('.toast').length >= 2");
		assertFalse(page.locator("#tag-set-menu").isVisible());
		assertFalse(page.locator("#btn-tag-set-save").isEnabled());
		assertFalse(page.locator("#btn-tag-set-revert").isEnabled());
		assertEquals("\"A\"", page.evaluate("localStorage.getItem('anet.metricsviewer.tagSet')"));
		assertEquals("[\"hidden\"]", activeTags());
	}

	@Test
	void failedRevertAndMutationPreserveSelectionAndConflictsKeepTheInput() {
		createSet("A", List.of(TAG_A));
		openFixture();
		page.click("#btn-tag-set");
		page.locator(".tag-set-name").click();
		page.locator("#tag-list li").filter(new com.microsoft.playwright.Locator.FilterOptions().setHasText(TAG_C)).click();
		Object before = activeTags();
		page.route("**/api/tag-sets.json", route -> route.abort());
		page.click("#btn-tag-set-revert");
		page.waitForFunction("document.querySelectorAll('.toast').length > 0");
		assertEquals(before, activeTags());
		// 開くときだけは取得済みの一覧を使い、書き込み失敗でも選択は変えない。
		page.click("#btn-tag-set");
		page.locator("#tag-set-menu").waitFor();
		page.getByText("+ Save current selection as…", new com.microsoft.playwright.Page.GetByTextOptions().setExact(true)).click();
		page.locator("#tag-set-menu input").fill("A");
		page.locator("#tag-set-menu input").press("Enter");
		page.waitForFunction("[...document.querySelectorAll('.toast')].some(el => el.textContent.includes('already exists'))");
		assertEquals("A", page.locator("#tag-set-menu input").inputValue());
		assertEquals(before, activeTags());
		page.locator("#tag-set-menu input").fill("Network failure");
		page.route("**/api/tag-sets/create", route -> route.abort());
		page.locator("#tag-set-menu input").press("Enter");
		page.waitForFunction("[...document.querySelectorAll('.toast')].some(el => el.textContent.includes('fetch'))");
		assertEquals(before, activeTags());
		assertEquals("\"A\"", page.evaluate("localStorage.getItem('anet.metricsviewer.tagSet')"));
	}

	@Test
	void anotherBrowserSeesCreatedSetsAndDetectsDeletionOnOpenOrRevert() {
		createSet("A", List.of(TAG_A));
		openFixture();
		// 別のlocalStorageを持つ実ブラウザ文脈から同じサーバーを使う。
		try (var other = context.browser().newContext()) {
			var second = other.newPage();
			second.route("**/api/runs.json", route -> fulfillJson(route, splitTagRunsJson()));
			second.route("**/api/metrics.json", route -> fulfillJson(route, "{\"data\":[]}"));
			second.navigate(baseUrl);
			waitForTag(second, TAG_B);
			second.click("#btn-tag-set");
			second.locator(".tag-set-name").click();
			createSet("B", List.of(TAG_C));
			second.click("#btn-tag-set");
			second.getByText("B", new com.microsoft.playwright.Page.GetByTextOptions().setExact(true)).waitFor();
			second.keyboard().press("Escape");
			other.request().post(baseUrl + "/api/tag-sets/delete", RequestOptions.create().setData(Map.of("name", "A")));
			second.click("#btn-tag-set");
			second.waitForFunction("localStorage.getItem('anet.metricsviewer.tagSet') === null");
			assertTrue(second.locator(".toast").allTextContents().stream().anyMatch(text -> text.contains("no longer exists")));
			second.locator(".tag-set-name").click();
			selectSingleRun(second, "run_a");
			waitForTag(second, TAG_C);
			second.click("#btn-clear-all");
			context.request().post(baseUrl + "/api/tag-sets/delete", RequestOptions.create().setData(Map.of("name", "B")));
			second.click("#btn-tag-set-revert");
			second.waitForFunction("localStorage.getItem('anet.metricsviewer.tagSet') === null");
		}
	}

	@Test
	void reloadActivatesNewTagsFromRealMetricsAndMarksTheSelectedSetModified() throws IOException {
		// マスタへ実際に追記し、取り込み・metadata・新規tag自動選択を通す。
		Path run = WORKSPACES.resolve("_default/runs/run_tmp_prd111");
		Files.createDirectories(run);
		Path metrics = run.resolve("metrics.jsonl");
		Files.writeString(metrics, "{\"type\":\"scalar\",\"tag\":\"real/old\",\"step\":0,\"value\":1}\n");
		page.navigate(baseUrl);
		// 進捗pollはtagを再取得しない既存契約なので、取り込み完了後にReloadする。
		page.waitForCondition(() -> context.request().get(baseUrl + "/api/runs.json").text().contains("\"key\":\"real/old\""));
		page.click("#btn-reload");
		try { waitForTag(page, "real/old"); }
		catch (com.microsoft.playwright.TimeoutError error) {
			fail("Real run metadata: " + context.request().get(baseUrl + "/api/runs.json").text()
					+ " UI: " + page.locator("#run-list").textContent(), error);
		}
		createSet("Real", List.of("real/old"));
		page.click("#btn-tag-set");
		page.locator(".tag-set-name").click();
		Files.writeString(metrics, "{\"type\":\"scalar\",\"tag\":\"real/new\",\"step\":1,\"value\":2}\n",
				java.nio.file.StandardOpenOption.APPEND);
		page.waitForCondition(() -> context.request().get(baseUrl + "/api/runs.json").text().contains("\"key\":\"real/new\""));
		page.click("#btn-reload");
		waitForTag(page, "real/new");
		assertTrue(page.locator("#btn-tag-set").textContent().contains("*"));
		assertTrue(activeTags().toString().contains("real/new"));
	}

	@Test
	void startupRestoresUnsavedChangesAndWorkspaceSwitchKeepsTheSelectedName() throws IOException {
		createSet("A", List.of(TAG_A, "hidden"));
		context.addInitScript("""
				localStorage.setItem('anet.metricsviewer.tagSet', '"A"');
				localStorage.setItem('anet.metricsviewer.activeTags', '["tag/c","hidden"]');
				localStorage.setItem('anet.metricsviewer.knownTags', '["tag/a","tag/b","tag/c"]');
				""");
		openFixture();
		assertTrue(page.locator("#btn-tag-set").textContent().contains("*"));
		assertEquals("[\"hidden\",\"" + TAG_C + "\"]", activeTags());
		page.reload();
		waitForTag(page, TAG_B);
		assertTrue(page.locator("#btn-tag-set").textContent().contains("*"));
		assertEquals("\"A\"", page.evaluate("localStorage.getItem('anet.metricsviewer.tagSet')"));
		Files.createDirectories(WORKSPACES.resolve("_new/config"));
		page.locator("#workspace-selector").focus();
		page.selectOption("#workspace-selector", "_new");
		page.waitForFunction("document.querySelector('#workspace-selector').value === '_new'");
		assertEquals("\"A\"", page.evaluate("localStorage.getItem('anet.metricsviewer.tagSet')"));
		// サーバーのcurrentはプロセス共有なので、次テストの前提へ戻す。
		context.request().post(baseUrl + "/api/workspace", RequestOptions.create().setData(Map.of("name", "_default")));
	}

	@org.junit.jupiter.params.ParameterizedTest
	@org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
	void switchingFromAToBNeedsExactlyTwoClicksOrTaps(boolean touch) {
		if (touch) reopenPage(new com.microsoft.playwright.Browser.NewContextOptions()
				.setViewportSize(390, 640).setIsMobile(true).setHasTouch(true));
		createSet("A", List.of(TAG_A, "hidden/A"));
		createSet("B", List.of(TAG_C, "hidden/B"));
		openFixture();
		page.click("#btn-tag-set");
		page.locator(".tag-set-name").getByText("A", new com.microsoft.playwright.Locator.GetByTextOptions().setExact(true)).click();
		setToggle(page, "#btn-selected-only", true);
		// A選択済みから入力数を数える。captureでメニュー内の伝播停止にも左右されない。
		page.evaluate("""
				window.tagSetSwitchInputs = 0;
				document.addEventListener('click', () => window.tagSetSwitchInputs++, true);
				""");
		if (touch) page.tap("#btn-tag-set");
		else page.click("#btn-tag-set");
		var row = page.locator(".tag-set-name").getByText("B",
				new com.microsoft.playwright.Locator.GetByTextOptions().setExact(true));
		if (touch) row.tap();
		else row.click();
		assertEquals(2, ((Number) page.evaluate("window.tagSetSwitchInputs")).intValue());
		assertEquals("\"B\"", page.evaluate("localStorage.getItem('anet.metricsviewer.tagSet')"));
		assertEquals("[\"hidden/B\",\"" + TAG_C + "\"]", activeTags());
		assertEquals(List.of(TAG_C), page.locator("#tag-list li:visible").allTextContents());
		assertTrue(isToggleOn(page, "#btn-selected-only"));
	}

	@Test
	void rowsUseUtf16OrderAndCountsFollowTheSelectedRuns() {
		// 名前順と異なる作成順で、Runごとに見える数が変わるセットを用意する。
		createSet("可塑性", List.of());
		createSet("eval 成績", List.of(TAG_A, TAG_B));
		createSet("PER", List.of(TAG_B, "hidden"));
		createSet("Atari スコア", List.of(TAG_A, TAG_C));
		openFixture();
		// run_aでの名前順と可視数を先に確かめ、切替前の基準にする。
		page.click("#btn-tag-set");
		page.locator("#tag-set-menu").waitFor();
		assertEquals(List.of("Atari スコア", "PER", "eval 成績", "可塑性"),
				page.locator(".tag-set-name").allTextContents());
		assertEquals(List.of("2/2", "0/2", "1/2", "0/0"), page.locator(".tag-set-count").allTextContents());
		// run_bへ切り替えて開き直すと、順序は同じまま可視数だけが変わる。
		selectSingleRun(page, "run_b");
		waitForTag(page, TAG_B);
		page.click("#btn-tag-set");
		page.locator("#tag-set-menu").waitFor();
		assertEquals(List.of("Atari スコア", "PER", "eval 成績", "可塑性"),
				page.locator(".tag-set-name").allTextContents());
		assertEquals(List.of("0/2", "1/2", "1/2", "0/0"), page.locator(".tag-set-count").allTextContents());
	}

	@Test
	void visibleOnlyDifferencesAndSaveAsCountsOmitParentheses() {
		// 1本だけのセットを呼び、可視tagを足す前の比較基準を決める。
		createSet("A", List.of(TAG_A));
		openFixture();
		page.click("#btn-tag-set");
		page.locator(".tag-set-name").click();
		// 全Runのtagを選び、差のすべてが可視ならtitleの括弧が省かれることを確かめる。
		page.click("#btn-select-all-runs");
		page.click("#btn-select-all");
		assertEquals("Modified: +2 −0", page.locator("#btn-tag-set").getAttribute("title"));
		// 同じ選択のSave Asでも不可視数の括弧を付けず、全体数だけを出す。
		page.click("#btn-tag-set");
		page.getByText("+ Save current selection as…", new com.microsoft.playwright.Page.GetByTextOptions().setExact(true)).click();
		assertEquals("3 tags", page.locator(".tag-set-save-count").textContent());
	}
}
