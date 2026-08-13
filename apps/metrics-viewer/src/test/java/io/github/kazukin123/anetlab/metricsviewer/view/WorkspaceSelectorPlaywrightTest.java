package io.github.kazukin123.anetlab.metricsviewer.view;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

import com.google.gson.JsonParser;
import com.microsoft.playwright.Page;
import com.microsoft.playwright.options.WaitUntilState;

@SpringBootTest(
		webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT,
		properties = "metricsviewer.workspaces-dir=target/playwright-test-empty-workspaces")
class WorkspaceSelectorPlaywrightTest extends MetricsViewerPlaywrightTestSupport {

	@Test
	void changingWorkspaceResetsWorkspaceStateAndSelectsLatestRun() {
		final AtomicReference<String> current = new AtomicReference<>("ws-a");
		final AtomicInteger switchRequests = new AtomicInteger();
		routeWorkspaceApi(current, switchRequests);

		page.navigate(baseUrl + "/?workspaceSwitch=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		page.waitForFunction("document.querySelector('#run-list .run-row.active')?.dataset.runId === 'run-a'");
		page.evaluate("""
			() => {
				app.viewports.set('old-tag', { autorange: false, from: 1, to: 2 });
				app.hiddenLegendSeries.set('old-tag', new Set(['run-a']));
				app.runColorMap.set('old-only-run', '#ffffff');
			}
			""");

		page.selectOption("#workspace-selector", "ws-b");
		page.waitForFunction("document.querySelector('#run-list .run-row.active')?.dataset.runId === 'run-b'");

		assertEquals(1, switchRequests.get());
		assertEquals("ws-b", page.evaluate("localStorage.getItem('anet.metricsviewer.workspace')"));
		assertEquals(false, page.evaluate("app.runColorMap.has('old-only-run')"));
		assertEquals(0, page.evaluate("app.viewports.size"));
		assertEquals(0, page.evaluate("app.hiddenLegendSeries.size"));
		assertEquals("run-b", page.getAttribute("#run-list .run-row.active", "data-run-id"));
	}

	@Test
	void validSavedWorkspaceIsRestoredBeforeRunsAreLoaded() {
		final AtomicReference<String> current = new AtomicReference<>("ws-a");
		final AtomicInteger switchRequests = new AtomicInteger();
		routeWorkspaceApi(current, switchRequests);
		context.addInitScript("localStorage.setItem('anet.metricsviewer.workspace', 'ws-b')");

		page.navigate(baseUrl + "/?workspaceRestore=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		page.waitForFunction("document.querySelector('#run-list .run-row.active')?.dataset.runId === 'run-b'");

		assertEquals(1, switchRequests.get());
		assertEquals("ws-b", page.inputValue("#workspace-selector"));
	}

	@Test
	void unknownSavedWorkspaceFallsBackWithoutPosting() {
		final AtomicReference<String> current = new AtomicReference<>("ws-a");
		final AtomicInteger switchRequests = new AtomicInteger();
		routeWorkspaceApi(current, switchRequests);
		context.addInitScript("localStorage.setItem('anet.metricsviewer.workspace', 'missing')");

		page.navigate(baseUrl + "/?workspaceFallback=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		page.waitForFunction("document.querySelector('#run-list .run-row.active')?.dataset.runId === 'run-a'");

		assertEquals(0, switchRequests.get());
		assertEquals("ws-a", page.evaluate("localStorage.getItem('anet.metricsviewer.workspace')"));
		assertEquals("ws-a", page.inputValue("#workspace-selector"));
	}

	@Test
	void missingWorkspaceIsRemovedAndPreviousSelectionIsRestoredWithToast() {
		final AtomicReference<String> current = new AtomicReference<>("ws-b");
		final AtomicReference<List<String>> workspaces =
				new AtomicReference<>(List.of("ws-a", "ws-b"));
		final AtomicInteger switchRequests = new AtomicInteger();
		routeWorkspaceApi(current, workspaces, switchRequests);

		page.navigate(baseUrl + "/?workspaceMissing=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		page.waitForFunction("document.querySelector('#run-list .run-row.active')?.dataset.runId === 'run-b'");
		workspaces.set(List.of("ws-b"));

		page.selectOption("#workspace-selector", "ws-a");
		page.waitForFunction("document.querySelector('[role=alert]')?.textContent.includes('no longer exists')");

		assertEquals(1, switchRequests.get());
		assertEquals("ws-b", page.inputValue("#workspace-selector"));
		assertEquals(false, page.evaluate("""
				Array.from(document.querySelectorAll('#workspace-selector option'))
					.some(option => option.value === 'ws-a')
				"""));
		assertEquals("ws-b", page.evaluate("localStorage.getItem('anet.metricsviewer.workspace')"));
	}

	@Test
	void missingCurrentWorkspaceIsShownAsDisabledAndWarnedOnce() {
		final AtomicReference<String> current = new AtomicReference<>("renamed-away");
		final AtomicReference<List<String>> workspaces =
				new AtomicReference<>(List.of("ws-b"));
		final AtomicInteger switchRequests = new AtomicInteger();
		routeWorkspaceApi(current, workspaces, switchRequests);

		page.navigate(baseUrl + "/?workspaceCurrentMissing=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		page.waitForFunction("document.querySelector('[role=alert]')?.textContent.includes('Current workspace')");

		assertEquals("renamed-away", page.inputValue("#workspace-selector"));
		assertEquals(true, page.evaluate(
				"document.querySelector('#workspace-selector option:checked').disabled"));
		assertEquals("(missing) renamed-away",
				page.textContent("#workspace-selector option:checked"));
		page.focus("#workspace-selector");
		page.waitForTimeout(100);
		assertEquals(1, ((Number) page.evaluate(
				"document.querySelectorAll('[role=alert]').length")).intValue());

		page.selectOption("#workspace-selector", "ws-b");
		page.waitForFunction("document.querySelector('#run-list .run-row.active')"
				+ "?.dataset.runId === 'run-b'");
		assertEquals(1, switchRequests.get());
		assertEquals("ws-b", page.inputValue("#workspace-selector"));
	}

	@Test
	void nonMissingSwitchFailureRestoresSelectionWithoutRemovingOption() {
		final AtomicReference<String> current = new AtomicReference<>("ws-b");
		final AtomicReference<List<String>> workspaces =
				new AtomicReference<>(List.of("ws-a", "ws-b"));
		final AtomicInteger switchRequests = new AtomicInteger();
		routeWorkspaceApi(
				current, workspaces, switchRequests, new AtomicReference<>(500));

		page.navigate(baseUrl + "/?workspaceSwitchFailure=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		page.waitForFunction("document.querySelector('#workspace-selector')?.value === 'ws-b'");

		page.selectOption("#workspace-selector", "ws-a");
		page.waitForFunction("document.querySelector('[role=alert]')?.textContent"
				+ ".includes('Workspace switch failed')");

		assertEquals(1, switchRequests.get());
		assertEquals("ws-b", page.inputValue("#workspace-selector"));
		assertEquals(true, page.evaluate("Array.from(document.querySelectorAll("
				+ "'#workspace-selector option')).some(option => option.value === 'ws-a')"));
		assertEquals("ws-b", page.evaluate(
				"localStorage.getItem('anet.metricsviewer.workspace')"));
	}

	@Test
	void refreshFailureAfterSuccessfulSwitchKeepsTheNewWorkspace() {
		final AtomicReference<String> current = new AtomicReference<>("ws-a");
		final AtomicReference<List<String>> workspaces =
				new AtomicReference<>(List.of("ws-a", "ws-b"));
		final AtomicInteger switchRequests = new AtomicInteger();
		final AtomicBoolean failRuns = new AtomicBoolean();
		routeWorkspaceApi(
				current, workspaces, switchRequests, new AtomicReference<>(204), failRuns);

		page.navigate(baseUrl + "/?workspaceRefreshFailure=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		page.waitForFunction("document.querySelector('#run-list .run-row.active')"
				+ "?.dataset.runId === 'run-a'");
		failRuns.set(true);

		page.selectOption("#workspace-selector", "ws-b");
		page.waitForFunction("document.querySelector('[role=alert]')?.textContent"
				+ ".includes('Workspace switched, but data refresh failed')");

		assertEquals(1, switchRequests.get());
		assertEquals("ws-b", page.inputValue("#workspace-selector"));
		assertEquals("ws-b", page.evaluate(
				"localStorage.getItem('anet.metricsviewer.workspace')"));
	}

	@Test
	void focusingWorkspaceSelectorRefreshesItsOptions() {
		final AtomicReference<String> current = new AtomicReference<>("ws-a");
		final AtomicReference<List<String>> workspaces =
				new AtomicReference<>(List.of("ws-a", "ws-b"));
		routeWorkspaceApi(current, workspaces, new AtomicInteger());

		page.navigate(baseUrl + "/?workspaceFocusRefresh=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		page.waitForFunction("document.querySelector('#workspace-selector')?.value === 'ws-a'");
		workspaces.set(List.of("ws-a", "ws-b", "ws-c"));

		page.focus("#workspace-selector");
		page.waitForFunction("Array.from(document.querySelectorAll('#workspace-selector option'))"
				+ ".some(option => option.value === 'ws-c')");
	}

	@Test
	void manualReloadRefreshesWorkspaceOptions() {
		final AtomicReference<String> current = new AtomicReference<>("ws-a");
		final AtomicReference<List<String>> workspaces =
				new AtomicReference<>(List.of("ws-a", "ws-b"));
		routeWorkspaceApi(current, workspaces, new AtomicInteger());

		page.navigate(baseUrl + "/?workspaceReloadRefresh=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		page.waitForFunction("document.querySelector('#workspace-selector')?.value === 'ws-a'");
		workspaces.set(List.of("ws-a", "ws-c"));

		page.click("#btn-reload");
		page.waitForFunction("Array.from(document.querySelectorAll('#workspace-selector option'))"
				+ ".some(option => option.value === 'ws-c')");
		assertEquals(false, page.evaluate("Array.from(document.querySelectorAll("
				+ "'#workspace-selector option')).some(option => option.value === 'ws-b')"));
	}

	@Test
	void autoReloadRefreshesWorkspaceOptions() {
		final AtomicReference<String> current = new AtomicReference<>("ws-a");
		final AtomicReference<List<String>> workspaces =
				new AtomicReference<>(List.of("ws-a", "ws-b"));
		routeWorkspaceApi(current, workspaces, new AtomicInteger());
		context.addInitScript("""
			const originalSetInterval = window.setInterval.bind(window);
			window.setInterval = (callback, delay, ...args) =>
				originalSetInterval(callback, delay === 30000 ? 25 : delay, ...args);
			""");

		page.navigate(baseUrl + "/?workspaceAutoReloadRefresh=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		page.waitForFunction("document.querySelector('#workspace-selector')?.value === 'ws-a'");
		workspaces.set(List.of("ws-a", "ws-b", "ws-c"));

		page.click("#btn-auto-reload");
		page.waitForFunction("Array.from(document.querySelectorAll('#workspace-selector option'))"
				+ ".some(option => option.value === 'ws-c')");
	}

	private void routeWorkspaceApi(
			AtomicReference<String> current,
			AtomicInteger switchRequests) {
		routeWorkspaceApi(
				current,
				new AtomicReference<>(List.of("ws-a", "ws-b")),
				switchRequests);
	}

	private void routeWorkspaceApi(
			AtomicReference<String> current,
			AtomicReference<List<String>> workspaces,
			AtomicInteger switchRequests) {
		routeWorkspaceApi(current, workspaces, switchRequests, new AtomicReference<>(204));
	}

	private void routeWorkspaceApi(
			AtomicReference<String> current,
			AtomicReference<List<String>> workspaces,
			AtomicInteger switchRequests,
			AtomicReference<Integer> switchStatus) {
		routeWorkspaceApi(
				current, workspaces, switchRequests, switchStatus, new AtomicBoolean());
	}

	private void routeWorkspaceApi(
			AtomicReference<String> current,
			AtomicReference<List<String>> workspaces,
			AtomicInteger switchRequests,
			AtomicReference<Integer> switchStatus,
			AtomicBoolean failRuns) {
		page.route("**/api/workspaces.json", route -> fulfillJson(route,
				"{\"current\":\"" + current.get()
						+ "\",\"workspaces\":"
						+ new com.google.gson.Gson().toJson(workspaces.get()) + "}"));
		page.route("**/api/workspace", route -> {
			switchRequests.incrementAndGet();
			final String requested = JsonParser.parseString(route.request().postData())
					.getAsJsonObject().get("name").getAsString();
			if (switchStatus.get() != 204) {
				route.fulfill(new com.microsoft.playwright.Route.FulfillOptions()
						.setStatus(switchStatus.get())
						.setContentType("application/json")
						.setBody("{\"code\":\"switch_failed\",\"message\":\"Switch failed\"}"));
				return;
			}
			if (!workspaces.get().contains(requested)) {
				route.fulfill(new com.microsoft.playwright.Route.FulfillOptions()
						.setStatus(404)
						.setContentType("application/json")
						.setBody("{\"code\":\"unknown_workspace\",\"message\":\"Unknown workspace\"}"));
				return;
			}
			current.set(requested);
			fulfillNoContent(route);
		});
		page.route("**/api/runs.json", route -> {
			if (failRuns.get()) {
				route.fulfill(new com.microsoft.playwright.Route.FulfillOptions().setStatus(500));
				return;
			}
			fulfillJson(route, runPayload(current.get()));
		});
		page.route("**/api/runs/prioritize", MetricsViewerPlaywrightTestSupport::fulfillNoContent);
	}

	private static String runPayload(String workspace) {
		final String runId = workspace.equals("ws-b") ? "run-b" : "run-a";
		return "{\"runs\":[{\"id\":\"" + runId
				+ "\",\"generation\":\"generation\","
				+ "\"ingest\":{\"state\":\"ready\",\"percentage\":100},\"tags\":[]}]}";
	}
}
