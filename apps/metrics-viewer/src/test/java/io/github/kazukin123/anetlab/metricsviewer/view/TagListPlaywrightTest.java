package io.github.kazukin123.anetlab.metricsviewer.view;

import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.RELOAD_NEW_TAG;
import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.RELOAD_OLD_TAG;
import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.TAG_A;
import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.TAG_B;
import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.TAG_C;
import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.manyTagRunsJson;
import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.reloadInitialMetricsJson;
import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.reloadNewTagMetricsJson;
import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.reloadTagRunsJson;
import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.splitTagMetricsJson;
import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.splitTagRunsJson;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

import com.microsoft.playwright.Browser;
import com.microsoft.playwright.Page;
import com.microsoft.playwright.options.WaitUntilState;

@SpringBootTest(
		webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT,
		properties = "metricsviewer.workspaces-dir=target/playwright-test-empty-workspaces")
class TagListPlaywrightTest extends MetricsViewerPlaywrightTestSupport {

	@Test
	void hiddenTagSelectionRestoresWhenRunContainsTagAgain() {
		page.route("**/api/runs.json", route -> fulfillJson(route, splitTagRunsJson()));
		page.route("**/api/metrics.json", route -> fulfillJson(route, splitTagMetricsJson()));

		page.navigate(baseUrl + "/?tagRestoreTest=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		waitForTag(page, TAG_B);

		selectSingleRun(page, "run_a");
		waitForTag(page, TAG_A);
		assertTrue(isTagActive(page, TAG_A));
		assertTrue(isTagActive(page, TAG_C));
		assertEquals(List.of(TAG_A, TAG_C), readTagList(page));

		waitForGraphTitle(page, TAG_A);
		assertEquals(List.of(TAG_A, TAG_C), readGraphTitles(page));

		selectSingleRun(page, "run_b");
		waitForTag(page, TAG_B);
		assertFalse(isTagVisible(page, TAG_A));
		assertFalse(isGraphTitleVisible(page, TAG_A));

		page.click("#btn-clear-all");
		assertFalse(isTagActive(page, TAG_B));

		selectSingleRun(page, "run_a");
		waitForTag(page, TAG_A);
		assertTrue(isTagActive(page, TAG_A));
		assertTrue(isTagActive(page, TAG_C));
		waitForGraphTitle(page, TAG_A);
		assertEquals(List.of(TAG_A, TAG_C), readTagList(page));
		assertEquals(List.of(TAG_A, TAG_C), readGraphTitles(page));
	}

	@Test
	void reloadActivatesNewTagReturnedByRunsMetadata() {
		final AtomicInteger runsRequestCount = new AtomicInteger();
		final AtomicInteger metricsRequestCount = new AtomicInteger();
		final List<String> metricsRequests = Collections.synchronizedList(new ArrayList<>());
		page.route("**/api/runs.json", route -> fulfillJson(
				route,
				reloadTagRunsJson(runsRequestCount.incrementAndGet() > 1)));
		page.route("**/api/metrics.json", route -> {
			final int requestIndex = metricsRequestCount.incrementAndGet();
			metricsRequests.add(route.request().postData());
			fulfillJson(route,
					requestIndex == 1 ? reloadInitialMetricsJson() : reloadNewTagMetricsJson());
		});

		page.navigate(baseUrl + "/?reloadNewTagTest=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		waitForGraphTitle(page, RELOAD_OLD_TAG);
		assertFalse(isTagVisible(page, RELOAD_NEW_TAG));

		page.click("#btn-reload");
		waitForTag(page, RELOAD_NEW_TAG);
		waitForGraphTitle(page, RELOAD_NEW_TAG);

		assertTrue(isTagActive(page, RELOAD_NEW_TAG));
		assertTrue(readTagList(page).contains(RELOAD_OLD_TAG));
		assertTrue(readTagList(page).contains(RELOAD_NEW_TAG));
		assertTrue(isGraphTitleVisible(page, RELOAD_OLD_TAG));
		assertTrue(isGraphTitleVisible(page, RELOAD_NEW_TAG));
		assertTrue(metricsRequests.size() >= 2);
		assertTrue(metricsRequests.get(1).contains(RELOAD_OLD_TAG));
		assertTrue(metricsRequests.get(1).contains(RELOAD_NEW_TAG));
	}

	@Test
	void tagListAllowsVerticalTouchScrollingOnMobile() {
		reopenPage(new Browser.NewContextOptions()
			.setViewportSize(390, 640)
			.setIsMobile(true)
			.setHasTouch(true));

		page.route("**/api/runs.json", route -> fulfillJson(route, manyTagRunsJson(60)));
		page.route("**/api/metrics.json", route -> fulfillJson(route, "{\"data\":[]}"));

		page.navigate(baseUrl + "/?mobileTagScrollTest=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		page.waitForFunction("document.querySelectorAll('#tag-list li').length === 60",
				null, new Page.WaitForFunctionOptions().setTimeout(30000));

		assertTrue(isTagListScrollable(page));
		assertTrue(canScrollTagList(page));

		final String touchAction = readTagListTouchAction(page);
		assertFalse("none".equals(touchAction));
		assertTrue(touchAction.contains("pan-y") || "auto".equals(touchAction)
				|| "manipulation".equals(touchAction));
	}
}
