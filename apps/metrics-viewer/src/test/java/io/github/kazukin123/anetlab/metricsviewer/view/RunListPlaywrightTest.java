package io.github.kazukin123.anetlab.metricsviewer.view;

import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.GENERATION;
import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.TAG_KEY;
import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.metricsJson;
import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.rawSeriesJson;
import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.runJson;
import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.runsJson;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

import com.microsoft.playwright.Page;
import com.microsoft.playwright.Route;
import com.microsoft.playwright.options.WaitUntilState;

@SpringBootTest(
		webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT,
		properties = "metricsviewer.runs-dir=target/playwright-test-empty-runs")
class RunListPlaywrightTest extends MetricsViewerPlaywrightTestSupport {

	@Test
	void runRowsToggleImmediatelyAllowEmptySelectionAndSoloOnTheSecondClick() {
		page.route("**/api/runs.json", route -> fulfillJson(route, runsJson()));
		page.route("**/api/metrics.json", route -> fulfillJson(route, metricsJson()));
		page.route("**/api/runs/prioritize", MetricsViewerPlaywrightTestSupport::fulfillNoContent);

		page.navigate(baseUrl + "/?runToggleTest=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		page.waitForFunction("document.querySelectorAll('#run-list .run-row').length === 11");
		assertEquals(List.of("run_11"), readActiveRunIds(page));

		page.click("#run-list .run-row.active");
		page.waitForFunction("document.querySelectorAll('#run-list .run-row.active').length === 0");
		assertEquals("No selection.", page.textContent("#main-area"));

		page.click("#run-list .run-row[data-run-id='run_03']");
		page.click("#run-list .run-row[data-run-id='run_04']");
		assertEquals(List.of("run_04", "run_03"), readActiveRunIds(page));

		page.evaluate("""
				runId => {
					const find = () => document.querySelector(
						`#run-list .run-row[data-run-id="${runId}"]`);
					find().click();
					find().click();
				}
				""", "run_03");
		assertEquals(List.of("run_03"), readActiveRunIds(page));
	}

	@Test
	void initialLoadErrorAllowsReloadRecoveryButNotScreenshotEscape() {
		final AtomicInteger runsRequests = new AtomicInteger();
		page.route("**/api/runs.json", route -> {
			if (runsRequests.incrementAndGet() == 1) route.abort();
			else fulfillJson(route, runsJson());
		});
		page.route("**/api/metrics.json", route -> fulfillJson(route, metricsJson()));
		page.route("**/api/runs/prioritize", MetricsViewerPlaywrightTestSupport::fulfillNoContent);

		page.navigate(baseUrl + "/?errorRecoveryTest=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		page.waitForFunction("document.body.classList.contains('error') && app?.mode === 'error'");

		assertEquals("none", page.evaluate(
				"() => getComputedStyle(document.body, '::before').pointerEvents"));
		page.click("#btn-screenshot");
		assertTrue(Boolean.TRUE.equals(page.evaluate("""
				() => document.body.classList.contains('error')
					&& !document.body.classList.contains('screenshot-mode')
					&& app?.mode === 'error'
				""")));

		page.click("#btn-reload");
		page.waitForFunction("""
				() => app?.mode === 'normal'
					&& document.querySelectorAll('.js-plotly-plot').length > 0
				""");
		assertFalse(Boolean.TRUE.equals(page.evaluate(
				"() => document.body.classList.contains('error')")));
	}

	@Test
	void updateFailuresRemainVisibleUntilEachRequestTypeSucceeds() {
		final AtomicInteger runsRequests = new AtomicInteger();
		final AtomicInteger metricsRequests = new AtomicInteger();
		page.route("**/api/runs.json", route -> {
			if (runsRequests.incrementAndGet() == 4) {
				route.fulfill(new Route.FulfillOptions()
						.setStatus(503)
						.setContentType("text/plain")
						.setBody("metadata unavailable"));
			} else {
				fulfillJson(route, runsJson());
			}
		});
		page.route("**/api/metrics.json", route -> {
			if (metricsRequests.incrementAndGet() == 2) {
				route.fulfill(new Route.FulfillOptions()
						.setStatus(502)
						.setContentType("text/plain")
						.setBody("metrics unavailable"));
			} else {
				fulfillJson(route, metricsJson());
			}
		});
		page.route("**/api/runs/prioritize", MetricsViewerPlaywrightTestSupport::fulfillNoContent);

		page.navigate(baseUrl + "/?updateFailureTest=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		waitForGraph(page);
		assertEquals(1, page.locator("#update-status").count());
		assertFalse(page.locator("#update-status").isVisible());

		page.click("#btn-reload");
		page.waitForFunction("""
				() => document.getElementById('update-status')?.title
					=== 'Metrics: Failed metrics.json: 502 metrics unavailable'
				""");
		assertEquals("Update failed", page.textContent("#update-status"));
		page.evaluate("() => app.refreshMetadata({ requestData: false })");
		assertEquals(
				"Metrics: Failed metrics.json: 502 metrics unavailable",
				page.getAttribute("#update-status", "title"));

		page.click("#btn-reload");
		page.waitForFunction("""
				() => document.getElementById('update-status')?.title
					=== 'Metadata: Failed runs.json: 503\\n'
						+ 'Metrics: Failed metrics.json: 502 metrics unavailable'
				""");

		page.click("#run-list .run-row[data-run-id='run_10']");
		page.waitForFunction("""
				() => document.getElementById('update-status')?.title
					=== 'Metadata: Failed runs.json: 503'
				""");

		page.click("#btn-reload");
		page.waitForFunction("""
				() => {
					const status = document.getElementById('update-status');
					return status?.hidden && status.textContent === '' && status.title === '';
				}
				""");
	}

	@Test
	void disappearingRunIsRemovedFromSelectionWindowsAndColorCache() {
		final AtomicInteger runsRequests = new AtomicInteger();
		page.route("**/api/runs.json", route -> fulfillJson(
				route,
				runsRequests.incrementAndGet() == 1
						? "{\"runs\":[" + runJson("run_vanish", 2, TAG_KEY) + "]}"
						: "{\"runs\":[]}"));
		page.route("**/api/metrics.json", route -> fulfillJson(
				route,
				"{\"data\":["
						+ rawSeriesJson(
								"run_vanish",
								TAG_KEY,
								new double[] {0, 1, 2},
								new float[] {1, 2, 3})
						+ "]}"));
		page.route("**/api/runs/prioritize", MetricsViewerPlaywrightTestSupport::fulfillNoContent);

		page.navigate(baseUrl + "/?runDisappearanceTest=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		waitForGraph(page);
		page.click("#btn-reload");
		page.waitForFunction("""
				() => document.querySelectorAll('#run-list .run-row').length === 0
					&& document.getElementById('main-area')?.textContent === 'No selection.'
				""");
		assertTrue(Boolean.TRUE.equals(page.evaluate("""
				() => app.cache.getRun('run_vanish') === null
					&& app.cache.getWindow('run_vanish', 'palette/test') === null
					&& !app.runColorMap.has('run_vanish')
				""")));
	}

	@Test
	void convertingRunPollOnlyUpdatesExistingPercentageUntilReload() {
		final String nextGeneration = "00000000-0000-0000-0000-000000000002";
		final String newTag = "poll/new-tag";
		final AtomicInteger runsRequests = new AtomicInteger();
		final AtomicInteger metricsRequests = new AtomicInteger();
		final AtomicBoolean reloaded = new AtomicBoolean();
		page.route("**/api/runs.json", route -> {
			final int index = runsRequests.incrementAndGet();
			final String pollRun = index == 1
					? runJson("run_poll", 2, GENERATION, "converting", 10, TAG_KEY)
					: runJson(
							"run_poll",
							8,
							nextGeneration,
							index == 2 ? "converting" : "ready",
							index == 2 ? 50 : 100,
							TAG_KEY,
							newTag);
			fulfillJson(
					route,
					"{\"runs\":["
							+ pollRun
							+ (index == 1
									? ""
									: "," + runJson(
											"run_new",
											8,
											nextGeneration,
											"ready",
											100,
											TAG_KEY))
							+ "]}");
		});
		page.route("**/api/metrics.json", route -> {
			metricsRequests.incrementAndGet();
			final String generation = reloaded.get() ? nextGeneration : GENERATION;
			final String extraSeries = reloaded.get()
					? "," + rawSeriesJson(
							"run_poll",
							newTag,
							generation,
							new double[] {0, 1, 2},
							new float[] {4, 5, 6})
					: "";
			fulfillJson(route, "{\"data\":["
					+ rawSeriesJson(
							"run_poll",
							TAG_KEY,
							generation,
							new double[] {0, 1, 2},
							new float[] {1, 2, 3})
					+ extraSeries
					+ "]}");
		});
		page.route("**/api/runs/prioritize", MetricsViewerPlaywrightTestSupport::fulfillNoContent);

		page.navigate(baseUrl + "/?ingestPollTest=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		waitForGraph(page);
		page.evaluate("""
				() => {
					const plot = document.querySelector('.js-plotly-plot');
					window.__pollPlot = plot;
					window.__pollRunRow = document.querySelector(
						"#run-list .run-row[data-run-id='run_poll']");
					window.__pollRunName = document.querySelector(
						"#run-list .run-row[data-run-id='run_poll'] .run-name").textContent;
					window.__pollTagList = document.getElementById('tag-list').textContent;
					window.__pollHeader = document.querySelector('.graph-header').textContent;
					window.__pollGeneration = app.cache.getRun('run_poll').generation;
					window.__pollMainScrollTop = document.getElementById('main-area').scrollTop;
					window.__pollXRange = JSON.stringify(plot.layout.xaxis.range ?? null);
					window.__pollYRange = JSON.stringify(plot.layout.yaxis.range ?? null);
				}
				""");
		final int initialMetricsRequests = metricsRequests.get();
		page.waitForFunction(
				"document.querySelector(\"#run-list .run-row[data-run-id='run_poll'] .run-progress\")"
						+ "?.textContent === '50%'",
				null,
				new Page.WaitForFunctionOptions().setTimeout(7000));

		assertEquals(initialMetricsRequests, metricsRequests.get());
		assertEquals(1, page.locator("#run-list .run-row").count());
		assertEquals(1, page.locator("#tag-list li").count());
		assertTrue(Boolean.TRUE.equals(page.evaluate("""
				() => window.__pollPlot === document.querySelector('.js-plotly-plot')
					&& window.__pollRunRow === document.querySelector(
						"#run-list .run-row[data-run-id='run_poll']")
					&& window.__pollRunName === document.querySelector(
						"#run-list .run-row[data-run-id='run_poll'] .run-name").textContent
					&& window.__pollTagList === document.getElementById('tag-list').textContent
					&& window.__pollHeader === document.querySelector('.graph-header').textContent
					&& window.__pollGeneration === app.cache.getRun('run_poll').generation
					&& app.cache.getRunIds().join(',') === 'run_poll'
					&& app.cache.getTag('run_poll', 'palette/test').stats.count === 3
					&& window.__pollMainScrollTop === document.getElementById('main-area').scrollTop
					&& window.__pollXRange === JSON.stringify(
						document.querySelector('.js-plotly-plot').layout.xaxis.range ?? null)
					&& window.__pollYRange === JSON.stringify(
						document.querySelector('.js-plotly-plot').layout.yaxis.range ?? null)
				""")));

		assertEquals(
				"rgb(36, 87, 125)",
				page.evaluate("""
						() => getComputedStyle(
							document.querySelector(
								"#run-list .run-row[data-run-id='run_poll']"),
							'::before').backgroundColor
						"""));
		assertEquals(
				"rgba(190, 190, 190, 0.24)",
				page.evaluate("""
						() => {
							const row = document.querySelector(
								"#run-list .run-row[data-run-id='run_poll']");
							row.classList.remove('active');
							const color = getComputedStyle(row, '::before').backgroundColor;
							row.classList.add('active');
							return color;
						}
						"""));
		assertEquals("rgb(36, 87, 125)", page.evaluate("""
				() => getComputedStyle(document.querySelector('.run-row.active')).backgroundColor
				"""));
		assertEquals("rgb(36, 87, 125)", page.evaluate("""
				() => getComputedStyle(document.querySelector('#tag-list li.active')).backgroundColor
				"""));
		assertEquals("rgb(196, 196, 196)", page.evaluate("""
				() => getComputedStyle(document.querySelector('.run-progress')).color
				"""));
		assertEquals(
				"rgb(36, 87, 125)|rgb(36, 87, 125)|rgb(36, 87, 125)"
						+ "|rgb(36, 87, 125)|rgb(73, 201, 255)|rgb(243, 250, 255)",
				page.evaluate("""
						() => {
							const autoReload = document.getElementById('btn-auto-reload');
							const scrollLock = document.getElementById('btn-graph-scroll-lock');
							const log = document.querySelector('.graph-log-toggle');
							const checkbox = document.getElementById('chk-lock-tags');
							autoReload.classList.add('active');
							scrollLock.classList.add('active');
							log.classList.add('active');
							checkbox.checked = true;
							return [
								getComputedStyle(autoReload).backgroundColor,
								getComputedStyle(scrollLock).backgroundColor,
								getComputedStyle(log).backgroundColor,
								getComputedStyle(checkbox).backgroundColor,
								getComputedStyle(checkbox).borderColor,
								getComputedStyle(autoReload).color
							].join('|');
						}
						"""));

		page.waitForFunction(
				"document.querySelector(\"#run-list .run-row[data-run-id='run_poll'] .run-progress\")"
						+ " === null && app.ingestPollTimer === null",
				null,
				new Page.WaitForFunctionOptions().setTimeout(7000));
		assertTrue(runsRequests.get() >= 3);
		assertEquals(initialMetricsRequests, metricsRequests.get());
		assertEquals(0, page.locator(
				"#run-list .run-row[data-run-id='run_poll'] .run-progress").count());
		assertEquals("0%", page.evaluate("""
				() => document.querySelector(
					"#run-list .run-row[data-run-id='run_poll']")
					.style.getPropertyValue('--ingest-progress')
				"""));

		reloaded.set(true);
		page.click("#btn-reload");
		page.waitForFunction(
				"document.querySelectorAll('#run-list .run-row').length === 2"
						+ " && document.querySelectorAll('#tag-list li').length === 2"
						+ " && app.cache.getRun('run_poll').generation"
						+ " === '00000000-0000-0000-0000-000000000002'"
						+ " && app.cache.getTag('run_poll', 'palette/test').stats.count === 9");
		assertTrue(metricsRequests.get() > initialMetricsRequests);
	}

	@Test
	void autoReloadButtonReflectsToggleState() {
		page.route("**/api/runs.json", route -> fulfillJson(route, runsJson()));
		page.route("**/api/metrics.json", route -> fulfillJson(route, metricsJson()));

		page.navigate(baseUrl + "/?autoReloadButtonTest=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		waitForGraph(page);

		assertEquals("Auto Reload: OFF", page.textContent("#btn-auto-reload"));
		assertEquals("false", page.getAttribute("#btn-auto-reload", "aria-pressed"));
		assertFalse(isAutoReloadButtonActive(page));

		page.click("#btn-auto-reload");
		assertEquals("Auto Reload: ON", page.textContent("#btn-auto-reload"));
		assertEquals("true", page.getAttribute("#btn-auto-reload", "aria-pressed"));
		assertTrue(isAutoReloadButtonActive(page));

		page.click("#btn-auto-reload");
		assertEquals("Auto Reload: OFF", page.textContent("#btn-auto-reload"));
		assertEquals("false", page.getAttribute("#btn-auto-reload", "aria-pressed"));
		assertFalse(isAutoReloadButtonActive(page));
	}
}
