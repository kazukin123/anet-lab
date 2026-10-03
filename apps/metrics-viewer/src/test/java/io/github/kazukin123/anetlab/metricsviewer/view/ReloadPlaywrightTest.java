package io.github.kazukin123.anetlab.metricsviewer.view;

import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.GENERATION;
import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.TAG_A;
import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.TAG_B;
import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.encodeFloat32;
import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.encodeFloat64;
import static io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerPlaywrightTestData.tagJson;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.microsoft.playwright.Page;
import com.microsoft.playwright.Route;
import com.microsoft.playwright.options.WaitUntilState;

/**
 * Reloadが、取得後にtagが変わったgraphだけを取り直して作り直すことを固定する。
 * runs.jsonとmetrics.jsonは同じ点数モデルから作り、metrics.jsonは実serverと同じく要求のstep範囲を返す。
 */
@SpringBootTest(
		webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT,
		properties = "metricsviewer.workspaces-dir=target/playwright-test-empty-workspaces")
class ReloadPlaywrightTest extends MetricsViewerPlaywrightTestSupport {

	private static final List<String> RUN_IDS = List.of("run_a", "run_b");
	private static final List<String> TAG_KEYS = List.of(TAG_A, TAG_B);
	private static final ObjectMapper JSON = new ObjectMapper();

	/** (runId, tagKey)ごとの点数。stepは0から1刻みで並ぶ。 */
	private final Map<String, Integer> pointCounts = new ConcurrentHashMap<>();
	/** 次の1回だけavailability=pendingで返す系列。 */
	private final Set<String> pendingOnce = ConcurrentHashMap.newKeySet();
	/** metrics.jsonごとの系列("runId|tagKey")。 */
	private final List<List<String>> metricsRequests = new CopyOnWriteArrayList<>();

	@BeforeEach
	void routeTheSharedPointModel() {
		pointCounts.clear();
		pendingOnce.clear();
		metricsRequests.clear();
		for (String runId : RUN_IDS) {
			for (String tagKey : TAG_KEYS) pointCounts.put(seriesKey(runId, tagKey), 100);
		}
		page.route("**/api/runs.json", route -> fulfillJson(route, runsJson()));
		page.route("**/api/metrics.json", this::fulfillMetrics);
		page.route("**/api/runs/prioritize", MetricsViewerPlaywrightTestSupport::fulfillNoContent);
	}

	@Test
	void reloadWithoutMetadataChangesSendsNoMetricsRequestAndKeepsEveryGraph() {
		openWithAllRunsSelected("reloadNoChangeTest");
		rememberGraphs();

		reload();

		assertEquals(List.of(), metricsRequests);
		assertTrue(isSameGraph(TAG_A));
		assertTrue(isSameGraph(TAG_B));
	}

	@Test
	void reloadRefetchesOnlyTheGraphWhoseTagGrewTogetherWithItsOtherRuns() {
		// run_aの方が長いので、run_bが伸びてもautorangeのviewportとwindowは変わらない。
		// 取り直しの契機は、window取得後にtagの点数が変わったことだけになる。
		pointCounts.put(seriesKey("run_a", TAG_A), 120);
		openWithAllRunsSelected("reloadAppendTest");
		rememberGraphs();

		pointCounts.put(seriesKey("run_b", TAG_A), 110);
		reload();

		// 同じgraphに載るrun_aも一緒に取り直し、graph内の点予算をそろえる。
		assertEquals(List.of(List.of("run_a|" + TAG_A, "run_b|" + TAG_A)), metricsRequests);
		assertFalse(isSameGraph(TAG_A));
		assertTrue(isSameGraph(TAG_B));
		assertEquals(110, traceLength(TAG_A, "run_b"));
	}

	@Test
	void reloadKeepsAGraphZoomedIntoThePastWhenAppendedPointsFallOutsideItsWindow() {
		openWithAllRunsSelected("reloadZoomedPastTest");
		// 生点のwindowは拡大しても取り直さないので、明示viewportへ合わせて取り直させる。
		page.evaluate("""
				async () => {
					await Plotly.relayout(document.getElementById(graphId('tag/a')), {'xaxis.range': [10, 20]});
					await new Promise(resolve => setTimeout(resolve, 300));
					await app.requestVisibleData({ force: true });
				}
				""");
		assertEquals(30L, ((Number) page.evaluate(
				"() => app.cache.getWindow('run_a', 'tag/a').toStep")).longValue());
		rememberGraphs();
		metricsRequests.clear();

		// 追記点はstep 100以降で、windowの右端30へ届かない。
		pointCounts.put(seriesKey("run_a", TAG_A), 110);
		reload();

		assertEquals(List.of(), metricsRequests);
		assertTrue(isSameGraph(TAG_A));
		assertTrue(isSameGraph(TAG_B));
	}

	@Test
	void reloadRetriesAPendingWindowEvenWithoutMetadataChanges() {
		openWithAllRunsSelected("reloadPendingTest");
		pendingOnce.add(seriesKey("run_b", TAG_B));
		page.evaluate("async () => { await app.requestVisibleData({ force: true }); }");
		assertEquals(1, traceCount(TAG_B));
		rememberGraphs();
		metricsRequests.clear();

		reload();

		assertEquals(List.of(List.of("run_a|" + TAG_B, "run_b|" + TAG_B)), metricsRequests);
		assertEquals(2, traceCount(TAG_B));
		assertTrue(isSameGraph(TAG_A));
	}

	@Test
	void togglingLogRebuildsOnlyItsOwnGraph() {
		openWithAllRunsSelected("logToggleReuseTest");
		rememberGraphs();

		page.evaluate("""
				() => document.getElementById(graphId('tag/a'))
					.closest('.graph-block').querySelector('.graph-log-toggle').click()
				""");

		assertFalse(isSameGraph(TAG_A));
		assertTrue(isSameGraph(TAG_B));
		assertEquals(List.of(), metricsRequests);
	}

	private void openWithAllRunsSelected(String testName) {
		page.navigate(baseUrl + "/?" + testName + "=" + System.nanoTime(),
				new Page.NavigateOptions().setWaitUntil(WaitUntilState.DOMCONTENTLOADED));
		page.waitForFunction("app?.mode === 'normal' && app.selectedRuns.length === 1");
		page.evaluate("() => app.setSelectedRuns(['run_a', 'run_b'])");
		page.waitForFunction("""
				() => ['tag/a', 'tag/b'].every(tagKey =>
					document.getElementById(graphId(tagKey))?.data?.length === 2)
				""");
		metricsRequests.clear();
	}

	private void reload() {
		page.evaluate("async () => { await app.onReload(); }");
	}

	private void rememberGraphs() {
		page.evaluate("""
				() => {
					window.__graphs = Object.fromEntries(['tag/a', 'tag/b']
						.map(tagKey => [tagKey, document.getElementById(graphId(tagKey))]));
				}
				""");
	}

	private boolean isSameGraph(String tagKey) {
		return Boolean.TRUE.equals(page.evaluate(
				"tagKey => window.__graphs[tagKey] === document.getElementById(graphId(tagKey))",
				tagKey));
	}

	private int traceCount(String tagKey) {
		return ((Number) page.evaluate(
				"tagKey => document.getElementById(graphId(tagKey)).data.length",
				tagKey)).intValue();
	}

	private int traceLength(String tagKey, String runId) {
		return ((Number) page.evaluate("""
				([tagKey, runId]) => document.getElementById(graphId(tagKey)).data
					.find(trace => trace.meta?.runId === runId).x.length
				""", List.of(tagKey, runId))).intValue();
	}

	private String runsJson() {
		final List<String> runs = new ArrayList<>();
		for (String runId : RUN_IDS) {
			final List<String> tags = new ArrayList<>();
			long maxStep = 0L;
			for (String tagKey : TAG_KEYS) {
				final int count = pointCounts.get(seriesKey(runId, tagKey));
				tags.add(tagJson(tagKey, count - 1L));
				maxStep = Math.max(maxStep, count - 1L);
			}
			runs.add("{\"id\":\"" + runId + "\","
					+ "\"generation\":\"" + GENERATION + "\","
					+ "\"stats\":{\"maxStep\":" + maxStep + "},"
					+ "\"ingest\":{\"state\":\"ready\",\"percentage\":100},"
					+ "\"tags\":[" + String.join(",", tags) + "]}");
		}
		return "{\"runs\":[" + String.join(",", runs) + "]}";
	}

	private void fulfillMetrics(Route route) {
		try {
			final List<String> requested = new ArrayList<>();
			final List<String> results = new ArrayList<>();
			for (JsonNode series : JSON.readTree(route.request().postData()).path("series")) {
				final String runId = series.path("runId").asText();
				final String tagKey = series.path("tagKey").asText();
				final long fromStep = series.path("fromStep").asLong();
				final long toStep = series.path("toStep").asLong();
				requested.add(runId + "|" + tagKey);
				results.add(seriesJson(runId, tagKey, fromStep, toStep));
			}
			metricsRequests.add(List.copyOf(requested));
			fulfillJson(route, "{\"data\":[" + String.join(",", results) + "]}");
		} catch (Exception e) {
			throw new IllegalStateException(e);
		}
	}

	private String seriesJson(String runId, String tagKey, long fromStep, long toStep) {
		final String head = "{\"runId\":\"" + runId + "\",\"tagKey\":\"" + tagKey + "\","
				+ "\"generation\":\"" + GENERATION + "\","
				+ "\"fromStep\":" + fromStep + ",\"toStep\":" + toStep + ",";
		if (pendingOnce.remove(seriesKey(runId, tagKey))) {
			return head + "\"availability\":\"pending\",\"pointBudget\":0,"
					+ "\"level\":null,\"bucketWidth\":null,\"projection\":null}";
		}
		final long first = Math.max(0L, fromStep);
		final long last = Math.min(pointCounts.get(seriesKey(runId, tagKey)) - 1L, toStep);
		final int size = (int) Math.max(0L, last - first + 1L);
		if (size == 0) {
			return head + "\"availability\":\"empty\",\"pointBudget\":0,"
					+ "\"level\":null,\"bucketWidth\":null,\"projection\":null}";
		}
		final double[] steps = new double[size];
		final float[] values = new float[size];
		for (int i = 0; i < size; i++) {
			steps[i] = first + i;
			values[i] = (float) Math.sin((first + i) / 7.0) + ("run_a".equals(runId) ? 0f : 2f);
		}
		return head + "\"availability\":\"ok\",\"pointBudget\":" + size + ","
				+ "\"level\":0,\"bucketWidth\":1,\"issues\":[],"
				+ "\"projection\":{\"kind\":\"raw\","
				+ "\"steps\":\"" + encodeFloat64(steps) + "\","
				+ "\"values\":\"" + encodeFloat32(values) + "\"}}";
	}

	private static String seriesKey(String runId, String tagKey) {
		return runId + "|" + tagKey;
	}
}
