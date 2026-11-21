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

	/**
	 * 重いReloadの応答待ちにスクロールなどの取得が割り込むと、Reloadのmetrics.jsonは打ち切られる。
	 * 失敗のToastを出さず、古くなったgraphの取り直しは割り込んだ側の取得が引き継ぐ。
	 */
	@Test
	void reloadInterruptedByALaterFetchShowsNoFailureAndTheLaterFetchRefetchesTheGrownGraph() {
		pointCounts.put(seriesKey("run_a", TAG_A), 120);
		openWithAllRunsSelected("reloadInterruptedTest");
		pointCounts.put(seriesKey("run_b", TAG_A), 110);

		// スクロールで見える範囲が変わったときと同じ取得を割り込ませる。
		interruptReloadWith("app.requestVisibleData()");

		// テスト用のserverはworkspaceを持たず、Reloadのたびに別件のToastが出るので、Reloadの失敗だけを見る。
		assertFalse(Boolean.TRUE.equals(page.evaluate("""
				() => Array.from(document.querySelectorAll('.toast'))
					.some(toast => toast.textContent.includes('Reload failed'))
				""")));
		assertEquals(List.of(List.of("run_a|" + TAG_A, "run_b|" + TAG_A)), metricsRequests);
		page.waitForFunction("""
				() => document.getElementById(graphId('tag/a')).data
					.find(trace => trace.meta?.runId === 'run_b')?.x.length === 110
				""");
		assertTrue(Boolean.TRUE.equals(page.evaluate("() => app.pendingRefreshScope === null")));
	}

	/**
	 * 手動Reloadを、最新へ追従中のgraphだけを見るAuto Reloadの取得が打ち切っても、
	 * 過去を拡大していて追従していないgraphの取り直しは落ちない。
	 */
	@Test
	void reloadInterruptedByAnAutoReloadFetchStillRefetchesAGraphThatIsNotFollowing() {
		openWithAllRunsSelected("reloadInterruptedByAutoReloadTest");
		// tag/bを過去へ拡大して追従を外し、run_bのwindowを取り込み中(pending)にしておく。
		page.evaluate("""
				async () => {
					await Plotly.relayout(document.getElementById(graphId('tag/b')), {'xaxis.range': [10, 20]});
					await new Promise(resolve => setTimeout(resolve, 300));
				}
				""");
		pendingOnce.add(seriesKey("run_b", TAG_B));
		page.evaluate("async () => { await app.requestVisibleData({ force: true }); }");
		assertEquals(1, traceCount(TAG_B));
		metricsRequests.clear();

		interruptReloadWith("app.requestVisibleData({ refresh: true, followOnly: true })");

		assertEquals(List.of(List.of("run_a|" + TAG_B, "run_b|" + TAG_B)), metricsRequests);
		page.waitForFunction("() => document.getElementById(graphId('tag/b')).data.length === 2");
		assertTrue(Boolean.TRUE.equals(page.evaluate("() => app.pendingRefreshScope === null")));
	}

	/**
	 * Auto Reloadの取得をスクロールなどの取得が打ち切ると、割り込んだ側はAuto Reloadと同じ範囲で取り直す。
	 * 最新へ追従中のgraphは取り直し、追従していないgraphの古いwindowは取り直さない。
	 */
	@Test
	void autoReloadInterruptedByALaterFetchRefetchesOnlyTheFollowingGraph() {
		pointCounts.put(seriesKey("run_a", TAG_A), 120);
		openWithAllRunsSelected("autoReloadInterruptedTest");
		// tag/bを過去へ拡大して追従を外し、run_bのwindowを取り込み中(pending)にしておく。
		page.evaluate("""
				async () => {
					await Plotly.relayout(document.getElementById(graphId('tag/b')), {'xaxis.range': [10, 20]});
					await new Promise(resolve => setTimeout(resolve, 300));
				}
				""");
		pendingOnce.add(seriesKey("run_b", TAG_B));
		page.evaluate("async () => { await app.requestVisibleData({ force: true }); }");
		assertEquals(1, traceCount(TAG_B));
		metricsRequests.clear();
		// 追従中のtag/aを伸ばす。run_aの方が長いのでviewportは変わらない。
		pointCounts.put(seriesKey("run_b", TAG_A), 110);

		// Auto Reloadのtickと同じ呼び出しを止め、スクロールと同じ取得を割り込ませる。
		interruptWith("""
				(async () => {
					await app.refreshMetadata({ requestData: false });
					await app.requestVisibleData({ refresh: true, followOnly: true });
				})()
				""", "app.requestVisibleData()");

		assertEquals(List.of(List.of("run_a|" + TAG_A, "run_b|" + TAG_A)), metricsRequests);
		page.waitForFunction("""
				() => document.getElementById(graphId('tag/a')).data
					.find(trace => trace.meta?.runId === 'run_b')?.x.length === 110
				""");
		assertEquals(1, traceCount(TAG_B));
		assertTrue(Boolean.TRUE.equals(page.evaluate("() => app.pendingRefreshScope === null")));
	}

	/** Reloadのmetrics.jsonを応答前で止め、そこへ別の取得を割り込ませて、打ち切りの信号だけで終わらせる。 */
	private void interruptReloadWith(String interruptingCall) {
		interruptWith("app.onReload()", interruptingCall);
	}

	/** heldCallが出す最初のmetrics.jsonを応答前で止め、interruptingCallを割り込ませて、打ち切りの信号だけで終わらせる。 */
	private void interruptWith(String heldCall, String interruptingCall) {
		page.evaluate("""
				async ([heldCall, interruptingCall]) => {
					const originalFetch = window.fetch;
					let held = false;
					window.fetch = (input, init) => {
						const url = typeof input === 'string' ? input : input.url;
						if (!held && url.endsWith('/api/metrics.json')) {
							held = true;
							window.fetch = originalFetch;
							return new Promise((resolve, reject) => init.signal.addEventListener('abort',
								() => reject(new DOMException('The operation was aborted.', 'AbortError'))));
						}
						return originalFetch(input, init);
					};
					const heldWork = new Function('return ' + heldCall)();
					while (!held) await new Promise(resolve => setTimeout(resolve, 10));
					await new Function('return ' + interruptingCall)();
					// 止めた側は打ち切りで終わる。onReloadは打ち切りを失敗にしないが、ほかの呼び出しは例外のまま返る。
					await heldWork.catch(error => {
						if (error?.name !== 'AbortError') throw error;
					});
				}
				""", List.of(heldCall, interruptingCall));
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
					+ "\"stepBucketWidth\":null,\"projection\":null}";
		}
		final long first = Math.max(0L, fromStep);
		final long last = Math.min(pointCounts.get(seriesKey(runId, tagKey)) - 1L, toStep);
		final int size = (int) Math.max(0L, last - first + 1L);
		if (size == 0) {
			return head + "\"availability\":\"empty\",\"pointBudget\":0,"
					+ "\"stepBucketWidth\":null,\"projection\":null}";
		}
		final double[] steps = new double[size];
		final float[] values = new float[size];
		for (int i = 0; i < size; i++) {
			steps[i] = first + i;
			values[i] = (float) Math.sin((first + i) / 7.0) + ("run_a".equals(runId) ? 0f : 2f);
		}
		return head + "\"availability\":\"ok\",\"pointBudget\":" + size + ","
				+ "\"stepBucketWidth\":1,\"issues\":[],"
				+ "\"projection\":{\"kind\":\"raw\","
				+ "\"steps\":\"" + encodeFloat64(steps) + "\","
				+ "\"values\":\"" + encodeFloat32(values) + "\"}}";
	}

	private static String seriesKey(String runId, String tagKey) {
		return runId + "|" + tagKey;
	}
}
