package io.github.kazukin123.anetlab.metricsviewer.view;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;

final class MetricsViewerPlaywrightTestData {

	static final String TAG_KEY = "palette/test";
	static final String TAG_A = "tag/a";
	static final String TAG_B = "tag/b";
	static final String TAG_C = "tag/c";
	static final String RELOAD_RUN = "run_reload";
	static final String RELOAD_OLD_TAG = "reload/old";
	static final String RELOAD_NEW_TAG = "reload/new";
	static final String SIGNED_LOG_TAG = "signed/log";
	static final String GENERATION = "00000000-0000-0000-0000-000000000001";
	static final List<String> RUN_IDS = List.of(
			"run_01", "run_02", "run_03", "run_04", "run_05", "run_06",
			"run_07", "run_08", "run_09", "run_10", "run_11");

	private MetricsViewerPlaywrightTestData() {
	}

	static String runsJson() {
		final StringBuilder sb = new StringBuilder();
		sb.append("{\"runs\":[");
		for (int i = 0; i < RUN_IDS.size(); i++) {
			if (i > 0) sb.append(',');
			sb.append(runJson(RUN_IDS.get(i), 2, TAG_KEY));
		}
		sb.append("]}");
		return sb.toString();
	}

	static String manyTagRunsJson(int tagCount) {
		final List<String> tags = new ArrayList<>();
		for (int i = 0; i < tagCount; i++) tags.add("mobile/tag_" + String.format("%02d", i));
		return "{\"runs\":[" + runJson("run_mobile", 2, tags.toArray(String[]::new)) + "]}";
	}

	static String manyGraphRunsJson(int graphCount) {
		final List<String> tags = new ArrayList<>();
		for (int i = 0; i < graphCount; i++) tags.add("mobile/graph_" + String.format("%02d", i));
		return "{\"runs\":[" + runJson("run_mobile_graph", 2, tags.toArray(String[]::new)) + "]}";
	}

	static String metricsJson() {
		final StringBuilder sb = new StringBuilder();
		sb.append("{\"data\":[");
		for (int i = 0; i < RUN_IDS.size(); i++) {
			if (i > 0) sb.append(',');
			final float baseValue = i + 1;
			sb.append(rawSeriesJson(
					RUN_IDS.get(i),
					TAG_KEY,
					new double[] {0, 1, 2},
					new float[] {baseValue, baseValue + 1, baseValue + 2}));
		}
		sb.append("]}");
		return sb.toString();
	}

	static String manyGraphMetricsJson(int graphCount) {
		final StringBuilder sb = new StringBuilder();
		sb.append("{\"data\":[");
		for (int i = 0; i < graphCount; i++) {
			if (i > 0) sb.append(',');
			sb.append(rawSeriesJson(
					"run_mobile_graph",
					"mobile/graph_" + String.format("%02d", i),
					new double[] {0, 1, 2},
					new float[] {i + 1, i + 2, i + 3}));
		}
		sb.append("]}");
		return sb.toString();
	}

	static String signedLogRunsJson() {
		return "{\"runs\":[" + runJson("run_signed", 4, SIGNED_LOG_TAG) + "]}";
	}

	static String signedLogMetricsJson() {
		return "{\"data\":[" + rawSeriesJson(
				"run_signed",
				SIGNED_LOG_TAG,
				new double[] {0, 1, 2, 3, 4},
				new float[] {-100, -9, 0, 9, 100}) + "]}";
	}

	static String signedLogZoomMetricsJson() {
		return "{\"data\":[" + rawSeriesJson(
				"run_signed",
				SIGNED_LOG_TAG,
				new double[] {0, 1, 2, 3, 4},
				new float[] {-100, 20, 25, 30, 100}) + "]}";
	}

	static String splitTagRunsJson() {
		return "{\"runs\":["
				+ runJson("run_a", 2, TAG_C, TAG_A)
				+ ","
				+ runJson("run_b", 2, TAG_B)
				+ "]}";
	}

	static String legendStateRunsJson() {
		return "{\"runs\":["
				+ runJson("run_a", 31, TAG_A, TAG_B)
				+ ","
				+ runJson("run_b", 31, TAG_A, TAG_B)
				+ "]}";
	}

	static String legendStateMetricsJson() {
		return "{\"data\":["
				+ rawSeriesJson("run_a", TAG_A, new double[] {0, 1, 2}, new float[] {1, 2, 3})
				+ ","
				+ rawSeriesJson("run_b", TAG_A, new double[] {0, 1, 2}, new float[] {3, 2, 1})
				+ ","
				+ rawSeriesJson("run_a", TAG_B, new double[] {0, 1, 2}, new float[] {4, 5, 6})
				+ ","
				+ rawSeriesJson("run_b", TAG_B, new double[] {0, 1, 2}, new float[] {6, 5, 4})
				+ "]}";
	}

	static String legendStateLodMetricsJson() {
		return "{\"data\":["
				+ lodSeriesJson("run_a", TAG_A, 0)
				+ ","
				+ lodSeriesJson("run_b", TAG_A, 10)
				+ ","
				+ lodSeriesJson("run_a", TAG_B, 20)
				+ ","
				+ lodSeriesJson("run_b", TAG_B, 30)
				+ "]}";
	}

	static String splitTagMetricsJson() {
		return "{\"data\":["
				+ rawSeriesJson("run_a", TAG_A, new double[] {0, 1, 2}, new float[] {1, 2, 3})
				+ ","
				+ rawSeriesJson("run_a", TAG_C, new double[] {0, 1, 2}, new float[] {7, 8, 9})
				+ ","
				+ rawSeriesJson("run_b", TAG_B, new double[] {0, 1, 2}, new float[] {4, 5, 6})
				+ "]}";
	}

	static String reloadTagRunsJson(boolean includeNewTag) {
		return "{\"runs\":["
				+ runJson(
						RELOAD_RUN,
						3,
						includeNewTag
								? new String[] {RELOAD_OLD_TAG, RELOAD_NEW_TAG}
								: new String[] {RELOAD_OLD_TAG})
				+ "]}";
	}

	static String reloadInitialMetricsJson() {
		return "{\"data\":["
				+ rawSeriesJson(RELOAD_RUN, RELOAD_OLD_TAG, new double[] {0, 1, 2}, new float[] {1, 2, 3})
				+ "]}";
	}

	static String reloadNewTagMetricsJson() {
		return "{\"data\":["
				+ rawSeriesJson(
						RELOAD_RUN,
						RELOAD_OLD_TAG,
						new double[] {0, 1, 2, 3},
						new float[] {1, 2, 3, 4})
				+ ","
				+ rawSeriesJson(
						RELOAD_RUN,
						RELOAD_NEW_TAG,
						new double[] {0, 1},
						new float[] {10, 11})
				+ "]}";
	}

	static String runJson(String runId, long maxStep, String... tagKeys) {
		return runJsonWithState(runId, maxStep, "ready", 100, tagKeys);
	}

	static String runJsonWithGeneration(
			String runId,
			long maxStep,
			String generation,
			String... tagKeys) {
		return runJson(runId, maxStep, generation, "ready", 100, tagKeys);
	}

	static String runJsonWithState(
			String runId,
			long maxStep,
			String state,
			int percentage,
			String... tagKeys) {
		return runJson(runId, maxStep, GENERATION, state, percentage, tagKeys);
	}

	static String runJson(
			String runId,
			long maxStep,
			String generation,
			String state,
			int percentage,
			String... tagKeys) {
		final StringBuilder tags = new StringBuilder();
		for (int i = 0; i < tagKeys.length; i++) {
			if (i > 0) tags.append(',');
			tags.append(tagJson(tagKeys[i], maxStep));
		}
		return "{\"id\":\"" + runId + "\","
				+ "\"generation\":\"" + generation + "\","
				+ "\"stats\":{\"maxStep\":" + maxStep + "},"
				+ "\"ingest\":{\"state\":\"" + state + "\",\"percentage\":" + percentage + "},"
				+ "\"tags\":[" + tags + "]}";
	}

	static String tagJson(String tagKey, long maxStep) {
		final long count = maxStep + 1;
		return "{\"key\":\"" + tagKey + "\","
				+ "\"type\":\"scalar\",\"status\":\"ok\","
				+ "\"stats\":{"
				+ "\"minStep\":0,\"maxStep\":" + maxStep + ","
				+ "\"count\":" + count + ",\"lastValue\":1,"
				+ "\"minValue\":0,\"maxValue\":1,\"mean\":0.5,"
				+ "\"variance\":0.25,\"stdDev\":0.5}}";
	}

	static String rawSeriesJson(
			String runId,
			String tagKey,
			double[] steps,
			float[] values) {
		return rawSeriesJson(runId, tagKey, GENERATION, steps, values);
	}

	static String rawSeriesJson(
			String runId,
			String tagKey,
			String generation,
			double[] steps,
			float[] values) {
		final double fromStep = steps.length == 0 ? 0 : steps[0];
		final double toStep = steps.length == 0 ? 0 : steps[steps.length - 1];
		return "{\"runId\":\"" + runId + "\","
				+ "\"tagKey\":\"" + tagKey + "\","
				+ "\"generation\":\"" + generation + "\","
				+ "\"fromStep\":" + fromStep + ",\"toStep\":" + toStep + ","
				+ "\"availability\":\"ok\",\"pointBudget\":" + steps.length + ","
				+ "\"level\":0,\"bucketWidth\":1,\"issues\":[],"
				+ "\"projection\":{\"kind\":\"raw\","
				+ "\"steps\":\"" + encodeFloat64(steps) + "\","
				+ "\"values\":\"" + encodeFloat32(values) + "\"}}";
	}

	static String lodMetricsJson() {
		return "{\"data\":[" + lodSeriesJson("run_lod_ui", TAG_KEY, 0) + "]}";
	}

	static String lodSeriesJson(String runId, String tagKey, float offset) {
		return "{"
				+ "\"runId\":\"" + runId + "\",\"tagKey\":\"" + tagKey + "\","
				+ "\"generation\":\"" + GENERATION + "\","
				+ "\"fromStep\":-31,\"toStep\":62,\"availability\":\"ok\","
				+ "\"pointBudget\":6,\"level\":1,\"bucketWidth\":16,\"issues\":[],"
				+ "\"projection\":{\"kind\":\"lod\","
				+ "\"minMax\":{\"steps\":\""
				+ encodeFloat64(new double[] {2, 8, 15, 18, 25, 31})
				+ "\",\"values\":\""
				+ encodeFloat32(new float[] {
						-2 + offset, 8 + offset, 1 + offset,
						-3 + offset, 9 + offset, 2 + offset})
				+ "\"},"
				+ "\"summary\":{\"steps\":\""
				+ encodeFloat64(new double[] {0, 16})
				+ "\",\"mins\":\""
				+ encodeFloat32(new float[] {-2 + offset, -3 + offset})
				+ "\",\"maxs\":\""
				+ encodeFloat32(new float[] {8 + offset, 9 + offset})
				+ "\",\"means\":\""
				+ encodeFloat32(new float[] {3 + offset, 4 + offset})
				+ "\",\"minSteps\":\""
				+ encodeFloat64(new double[] {2, 18})
				+ "\",\"maxSteps\":\""
				+ encodeFloat64(new double[] {8, 25})
				+ "\"}}}";
	}

	static String statsWarningRunsJson() {
		return """
				{"runs":[
				  {
				    "id":"run_stats_a",
				    "generation":"%s",
				    "stats":{"maxStep":1},
				    "ingest":{"state":"ready","percentage":100},
				    "tags":[{
				      "key":"%s","type":"scalar","status":"ok",
				      "stats":{"minStep":0,"maxStep":1,"count":2,"lastValue":1,
				        "minValue":-1,"maxValue":1,"mean":0,"variance":1,"stdDev":1}
				    }]
				  },
				  {
				    "id":"run_stats_b",
				    "generation":"%s",
				    "stats":{"maxStep":1},
				    "ingest":{"state":"error","percentage":80,
				      "error":{"code":"invalid_json","message":"bad source line"}},
				    "tags":[{
				      "key":"%s","type":"scalar","status":"error",
				      "stats":{"minStep":0,"maxStep":1,"count":2,"lastValue":11,
				        "minValue":9,"maxValue":11,"mean":10,"variance":1,"stdDev":1},
				      "error":{"code":"tag_step_regression","message":"step regressed"}
				    }]
				  }
				]}
				""".formatted(GENERATION, TAG_KEY, GENERATION, TAG_KEY);
	}

	static String statsWarningMetricsJson() {
		return "{\"data\":["
				+ rawSeriesJson(
						"run_stats_a",
						TAG_KEY,
						new double[] {0, 1},
						new float[] {-1, 1})
				+ ","
				+ rawSeriesJson(
						"run_stats_b",
						TAG_KEY,
						new double[] {0, 1},
						new float[] {9, 11})
				+ "]}";
	}

	static String encodeFloat64(double[] values) {
		final ByteBuffer buffer = ByteBuffer.allocate(values.length * Double.BYTES)
				.order(ByteOrder.LITTLE_ENDIAN);
		for (double value : values) buffer.putDouble(value);
		return Base64.getEncoder().encodeToString(buffer.array());
	}

	static String encodeFloat32(float[] values) {
		final ByteBuffer buffer = ByteBuffer.allocate(values.length * Float.BYTES)
				.order(ByteOrder.LITTLE_ENDIAN);
		for (float value : values) buffer.putFloat(value);
		return Base64.getEncoder().encodeToString(buffer.array());
	}
}
