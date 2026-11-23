package io.github.kazukin123.anetlab.metricsviewer.service;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.PreparedStatement;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.system.CapturedOutput;
import org.springframework.boot.test.system.OutputCaptureExtension;
import org.springframework.http.MediaType;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.context.bean.override.mockito.MockitoSpyBean;
import org.springframework.test.web.servlet.MockMvc;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsCacheDatabase;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsCacheDatabase.IngestState;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsSource;

@SpringBootTest
@AutoConfigureMockMvc
class MetricsApiIntegrationTest {

	private static final Path RUNS_DIR = createFixture();
	private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

	@Autowired
	private MockMvc mockMvc;

	@MockitoBean
	private LoadingThread loadingThread;

	@MockitoSpyBean
	private MetricsCacheDatabase cacheDatabase;

	@DynamicPropertySource
	static void configure(DynamicPropertyRegistry registry) {
		registry.add("metricsviewer.workspaces-dir",
				() -> RUNS_DIR.getParent().getParent().toString());
		registry.add("metricsviewer.initial-workspace",
				() -> RUNS_DIR.getParent().getFileName().toString());
		registry.add("metricsviewer.cache-memory-mb", () -> "0");
		registry.add("metricsviewer.max-concurrent-queries", () -> "3");
		registry.add("metricsviewer.target-points-per-series", () -> "60");
		registry.add("metricsviewer.max-points-per-request", () -> "100");
	}

	@Test
	void validBatchKeepsOrderAndSeparatesAvailabilityFromIssues() throws Exception {
		awaitStates();
		final String request = """
				{"series":[
				  {"runId":"run-ready","tagKey":"a","stepBucketWidth":1,"fromStep":0,"toStep":2,"maxPoints":3},
				  {"runId":"run-pending","tagKey":"loss","stepBucketWidth":1,"fromStep":0,"toStep":1,"maxPoints":3},
				  {"runId":"run-ready","tagKey":"a","stepBucketWidth":1,"fromStep":100,"toStep":101,"maxPoints":3},
				  {"runId":"run-ready","tagKey":"missing","stepBucketWidth":1,"fromStep":0,"toStep":1,"maxPoints":3},
				  {"runId":"run-absent","tagKey":"a","stepBucketWidth":1,"fromStep":0,"toStep":1,"maxPoints":3},
				  {"runId":"run-error","tagKey":"loss","stepBucketWidth":1,"fromStep":0,"toStep":0,"maxPoints":3}
				]}
				""";

		final JsonNode data = postMetrics(request, 200).path("data");
		assertEquals(6, data.size());
		assertEquals(List.of("ok", "pending", "empty", "not_found", "not_found", "ok"),
				statuses(data));
		assertTrue(data.get(2).has("stepBucketWidth") && data.get(2).path("stepBucketWidth").isNull());
		assertTrue(data.get(2).has("projection") && data.get(2).path("projection").isNull());
		assertTrue(data.get(4).has("generation") && data.get(4).path("generation").isNull());
		assertEquals("run-error", data.get(5).path("runId").asText());
		assertEquals("raw", data.get(5).path("projection").path("kind").asText());
		assertEquals("invalid_json", data.get(5).path("issues").get(0).path("code").asText());
	}

	@Test
	void duplicateStepsOnAnInclusiveBoundaryAreAllReturned() throws Exception {
		awaitStates();
		final JsonNode result = postMetrics("""
				{"series":[{
				  "runId":"run-ready",
				  "tagKey":"dup",
				  "stepBucketWidth":1,"fromStep":1,
				  "toStep":1,
				  "maxPoints":3
				}]}
				""", 200).path("data").get(0);

		assertEquals("ok", result.path("availability").asText());
		assertEquals(List.of(1.0, 1.0), decodeDoubles(
				result.path("projection").path("steps").get(0).asText()));
	}

	@Test
	void runsApiUsesOnePercentageContractWithoutSourceKindFields() throws Exception {
		awaitStates();
		final JsonNode runs = OBJECT_MAPPER.readTree(mockMvc.perform(get("/api/runs.json"))
						.andReturn()
						.getResponse()
						.getContentAsString(StandardCharsets.UTF_8))
				.path("runs");
		final java.util.Map<String, JsonNode> byId = new java.util.HashMap<>();
		for (JsonNode run : runs) byId.put(run.path("id").asText(), run);

		assertEquals(100, byId.get("run-ready").path("ingest").path("percentage").asInt());
		assertEquals(0, byId.get("run-pending").path("ingest").path("percentage").asInt());
		assertTrue(byId.get("run-error").path("ingest").path("percentage").asInt() < 100);
		assertEquals("converting", byId.get("run-converting").path("ingest").path("state").asText());
		for (JsonNode run : runs) {
			assertTrue(run.path("ingest").path("percentage").asInt() >= 0);
			assertTrue(run.path("ingest").path("percentage").asInt() <= 100);
			assertTrue(!run.path("ingest").has("approximate"));
			assertTrue(!run.path("ingest").has("ingestedBytes"));
			assertTrue(!run.path("ingest").has("sourceBytes"));
		}
	}

	@Test
	@ExtendWith(OutputCaptureExtension.class)
	void unknownDatabaseStateFallsBackSafelyUntilCacheRebuild(CapturedOutput output)
			throws Exception {
		awaitStates();
		updateSourceMeta("run-ready", "state", "unknown");
		try {
			final String expectedStateError = "Unknown Metrics cache ingest state: unknown;"
					+ " expected one of: pending, converting, ready, error";
			final JsonNode runs = OBJECT_MAPPER.readTree(mockMvc.perform(get("/api/runs.json"))
						.andReturn()
						.getResponse()
						.getContentAsString(StandardCharsets.UTF_8))
					.path("runs");
			final JsonNode run = findRun(runs, "run-ready");
			assertEquals("pending", run.path("ingest").path("state").asText());
			assertEquals(0, run.path("ingest").path("percentage").asInt());
			assertTrue(run.path("generation").isNull());
			assertEquals(0, run.path("tags").size());

			final JsonNode result = postMetrics("""
					{"series":[{
					  "runId":"run-ready","tagKey":"a",
					  "stepBucketWidth":1,"fromStep":0,"toStep":1,"maxPoints":3
					}]}
					""", 200).path("data").get(0);
			assertEquals("pending", result.path("availability").asText());
			assertTrue(result.path("projection").isNull());
			assertEquals("run", result.path("issues").get(0).path("scope").asText());
			assertEquals("query_error", result.path("issues").get(0).path("code").asText());
			assertEquals(expectedStateError,
					result.path("issues").get(0).path("message").asText());
			assertTrue(output.getAll().contains(
					"Failed to read Run metadata: run=run-ready"
							+ " message=" + expectedStateError));
		} finally {
			updateSourceMeta("run-ready", "state", IngestState.READY.externalName());
		}
	}

	@Test
	void invalidSourceSizeUsesTheRunsDisplayFallback() throws Exception {
		awaitStates();
		final String originalSourceSize = sourceMetaValue("run-ready", "source_size");
		updateSourceMeta("run-ready", "source_size", "not-a-number");
		try {
			final JsonNode runs = OBJECT_MAPPER.readTree(mockMvc.perform(get("/api/runs.json"))
						.andReturn()
						.getResponse()
						.getContentAsString(StandardCharsets.UTF_8))
					.path("runs");
			final JsonNode run = findRun(runs, "run-ready");
			assertEquals("ready", run.path("ingest").path("state").asText());
			assertEquals(100, run.path("ingest").path("percentage").asInt());
		} finally {
			updateSourceMeta("run-ready", "source_size", originalSourceSize);
		}
	}

	@Test
	void fullResolutionOverflowReturnsRequiredPoints() throws Exception {
		awaitStates();
		final JsonNode body = postMetrics("""
				{"series":[
				  {"runId":"run-ready","tagKey":"a","stepBucketWidth":1,"fromStep":0,"toStep":63,"maxPoints":80},
				  {"runId":"run-ready","tagKey":"b","stepBucketWidth":1,"fromStep":0,"toStep":63,"maxPoints":80},
				  {"runId":"run-ready","tagKey":"c","stepBucketWidth":1,"fromStep":0,"toStep":63,"maxPoints":80}
				]}
				""", 422);

		assertEquals(3, body.path("seriesCount").asInt());
		assertEquals(192, body.path("requiredPoints").asInt());
		assertEquals(100, body.path("maxPointsPerRequest").asInt());
	}

	@Test
	void requestOverflowIsRejectedInsteadOfReducingComparisonResolution() throws Exception {
		awaitStates();
		final JsonNode body = postMetrics("""
				{"series":[
				  {"runId":"run-ready","tagKey":"a","fromStep":0,"toStep":63,"maxPoints":80,"stepBucketWidth":1},
				  {"runId":"run-ready","tagKey":"b","fromStep":0,"toStep":63,"maxPoints":80,"stepBucketWidth":1}
				]}
				""", 422);
		assertEquals(128, body.path("requiredPoints").asInt());
		assertEquals(100, body.path("maxPointsPerRequest").asInt());
	}

	@Test
	void retiredAndUnknownRequestFieldsAreRejected() throws Exception {
		awaitStates();
		mockMvc.perform(post("/api/metrics.json")
						.contentType("application/json")
						.header("X-Query-Channel", UUID.randomUUID().toString())
						.header("X-Query-Sequence", "0")
						.content("""
								{"series":[{
								  "runId":"run-ready","tagKey":"a",
								  "stepBucketWidth":1,"fromStep":0,"toStep":1,"fromOrdinal":0
								}]}
								"""))
				.andExpect(status().isBadRequest());
		mockMvc.perform(post("/api/metrics.json")
						.contentType("application/json")
						.header("X-Query-Channel", UUID.randomUUID().toString())
						.header("X-Query-Sequence", "0")
						.content("""
								{"series":[{
								  "runId":"run-ready","tagKey":"a",
								  "stepBucketWidth":1,"fromStep":9007199254740992,"toStep":9007199254740992
								}]}
								"""))
				.andExpect(status().isBadRequest());
	}

	@Test
	void requestShapeDoesNotCoerceStringsOrFractionsIntoNumericFields() throws Exception {
		awaitStates();
		for (String request : List.of(
				"""
				{"series":[{
				  "runId":"run-ready","tagKey":"a",
				  "stepBucketWidth":1,"fromStep":"0","toStep":1
				}]}
				""",
				"""
				{"series":[{
				  "runId":"run-ready","tagKey":"a",
				  "stepBucketWidth":1,"fromStep":0.5,"toStep":1
				}]}
				""",
				"""
				{"series":[{
				  "runId":123,"tagKey":"a",
				  "stepBucketWidth":1,"fromStep":0,"toStep":1
				}]}
				""",
				"""
				{"series":[{
				  "runId":"run-ready","tagKey":"a",
				  "stepBucketWidth":1,"fromStep":0,"toStep":1,"maxPoints":3.5
				}]}
				""")) {
			mockMvc.perform(post("/api/metrics.json")
							.contentType("application/json")
							.header("X-Query-Channel", UUID.randomUUID().toString())
							.header("X-Query-Sequence", "0")
							.content(request))
					.andExpect(status().isBadRequest());
		}
	}

	@Test
	void queryHeadersAreRequiredAndStrictlyValidated() throws Exception {
		awaitStates();
		final String request = """
				{"series":[{
				  "runId":"run-ready","tagKey":"a",
				  "stepBucketWidth":1,"fromStep":0,"toStep":1,"maxPoints":3
				}]}
				""";

		assertInvalidQueryHeaders(request, null, "0");
		assertInvalidQueryHeaders(request, "browser-tab", null);
		assertInvalidQueryHeaders(request, " ", "0");
		assertInvalidQueryHeaders(request, "x".repeat(129), "0");
		assertInvalidQueryHeaders(request, "browser-tab", "-1");
		assertInvalidQueryHeaders(request, "browser-tab", "9007199254740992");
		assertInvalidQueryHeaders(request, "browser-tab", "not-a-number");
	}

	@Test
	void aLateSequenceReturnsTheSupersededHttpContract() throws Exception {
		awaitStates();
		final String request = """
				{"series":[{
				  "runId":"run-ready","tagKey":"a",
				  "stepBucketWidth":1,"fromStep":0,"toStep":1,"maxPoints":3
				}]}
				""";
		final String channel = "late-sequence-" + UUID.randomUUID();
		mockMvc.perform(post("/api/metrics.json")
						.contentType("application/json")
						.header("X-Query-Channel", channel)
						.header("X-Query-Sequence", "1")
						.content(request))
				.andExpect(status().isOk());

		final var response = mockMvc.perform(post("/api/metrics.json")
						.contentType("application/json")
						.header("X-Query-Channel", channel)
						.header("X-Query-Sequence", "0")
						.content(request))
				.andExpect(status().isConflict())
				.andReturn()
				.getResponse();
		assertNull(response.getHeader("Retry-After"));
		assertEquals("superseded", OBJECT_MAPPER.readTree(
				response.getContentAsString(StandardCharsets.UTF_8)).path("code").asText());
	}

	@Test
	void invalidOrMissingStepBucketWidthIsRejected() throws Exception {
		for (String width:List.of("null","0","-1","9007199254740992","1.5","\"2\"")) {
			postMetrics("{\"series\":[{\"runId\":\"run-ready\",\"tagKey\":\"a\",\"fromStep\":0,\"toStep\":1,\"stepBucketWidth\":"+width+"}]}",400);
		}
		postMetrics("{\"series\":[{\"runId\":\"run-ready\",\"tagKey\":\"a\",\"fromStep\":0,\"toStep\":1}]}",400);
	}

	private JsonNode postMetrics(String request, int expectedStatus) throws Exception {
		final var response = mockMvc.perform(post("/api/metrics.json")
						.contentType("application/json")
						.header("X-Query-Channel", UUID.randomUUID().toString())
						.header("X-Query-Sequence", "0")
						.content(request))
				.andExpect(status().is(expectedStatus))
				.andReturn()
				.getResponse();
		return OBJECT_MAPPER.readTree(response.getContentAsString(StandardCharsets.UTF_8));
	}

	private void assertInvalidQueryHeaders(String request, String channel, String sequence)
			throws Exception {
		final var builder = post("/api/metrics.json")
				.contentType("application/json")
				.content(request);
		if (channel != null) builder.header("X-Query-Channel", channel);
		if (sequence != null) builder.header("X-Query-Sequence", sequence);
		final String body = mockMvc.perform(builder)
				.andExpect(status().isBadRequest())
				.andReturn()
				.getResponse()
				.getContentAsString(StandardCharsets.UTF_8);
		assertEquals("invalid_request", OBJECT_MAPPER.readTree(body).path("code").asText());
	}

	private void awaitStates() throws Exception {
		final Instant deadline = Instant.now().plus(Duration.ofSeconds(10));
		JsonNode last = null;
		while (Instant.now().isBefore(deadline)) {
			final String body = mockMvc.perform(get("/api/runs.json"))
					.andReturn()
					.getResponse()
					.getContentAsString(StandardCharsets.UTF_8);
			last = OBJECT_MAPPER.readTree(body).path("runs");
			final java.util.Map<String, String> states = new java.util.HashMap<>();
			for (JsonNode run : last) {
				states.put(run.path("id").asText(), run.path("ingest").path("state").asText());
			}
			if ("ready".equals(states.get("run-ready"))
					&& "pending".equals(states.get("run-pending"))
					&& "converting".equals(states.get("run-converting"))
					&& "error".equals(states.get("run-error"))) {
				return;
			}
			Thread.sleep(50);
		}
		throw new AssertionError("Runs did not reach expected states: " + last);
	}

	private static List<String> statuses(JsonNode data) {
		final List<String> statuses = new ArrayList<>();
		for (JsonNode result : data) statuses.add(result.path("availability").asText());
		return statuses;
	}

	private static JsonNode findRun(JsonNode runs, String runId) {
		for (JsonNode run : runs) {
			if (runId.equals(run.path("id").asText())) return run;
		}
		throw new AssertionError("Run not found: " + runId);
	}

	private static String sourceMetaValue(String runId, String key) throws Exception {
		final MetricsCacheDatabase database = new MetricsCacheDatabase();
		try (MetricsCacheDatabase.ConnectionHandle handle =
				database.openRead(RUNS_DIR.resolve(runId));
				PreparedStatement statement = handle.connection().prepareStatement(
						"SELECT v FROM source_meta WHERE k=?")) {
			statement.setString(1, key);
			try (var result = statement.executeQuery()) {
				assertTrue(result.next());
				return result.getString(1);
			}
		}
	}

	private static void updateSourceMeta(String runId, String key, String value) throws Exception {
		final MetricsCacheDatabase database = new MetricsCacheDatabase();
		try (MetricsCacheDatabase.ConnectionHandle handle =
				database.openWrite(RUNS_DIR.resolve(runId));
				PreparedStatement statement = handle.connection().prepareStatement(
						"UPDATE source_meta SET v=? WHERE k=?")) {
			statement.setString(1, value);
			statement.setString(2, key);
			assertEquals(1, statement.executeUpdate());
		}
	}

	private static List<Double> decodeDoubles(String encoded) {
		final ByteBuffer buffer = ByteBuffer.wrap(Base64.getDecoder().decode(encoded))
				.order(ByteOrder.LITTLE_ENDIAN);
		final List<Double> values = new ArrayList<>();
		while (buffer.remaining() >= Double.BYTES) values.add(buffer.getDouble());
		return values;
	}


	@Test
	void configRequestReportsEachRunIndependentlyAndRejectsInvalidBodies() throws Exception {
		writeConfig("run-ready", "z = last\na = first\n");
		writeConfig("run-error", "a=1\n$ INCLUDE extra.txt\n");
		writeConfig("run-converting", "a=1\nbad? = 2\n");

		final JsonNode rows = OBJECT_MAPPER.readTree(mockMvc.perform(post("/api/config.json")
						.contentType(MediaType.APPLICATION_JSON)
						.content("""
								{"runIds":["run-ready","run-pending","run-error","run-converting","../unknown"]}
								"""))
				.andExpect(status().isOk()).andReturn().getResponse().getContentAsString()).path("runs");
		assertEquals(5, rows.size());
		assertEquals("ok", rows.get(0).path("status").asText());
		assertEquals("a", rows.get(0).path("entries").get(0).path("key").asText());
		assertEquals("first", rows.get(0).path("entries").get(0).path("value").asText());
		assertEquals("z", rows.get(0).path("entries").get(1).path("key").asText());
		assertEquals("missing", rows.get(1).path("status").asText());
		assertEquals("error", rows.get(2).path("status").asText());
		assertTrue(rows.get(2).path("message").asText().contains("config_data.txt:2 (R5)"));
		assertEquals("error", rows.get(3).path("status").asText());
		assertTrue(rows.get(3).path("message").asText().contains("config_data.txt:2 (R8)"));
		assertEquals("not_found", rows.get(4).path("status").asText());
		assertEquals("../unknown", rows.get(4).path("runId").asText());

		for (String body : List.of("{}", "{\"unknown\":true,\"runIds\":[]}",
				"{\"runIds\":[17]}", "{\"runIds\":[null]}", "{\"runIds\":[]} {}")) {
			final JsonNode error = OBJECT_MAPPER.readTree(mockMvc.perform(post("/api/config.json")
							.contentType(MediaType.APPLICATION_JSON).content(body))
					.andExpect(status().isBadRequest()).andReturn().getResponse().getContentAsString());
			assertEquals("invalid_request", error.path("code").asText());
		}
	}


	@Test
	void rangeStatsUsesEveryCommittedPointInsideTheClosedStepRange() throws Exception {
		awaitStates();
		final JsonNode rows = postRangeStats("""
				{"series":[
				  {"runId":"run-ready","tagKey":"a","fromStep":2,"toStep":4},
				  {"runId":"run-ready","tagKey":"dup","fromStep":1,"toStep":1},
				  {"runId":"run-ready","tagKey":"dup","fromStep":2,"toStep":2},
				  {"runId":"run-ready","tagKey":"dup","fromStep":3,"toStep":4},
				  {"runId":"run-ready","tagKey":"missing","fromStep":0,"toStep":2},
				  {"runId":"run-absent","tagKey":"a","fromStep":0,"toStep":2},
				  {"runId":"run-pending","tagKey":"loss","fromStep":0,"toStep":2},
				  {"runId":"run-ready","tagKey":"a","fromStep":0,"toStep":63},
				  {"runId":"run-ready","tagKey":"b","fromStep":0,"toStep":63}
				]}
				""", UUID.randomUUID().toString(), "0", 200).path("data");
		assertEquals(List.of("ok", "ok", "ok", "empty", "not_found", "not_found", "pending", "ok", "ok"),
				statuses(rows));
		final JsonNode first = rows.get(0);
		assertEquals("run-ready", first.path("runId").asText());
		assertEquals("a", first.path("tagKey").asText());
		assertEquals(2, first.path("fromStep").asInt());
		assertEquals(4, first.path("toStep").asInt());
		assertTrue(first.path("generation").isTextual());
		assertEquals(3L, first.path("count").asLong());
		assertEquals(3.0, first.path("mean").asDouble());
		assertEquals(2.0 / 3.0, first.path("variance").asDouble(), 1e-12);
		assertEquals(Math.sqrt(2.0 / 3.0), first.path("stdDev").asDouble(), 1e-12);
		assertEquals(2.0, first.path("minValue").asDouble());
		assertEquals(4.0, first.path("maxValue").asDouble());
		assertEquals(4.0, first.path("lastValue").asDouble());
		assertEquals(2L, first.path("minStep").asLong());
		assertEquals(4L, first.path("maxStep").asLong());
		assertEquals(2L, rows.get(1).path("count").asLong());
		assertEquals(2.0, rows.get(1).path("lastValue").asDouble());
		assertEquals(1L, rows.get(2).path("count").asLong());
		assertEquals(3.0, rows.get(2).path("lastValue").asDouble());
		assertTrue(!rows.get(3).has("count"));
		assertTrue(rows.get(5).path("generation").isNull());
		assertEquals(64L, rows.get(8).path("count").asLong());
	}

	@Test
	void rangeStatsRejectsInvalidSchemaAndHeaders() throws Exception {
		final String valid = """
				{"series":[{"runId":"run-ready","tagKey":"a","fromStep":0,"toStep":2}]}
				""";
		final String channel = UUID.randomUUID().toString();
		for (String body : List.of("{}", "{\"series\":[null]}",
				valid.replace("\"fromStep\":0,", ""),
				valid.replace("\"fromStep\":0", "\"fromStep\":3"),
				valid.replace("\"fromStep\":0", "\"fromStep\":0.5"),
				valid.replace("\"fromStep\":0", "\"fromStep\":\"0\""),
				valid.replace("\"fromStep\":0", "\"fromStep\":-9007199254740992"),
				valid.replace("\"fromStep\":0", "\"fromStep\":0,\"stepBucketWidth\":1"))) {
			assertEquals("invalid_request", postRangeStats(body, channel, "0", 400).path("code").asText());
		}
		assertEquals("invalid_request", postRangeStats(valid, " ", "0", 400).path("code").asText());
		assertEquals("invalid_request", postRangeStats(valid, channel, "-1", 400).path("code").asText());
	}


	@Test
	void separateGraphAndRangeChannelsOnlySupersedeTheirOwnRequests() throws Exception {
		awaitStates();
		final String scope = UUID.randomUUID().toString();
		final String graphChannel = scope + "-graph";
		final String rangeChannel = scope + "-range";
		final String graphRequest = """
				{"series":[{"runId":"run-ready","tagKey":"a","fromStep":0,"toStep":2,"stepBucketWidth":1}]}
				""";
		final String rangeRequest = """
				{"series":[{"runId":"run-ready","tagKey":"a","fromStep":0,"toStep":2}]}
				""";
		final CountDownLatch graphEntered = new CountDownLatch(1);
		final CountDownLatch rangeEntered = new CountDownLatch(1);
		final CountDownLatch release = new CountDownLatch(1);
		final ThreadLocal<CountDownLatch> pauseAtRead = new ThreadLocal<>();

		// SQLite の読み取り直前で止め、両方の HTTP 要求が生きている間に新しい範囲統計を送る。
		org.mockito.Mockito.doAnswer(call -> {
			final CountDownLatch entered = pauseAtRead.get();
			if (entered != null) {
				entered.countDown();
				assertTrue(release.await(10, TimeUnit.SECONDS), "Timed out waiting to release query");
			}
			return call.callRealMethod();
		}).when(cacheDatabase).openRead(RUNS_DIR.resolve("run-ready"));

		final ExecutorService executor = Executors.newFixedThreadPool(2);
		try {
			final var graph = executor.submit(() -> {
				pauseAtRead.set(graphEntered);
				try {
					return mockMvc.perform(post("/api/metrics.json")
									.contentType(MediaType.APPLICATION_JSON).content(graphRequest)
									.header("X-Query-Channel", graphChannel).header("X-Query-Sequence", "0"))
							.andReturn().getResponse().getStatus();
				} finally {
					pauseAtRead.remove();
				}
			});
			assertTrue(graphEntered.await(5, TimeUnit.SECONDS));
			final var oldRange = executor.submit(() -> {
				pauseAtRead.set(rangeEntered);
				try {
					return postRangeStats(rangeRequest, rangeChannel, "0", 409);
				} finally {
					pauseAtRead.remove();
				}
			});
			assertTrue(rangeEntered.await(5, TimeUnit.SECONDS));
			assertEquals("ok", postRangeStats(rangeRequest, rangeChannel, "1", 200)
					.path("data").get(0).path("availability").asText());
			release.countDown();
			assertEquals(200, graph.get(5, TimeUnit.SECONDS));
			assertEquals("superseded", oldRange.get(5, TimeUnit.SECONDS).path("code").asText());
		} finally {
			release.countDown();
			executor.shutdownNow();
		}
	}

	private JsonNode postRangeStats(String request, String channel, String sequence, int expectedStatus)
			throws Exception {
		final var response = mockMvc.perform(post("/api/range-stats.json")
						.contentType(MediaType.APPLICATION_JSON)
						.header("X-Query-Channel", channel)
						.header("X-Query-Sequence", sequence)
						.content(request))
				.andExpect(status().is(expectedStatus))
				.andReturn().getResponse();
		return OBJECT_MAPPER.readTree(response.getContentAsString(StandardCharsets.UTF_8));
	}

	private static void writeConfig(String runId, String content) throws Exception {
		final Path configDir = RUNS_DIR.resolve(runId).resolve("config");
		Files.createDirectories(configDir);
		Files.writeString(configDir.resolve("config_data.txt"), content);
	}

	private static Path createFixture() {
		try {
			final Path runsDir = Path.of(
					"target", "metrics-api-integration-" + System.nanoTime(), "_test", "runs")
					.toAbsolutePath()
					.normalize();
			final Path ready = runsDir.resolve("run-ready");
			final Path pending = runsDir.resolve("run-pending");
			final Path converting = runsDir.resolve("run-converting");
			final Path error = runsDir.resolve("run-error");
			Files.createDirectories(ready);
			Files.createDirectories(pending);
			Files.createDirectories(converting);
			Files.createDirectories(error);

			final StringBuilder jsonl = new StringBuilder();
			for (int step = 0; step < 64; step++) {
				for (String tag : List.of("a", "b", "c")) {
					jsonl.append("{\"type\":\"scalar\",\"tag\":\"")
							.append(tag)
							.append("\",\"step\":")
							.append(step)
							.append(",\"value\":")
							.append(step)
							.append(".0}\n");
				}
			}
			jsonl.append("{\"type\":\"scalar\",\"tag\":\"dup\",\"step\":1,\"value\":1.0}\n")
					.append("{\"type\":\"scalar\",\"tag\":\"dup\",\"step\":1,\"value\":2.0}\n")
					.append("{\"type\":\"scalar\",\"tag\":\"dup\",\"step\":2,\"value\":3.0}\n");
			Files.writeString(ready.resolve("metrics.jsonl"), jsonl, StandardCharsets.UTF_8);
			Files.writeString(
					pending.resolve("metrics.jsonl"),
					"{\"type\":\"scalar\",\"tag\":\"loss\",\"step\":0,\"value\":1.0}",
					StandardCharsets.UTF_8);
			Files.writeString(
					converting.resolve("metrics.jsonl"),
					"""
					{"type":"scalar","tag":"loss","step":0,"value":1.0}
					{"type":"scalar","tag":"loss","step":1,"value":2.0}
					""",
					StandardCharsets.UTF_8);
			Files.writeString(
					error.resolve("metrics.jsonl"),
					"""
					{"type":"scalar","tag":"loss","step":0,"value":1.0}
					{invalid json}
					""",
					StandardCharsets.UTF_8);

			final MetricsCacheDatabase database = new MetricsCacheDatabase();
			final MetricsSource readySource = MetricsSource.select(ready).orElseThrow();
			assertEquals(IngestState.READY,
					new MetricsIngestor(database).ingestBlock(
							"run-ready", ready, readySource).state());
			final MetricsSource pendingSource = MetricsSource.select(pending).orElseThrow();
			database.prepare(pending, pendingSource, false);
			final MetricsIngestor ingestor =
					new MetricsIngestor(database, new GzipInputSessions(), 1);
			final MetricsSource convertingSource = MetricsSource.select(converting).orElseThrow();
			assertEquals(IngestState.CONVERTING,
					ingestor.ingestBlock("run-converting", converting, convertingSource).state());
			final MetricsSource errorSource = MetricsSource.select(error).orElseThrow();
			assertEquals(IngestState.CONVERTING,
					ingestor.ingestBlock("run-error", error, errorSource).state());
			assertEquals(IngestState.ERROR,
					ingestor.ingestBlock("run-error", error, errorSource).state());
			return runsDir;
		} catch (Exception e) {
			throw new ExceptionInInitializerError(e);
		}
	}
}
