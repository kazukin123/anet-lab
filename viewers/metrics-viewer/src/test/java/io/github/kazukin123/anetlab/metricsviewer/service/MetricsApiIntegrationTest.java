package io.github.kazukin123.anetlab.metricsviewer.service;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.MockMvc;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsCacheDatabase;
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

	@DynamicPropertySource
	static void configure(DynamicPropertyRegistry registry) {
		registry.add("metricsviewer.runs-dir", () -> RUNS_DIR.toString());
		registry.add("metricsviewer.cache-memory-mb", () -> "0");
		registry.add("metricsviewer.target-points-per-series", () -> "60");
		registry.add("metricsviewer.max-points-per-request", () -> "100");
	}

	@Test
	void validBatchKeepsOrderAndSeparatesAvailabilityFromIssues() throws Exception {
		awaitStates();
		final String request = """
				{"series":[
				  {"runId":"run-ready","tagKey":"a","fromStep":0,"toStep":2,"maxPoints":3},
				  {"runId":"run-pending","tagKey":"loss","fromStep":0,"toStep":1,"maxPoints":3},
				  {"runId":"run-ready","tagKey":"a","fromStep":100,"toStep":101,"maxPoints":3},
				  {"runId":"run-ready","tagKey":"missing","fromStep":0,"toStep":1,"maxPoints":3},
				  {"runId":"run-absent","tagKey":"a","fromStep":0,"toStep":1,"maxPoints":3},
				  {"runId":"run-error","tagKey":"loss","fromStep":0,"toStep":0,"maxPoints":3}
				]}
				""";

		final JsonNode data = postMetrics(request, 200).path("data");
		assertEquals(6, data.size());
		assertEquals(List.of("ok", "pending", "empty", "not_found", "not_found", "ok"),
				statuses(data));
		assertTrue(data.get(2).has("level") && data.get(2).path("level").isNull());
		assertTrue(data.get(2).has("bucketWidth") && data.get(2).path("bucketWidth").isNull());
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
				  "fromStep":1,
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
		for (JsonNode run : runs) {
			assertTrue(run.path("ingest").path("percentage").asInt() >= 0);
			assertTrue(run.path("ingest").path("percentage").asInt() <= 100);
			assertTrue(!run.path("ingest").has("approximate"));
			assertTrue(!run.path("ingest").has("ingestedBytes"));
			assertTrue(!run.path("ingest").has("sourceBytes"));
		}
	}

	@Test
	void minimumQuotaOverflowReturnsTheThreeContractFields() throws Exception {
		awaitStates();
		final JsonNode body = postMetrics("""
				{"series":[
				  {"runId":"run-ready","tagKey":"a","fromStep":0,"toStep":63,"maxPoints":60},
				  {"runId":"run-ready","tagKey":"b","fromStep":0,"toStep":63,"maxPoints":60},
				  {"runId":"run-ready","tagKey":"c","fromStep":0,"toStep":63,"maxPoints":60}
				]}
				""", 422);

		assertEquals(3, body.path("seriesCount").asInt());
		assertEquals(150, body.path("requiredMinimumPoints").asInt());
		assertEquals(100, body.path("maxPointsPerRequest").asInt());
	}

	@Test
	void waterFillingKeepsEqualSeriesBudgetsFair() throws Exception {
		awaitStates();
		final JsonNode data = postMetrics("""
				{"series":[
				  {"runId":"run-ready","tagKey":"a","fromStep":0,"toStep":63,"maxPoints":80},
				  {"runId":"run-ready","tagKey":"b","fromStep":0,"toStep":63,"maxPoints":80}
				]}
				""", 200).path("data");

		assertEquals(50, data.get(0).path("pointBudget").asInt());
		assertEquals(50, data.get(1).path("pointBudget").asInt());
	}

	@Test
	void retiredAndUnknownRequestFieldsAreRejected() throws Exception {
		awaitStates();
		mockMvc.perform(post("/api/metrics.json")
						.contentType("application/json")
						.content("""
								{"series":[{
								  "runId":"run-ready","tagKey":"a",
								  "fromStep":0,"toStep":1,"fromOrdinal":0
								}]}
								"""))
				.andExpect(status().isBadRequest());
		mockMvc.perform(post("/api/metrics.json")
						.contentType("application/json")
						.content("""
								{"series":[{
								  "runId":"run-ready","tagKey":"a",
								  "fromStep":9007199254740992,"toStep":9007199254740992
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
				  "fromStep":"0","toStep":1
				}]}
				""",
				"""
				{"series":[{
				  "runId":"run-ready","tagKey":"a",
				  "fromStep":0.5,"toStep":1
				}]}
				""",
				"""
				{"series":[{
				  "runId":123,"tagKey":"a",
				  "fromStep":0,"toStep":1
				}]}
				""",
				"""
				{"series":[{
				  "runId":"run-ready","tagKey":"a",
				  "fromStep":0,"toStep":1,"maxPoints":3.5
				}]}
				""")) {
			mockMvc.perform(post("/api/metrics.json")
							.contentType("application/json")
							.content(request))
					.andExpect(status().isBadRequest());
		}
	}

	private JsonNode postMetrics(String request, int expectedStatus) throws Exception {
		final var response = mockMvc.perform(post("/api/metrics.json")
						.contentType("application/json")
						.content(request))
				.andExpect(status().is(expectedStatus))
				.andReturn()
				.getResponse();
		return OBJECT_MAPPER.readTree(response.getContentAsString(StandardCharsets.UTF_8));
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

	private static List<Double> decodeDoubles(String encoded) {
		final ByteBuffer buffer = ByteBuffer.wrap(Base64.getDecoder().decode(encoded))
				.order(ByteOrder.LITTLE_ENDIAN);
		final List<Double> values = new ArrayList<>();
		while (buffer.remaining() >= Double.BYTES) values.add(buffer.getDouble());
		return values;
	}

	private static Path createFixture() {
		try {
			final Path runsDir = Path.of("target", "metrics-api-integration-" + System.nanoTime())
					.toAbsolutePath()
					.normalize();
			final Path ready = runsDir.resolve("run-ready");
			final Path pending = runsDir.resolve("run-pending");
			final Path error = runsDir.resolve("run-error");
			Files.createDirectories(ready);
			Files.createDirectories(pending);
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
					error.resolve("metrics.jsonl"),
					"""
					{"type":"scalar","tag":"loss","step":0,"value":1.0}
					{invalid json}
					""",
					StandardCharsets.UTF_8);

			final MetricsCacheDatabase database = new MetricsCacheDatabase();
			final MetricsSource readySource = MetricsSource.select(ready).orElseThrow();
			assertEquals("ready",
					new MetricsIngestor(database).ingestBlock(
							"run-ready", ready, readySource).state());
			final MetricsSource pendingSource = MetricsSource.select(pending).orElseThrow();
			database.prepare(pending, pendingSource, false);
			final MetricsIngestor ingestor =
					new MetricsIngestor(database, new GzipInputSessions(), 1);
			final MetricsSource errorSource = MetricsSource.select(error).orElseThrow();
			assertEquals("converting",
					ingestor.ingestBlock("run-error", error, errorSource).state());
			assertEquals("error",
					ingestor.ingestBlock("run-error", error, errorSource).state());
			return runsDir;
		} catch (Exception e) {
			throw new ExceptionInInitializerError(e);
		}
	}
}
