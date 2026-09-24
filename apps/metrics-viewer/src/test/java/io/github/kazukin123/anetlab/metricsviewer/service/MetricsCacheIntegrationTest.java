package io.github.kazukin123.anetlab.metricsviewer.service;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.charset.StandardCharsets;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;
import java.util.Base64;
import java.util.List;
import java.util.Set;
import java.util.UUID;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.springframework.test.web.servlet.MockMvc;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@SpringBootTest
@AutoConfigureMockMvc
class MetricsCacheIntegrationTest {

	private static final Path RUNS_DIR = createFixture();
	private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

	@Autowired
	private MockMvc mockMvc;

	@Autowired
	private WorkspaceManager workspaceManager;

	@DynamicPropertySource
	static void configureRunsDir(DynamicPropertyRegistry registry) {
		registry.add("metricsviewer.workspaces-dir",
				() -> RUNS_DIR.getParent().getParent().toString());
		registry.add("metricsviewer.initial-workspace",
				() -> RUNS_DIR.getParent().getFileName().toString());
		registry.add("metricsviewer.cache-memory-mb", () -> "0");
	}

	@Test
	void jsonlIsIngestedIntoSqliteAndRunsApiPublishesExactTagStats() throws Exception {
		final JsonNode run = awaitReadyRun();

		assertEquals("run-a", run.path("id").asText());
		assertNotNull(UUID.fromString(run.path("generation").asText()));
		assertEquals("ready", run.path("ingest").path("state").asText());
		assertEquals(100, run.path("ingest").path("percentage").asInt());
		assertEquals(2, run.path("stats").path("maxStep").asLong());

		final JsonNode tag = run.path("tags").get(0);
		assertEquals("loss", tag.path("key").asText());
		assertEquals("scalar", tag.path("type").asText());
		assertEquals("ok", tag.path("status").asText());
		assertEquals(2, tag.path("stats").path("count").asLong());
		assertEquals(1, tag.path("stats").path("minStep").asLong());
		assertEquals(2, tag.path("stats").path("maxStep").asLong());
		assertEquals(-2.0, tag.path("stats").path("minValue").asDouble());
		assertEquals(4.0, tag.path("stats").path("maxValue").asDouble());
		assertEquals(-2.0, tag.path("stats").path("lastValue").asDouble());
		assertEquals(1.0, tag.path("stats").path("mean").asDouble());
		assertEquals(9.0, tag.path("stats").path("variance").asDouble());
		assertEquals(3.0, tag.path("stats").path("stdDev").asDouble());
		final JsonNode single = run.path("tags").get(1);
		assertEquals("single", single.path("key").asText());
		assertEquals(1, single.path("stats").path("count").asLong());
		assertEquals(0.0, single.path("stats").path("variance").asDouble());
		assertEquals(0.0, single.path("stats").path("stdDev").asDouble());

		final Path database = RUNS_DIR.resolve("run-a").resolve("metrics_cache.db");
		assertTrue(Files.isRegularFile(database));
	}

	@Test
	void rangeApiReturnsInclusiveRawProjection() throws Exception {
		final JsonNode run = awaitReadyRun();
		final String request = """
				{
				  "series": [{
				    "runId": "run-a",
				    "tagKey": "loss",
				    "fromStep": 1,
				    "toStep": 2,
				    "maxPoints": 3
				  }]
				}
				""";

		final String body = mockMvc.perform(post("/api/metrics.json")
						.contentType("application/json")
						.header("X-Query-Channel", "metrics-cache-test")
						.header("X-Query-Sequence", "0")
						.content(request))
				.andReturn()
				.getResponse()
				.getContentAsString(StandardCharsets.UTF_8);
		final JsonNode result = OBJECT_MAPPER.readTree(body).path("data").get(0);

		assertEquals("run-a", result.path("runId").asText());
		assertEquals("loss", result.path("tagKey").asText());
		assertEquals(run.path("generation").asText(), result.path("generation").asText());
		assertEquals(1, result.path("fromStep").asLong());
		assertEquals(2, result.path("toStep").asLong());
		assertEquals("ok", result.path("availability").asText());
		assertEquals(2, result.path("pointBudget").asInt());
		assertEquals(0, result.path("level").asInt());
		assertEquals(1, result.path("bucketWidth").asLong());
		assertEquals("raw", result.path("projection").path("kind").asText());
		assertEquals(List.of(1.0, 2.0),
				decodeDoubles(result.path("projection").path("steps").get(0).asText()));
		assertEquals(List.of(4.0f, -2.0f),
				decodeFloats(result.path("projection").path("values").get(0).asText()));
	}

	@Test
	void prioritizeApiReplacesTheSetAndRejectsUnknownRuns() throws Exception {
		awaitReadyRun();
		mockMvc.perform(post("/api/runs/prioritize")
						.contentType("application/json")
						.content("{\"runIds\":[\"run-a\",\"run-a\"]}"))
				.andExpect(status().isNoContent());
		assertEquals(Set.of("run-a"), priorityRunIds());
		mockMvc.perform(post("/api/runs/prioritize")
						.contentType("application/json")
						.content("{\"runIds\":[]}"))
				.andExpect(status().isNoContent());
		assertEquals(Set.of(), priorityRunIds());
		mockMvc.perform(post("/api/runs/prioritize")
						.contentType("application/json")
						.content("{\"runIds\":[\"missing\"]}"))
				.andExpect(status().isBadRequest());
	}

	private Set<String> priorityRunIds() {
		try (WorkspaceManager.Lease lease = workspaceManager.acquireLease()) {
			return lease.ingestScheduler().priorityRunIds();
		}
	}

	private JsonNode awaitReadyRun() throws Exception {
		final Instant deadline = Instant.now().plus(Duration.ofSeconds(10));
		JsonNode lastResponse = null;
		while (Instant.now().isBefore(deadline)) {
			final String body = mockMvc.perform(get("/api/runs.json"))
					.andReturn()
					.getResponse()
					.getContentAsString(StandardCharsets.UTF_8);
			lastResponse = OBJECT_MAPPER.readTree(body);
			final JsonNode runs = lastResponse.path("runs");
			if (runs.size() == 1 && "ready".equals(runs.get(0).path("ingest").path("state").asText())) {
				return runs.get(0);
			}
			Thread.sleep(50);
		}
		throw new AssertionError("Run did not become ready: " + lastResponse);
	}

	private static Path createFixture() {
		try {
			final Path runsDir = Path.of(
					"target", "metrics-cache-integration-" + System.nanoTime(), "_test", "runs")
					.toAbsolutePath()
					.normalize();
			final Path runDir = runsDir.resolve("run-a");
			Files.createDirectories(runDir);
			Files.writeString(runDir.resolve("metrics.jsonl"),
					"""
					{"type":"scalar","tag":"loss","step":1,"value":4.0}
					{"type":"scalar","tag":"loss","step":2,"value":-2.0}
					{"type":"scalar","tag":"single","step":1,"value":7.0}
					""",
					StandardCharsets.UTF_8);
			return runsDir;
		} catch (Exception e) {
			throw new ExceptionInInitializerError(e);
		}
	}

	private static List<Double> decodeDoubles(String encoded) {
		final ByteBuffer buffer = ByteBuffer.wrap(Base64.getDecoder().decode(encoded))
				.order(ByteOrder.LITTLE_ENDIAN);
		final java.util.ArrayList<Double> values = new java.util.ArrayList<>();
		while (buffer.remaining() >= Double.BYTES) values.add(buffer.getDouble());
		return values;
	}

	private static List<Float> decodeFloats(String encoded) {
		final ByteBuffer buffer = ByteBuffer.wrap(Base64.getDecoder().decode(encoded))
				.order(ByteOrder.LITTLE_ENDIAN);
		final java.util.ArrayList<Float> values = new java.util.ArrayList<>();
		while (buffer.remaining() >= Float.BYTES) values.add(buffer.getFloat());
		return values;
	}
}
