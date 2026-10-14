package io.github.kazukin123.anetlab.metricsviewer.service;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;

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
import java.util.Map;
import java.util.LinkedHashMap;
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

@SpringBootTest
@AutoConfigureMockMvc
class MetricsLodIntegrationTest {

	private static final Path RUNS_DIR = createFixture();
	private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

	@Autowired
	private MockMvc mockMvc;

	@DynamicPropertySource
	static void configureRunsDir(DynamicPropertyRegistry registry) {
		registry.add("metricsviewer.workspaces-dir",
				() -> RUNS_DIR.getParent().getParent().toString());
		registry.add("metricsviewer.initial-workspace",
				() -> RUNS_DIR.getParent().getFileName().toString());
		registry.add("metricsviewer.cache-memory-mb", () -> "0");
	}

	@Test
	void stepBinsUseActualExtremaAndCommonCoordinates() throws Exception {
		awaitReady();
		final String request = """
				{"series":[{
				  "runId":"run-lod",
				  "tagKey":"loss",
				  "fromStep":0,
				  "toStep":31,
				  "maxPoints":6,
				  "stepBucketWidth":16
				}]}
				""";

		final String body = mockMvc.perform(post("/api/metrics.json")
						.contentType("application/json")
						.header("X-Query-Channel", "metrics-lod-test-full-range")
						.header("X-Query-Sequence", "0")
						.content(request))
				.andReturn()
				.getResponse()
				.getContentAsString(StandardCharsets.UTF_8);
		final JsonNode result = OBJECT_MAPPER.readTree(body).path("data").get(0);

		assertEquals("ok", result.path("availability").asText());
		assertEquals(6, result.path("pointBudget").asInt());
		assertEquals(16, result.path("stepBucketWidth").asLong());
		assertEquals("lod", result.path("projection").path("kind").asText());

		final JsonNode minMax = result.path("projection").path("minMax");
		assertEquals(List.of(3.0, 10.0, 15.0, 20.0, 25.0, 31.0),
				decodeDoubles(minMax.path("steps").get(0).asText()));
		assertEquals(List.of(-5.0f, 20.0f, 7.0f, -8.0f, 30.0f, 9.0f),
				decodeFloats(minMax.path("values").get(0).asText()));

		final JsonNode summary = result.path("projection").path("summary");
		assertEquals(List.of(0.0, 16.0),
				decodeDoubles(summary.path("steps").get(0).asText()));
		assertEquals(List.of(-5.0f, -8.0f),
				decodeFloats(summary.path("mins").get(0).asText()));
		assertEquals(List.of(20.0f, 30.0f),
				decodeFloats(summary.path("maxs").get(0).asText()));
		assertEquals(List.of(3.0, 20.0),
				decodeDoubles(summary.path("minSteps").get(0).asText()));
		assertEquals(List.of(10.0, 25.0),
				decodeDoubles(summary.path("maxSteps").get(0).asText()));
	}

	@Test
	void partialBoundaryBucketsContainOnlyTheInclusiveRequestedRange() throws Exception {
		awaitReady();
		final JsonNode result = query(5, 30, 6);

		assertEquals(16, result.path("stepBucketWidth").asLong());
		assertEquals(List.of(8.0, 10.0, 15.0, 20.0, 25.0, 30.0),
				decodeDoubles(result.path("projection").path("minMax").path("steps").get(0).asText()));
		assertEquals(List.of(0.0, 16.0),
				decodeDoubles(result.path("projection").path("summary").path("steps").get(0).asText()));
	}

	@Test
	void stepBinWiderThanTheStorageTopIsAggregatedExactly() throws Exception {
		awaitReady();
		final JsonNode result = query(0, 31, 3);

		assertEquals(32, result.path("stepBucketWidth").asLong());
		assertEquals(List.of(20.0, 25.0, 31.0),
				decodeDoubles(result.path("projection").path("minMax").path("steps").get(0).asText()));
		assertEquals(List.of(-8.0f, 30.0f, 9.0f),
				decodeFloats(result.path("projection").path("minMax").path("values").get(0).asText()));
	}

	@Test
	void differentFrequenciesDuplicatesAndLengthsUseExactCommonBinsForOneToFourRuns() throws Exception {
		awaitReady();
		for (int runCount = 1; runCount <= 4; runCount++) {
			final List<String> series = new ArrayList<>();
			for (int run=0;run<runCount;run++) series.add("""
					{"runId":"run-compare-%d","tagKey":"loss","fromStep":-11,"toStep":83,"maxPoints":24,"stepBucketWidth":17}
					""".formatted(run));
			final var response=mockMvc.perform(post("/api/metrics.json").contentType("application/json")
					.header("X-Query-Channel",UUID.randomUUID().toString()).header("X-Query-Sequence","0")
					.content("{\"series\":["+String.join(",",series)+"]}")).andReturn().getResponse();
			assertEquals(200,response.getStatus());
			final JsonNode results=OBJECT_MAPPER.readTree(response.getContentAsString()).path("data");
			for (int run=0;run<runCount;run++) {
				final JsonNode result=results.get(run);
				assertEquals(17,result.path("stepBucketWidth").asInt());
				final Map<Long,List<Integer>> expected=new LinkedHashMap<>();
				for (int i=0;i<300*(run+1);i++) {
					long step=compareStep(run,i);
					if(step>=-11 && step<=83) expected.computeIfAbsent(Math.floorDiv(step,17)*17,k->new ArrayList<>()).add(i%97-48);
				}
				final JsonNode summary=result.path("projection").path("summary");
				assertEquals(expected.keySet().stream().map(Long::doubleValue).toList(),decodeDoubles(summary.path("steps").get(0).asText()));
				final List<Float> mins=decodeFloats(summary.path("mins").get(0).asText());
				final List<Float> maxs=decodeFloats(summary.path("maxs").get(0).asText());
				final List<Float> means=decodeFloats(summary.path("means").get(0).asText());
				int bin=0;
				for (List<Integer> values:expected.values()) {
					assertEquals(values.stream().mapToInt(Integer::intValue).min().orElseThrow(),mins.get(bin));
					assertEquals(values.stream().mapToInt(Integer::intValue).max().orElseThrow(),maxs.get(bin));
					assertEquals(values.stream().mapToInt(Integer::intValue).average().orElseThrow(),means.get(bin),0.00001);
					bin++;
				}
			}
		}
	}

	private static long compareStep(int run,int ordinal) {
		return (ordinal/(run%3+1))*(run+1)-37L;
	}

	@Test
	void rawWidthOneKeepsAllDuplicateStepPointsWhenTheyFitTheExplicitCap() throws Exception {
		awaitReady();
		final var response=mockMvc.perform(post("/api/metrics.json").contentType("application/json")
				.header("X-Query-Channel",UUID.randomUUID().toString()).header("X-Query-Sequence","0")
				.content("{\"series\":[{\"runId\":\"run-lod\",\"tagKey\":\"duplicate\",\"fromStep\":2,\"toStep\":2,\"maxPoints\":8,\"stepBucketWidth\":1}]}"))
				.andReturn().getResponse();
		assertEquals(200,response.getStatus());
		final JsonNode result=OBJECT_MAPPER.readTree(response.getContentAsString()).path("data").get(0);
		assertEquals("raw",result.path("projection").path("kind").asText());
		assertEquals(8,decodeDoubles(result.path("projection").path("steps").get(0).asText()).size());
	}

	private JsonNode query(long fromStep, long toStep, int maxPoints) throws Exception {
		final String request = """
				{"series":[{
				  "runId":"run-lod",
				  "tagKey":"loss",
				  "fromStep":%d,
				  "toStep":%d,
				  "maxPoints":%d,
				  "stepBucketWidth":%d
				}]}
				""".formatted(fromStep, toStep, maxPoints, maxPoints == 3 ? 32 : 16);
		final String body = mockMvc.perform(post("/api/metrics.json")
						.contentType("application/json")
						.header("X-Query-Channel", UUID.randomUUID().toString())
						.header("X-Query-Sequence", "0")
						.content(request))
				.andReturn()
				.getResponse()
				.getContentAsString(StandardCharsets.UTF_8);
		return OBJECT_MAPPER.readTree(body).path("data").get(0);
	}

	private void awaitReady() throws Exception {
		final Instant deadline = Instant.now().plus(Duration.ofSeconds(30));
		while (Instant.now().isBefore(deadline)) {
			final String body = mockMvc.perform(get("/api/runs.json"))
					.andReturn()
					.getResponse()
					.getContentAsString(StandardCharsets.UTF_8);
			final JsonNode runs = OBJECT_MAPPER.readTree(body).path("runs");
			if (runs.size() == 5 && java.util.stream.StreamSupport.stream(runs.spliterator(), false)
					.allMatch(run -> "ready".equals(run.path("ingest").path("state").asText()))) {
				return;
			}
			Thread.sleep(50);
		}
		throw new AssertionError("LOD fixture did not become ready");
	}

	private static Path createFixture() {
		try {
			final Path runsDir = Path.of(
					"target", "metrics-lod-integration-" + System.nanoTime(), "_test", "runs")
					.toAbsolutePath()
					.normalize();
			final Path runDir = runsDir.resolve("run-lod");
			Files.createDirectories(runDir);
			final StringBuilder jsonl = new StringBuilder();
			for (int step = 0; step < 32; step++) {
				double value = step % 8;
				if (step == 3) value = -5.0;
				if (step == 10) value = 20.0;
				if (step == 15) value = 7.0;
				if (step == 20) value = -8.0;
				if (step == 25) value = 30.0;
				if (step == 31) value = 9.0;
				jsonl.append("{\"type\":\"scalar\",\"tag\":\"loss\",\"step\":")
						.append(step)
						.append(",\"value\":")
						.append(value)
						.append("}\n");
			}
			for(int i=0;i<8;i++) jsonl.append("{\"type\":\"scalar\",\"tag\":\"duplicate\",\"step\":2,\"value\":").append(i).append("}\n");
			Files.writeString(runDir.resolve("metrics.jsonl"), jsonl, StandardCharsets.UTF_8);
			for (int run=0;run<4;run++) {
				final Path dir=runsDir.resolve("run-compare-"+run);
				Files.createDirectories(dir);
				final StringBuilder data=new StringBuilder();
				for(int i=0;i<300*(run+1);i++) data.append("{\"type\":\"scalar\",\"tag\":\"loss\",\"step\":")
					.append(compareStep(run,i)).append(",\"value\":").append(i%97-48).append("}\n");
				Files.writeString(dir.resolve("metrics.jsonl"),data,StandardCharsets.UTF_8);
			}
			return runsDir;
		} catch (Exception e) {
			throw new ExceptionInInitializerError(e);
		}
	}

	private static List<Double> decodeDoubles(String encoded) {
		final ByteBuffer buffer = ByteBuffer.wrap(Base64.getDecoder().decode(encoded))
				.order(ByteOrder.LITTLE_ENDIAN);
		final List<Double> values = new ArrayList<>();
		while (buffer.remaining() >= Double.BYTES) values.add(buffer.getDouble());
		return values;
	}

	private static List<Float> decodeFloats(String encoded) {
		final ByteBuffer buffer = ByteBuffer.wrap(Base64.getDecoder().decode(encoded))
				.order(ByteOrder.LITTLE_ENDIAN);
		final List<Float> values = new ArrayList<>();
		while (buffer.remaining() >= Float.BYTES) values.add(buffer.getFloat());
		return values;
	}
}
