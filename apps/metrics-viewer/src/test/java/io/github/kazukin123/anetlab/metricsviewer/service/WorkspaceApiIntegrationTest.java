package io.github.kazukin123.anetlab.metricsviewer.service;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
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

@SpringBootTest
@AutoConfigureMockMvc
class WorkspaceApiIntegrationTest {

	private static final Path WORKSPACES_DIR = createFixture();
	private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

	@Autowired
	private MockMvc mockMvc;

	@Autowired
	private WorkspaceManager workspaceManager;

	@MockitoBean
	private LoadingThread loadingThread;

	@DynamicPropertySource
	static void configure(DynamicPropertyRegistry registry) {
		registry.add("metricsviewer.workspaces-dir", () -> WORKSPACES_DIR.toString());
		registry.add("metricsviewer.initial-workspace", () -> "ws-a");
		registry.add("metricsviewer.cache-memory-mb", () -> "0");
	}

	@Test
	void workspaceApiListsAndSwitchesTheRunWorkingSet() throws Exception {
		final JsonNode workspaces = readJson(mockMvc.perform(get("/api/workspaces.json"))
				.andExpect(status().isOk())
				.andReturn()
				.getResponse()
				.getContentAsString(StandardCharsets.UTF_8));
		assertEquals("ws-a", workspaces.path("current").asText());
		assertEquals(List.of("config-only", "ws-a", "ws-b"),
				strings(workspaces.path("workspaces")));
		assertEquals(List.of("run-a"), runIds());

		mockMvc.perform(post("/api/workspace")
				.contentType("application/json")
				.content("{\"name\":\"ws-b\"}"))
				.andExpect(status().isNoContent());

		assertEquals(List.of("run-b"), runIds());
	}

	@Test
	void switchWorkspaceUsesClosedRequestSchemaAndNoOpContract() throws Exception {
		final String current = workspaceManager.currentName();
		final long epoch = workspaceManager.currentEpoch();

		mockMvc.perform(post("/api/workspace")
				.contentType("application/json")
				.content("{\"name\":\"" + current + "\",\"extra\":true}"))
				.andExpect(status().isBadRequest())
				.andExpect(jsonPath("$.code").value("invalid_request"));

		mockMvc.perform(post("/api/workspace")
				.contentType("application/json")
				.content("{\"name\":\"../outside\"}"))
				.andExpect(status().isNotFound())
				.andExpect(jsonPath("$.code").value("unknown_workspace"));

		mockMvc.perform(post("/api/workspace")
				.contentType("application/json")
				.content("{\"name\":123}"))
				.andExpect(status().isBadRequest())
				.andExpect(jsonPath("$.code").value("invalid_request"));

		mockMvc.perform(post("/api/workspace")
				.contentType("application/json")
				.content("{\"name\":\"" + current + "\"}"))
				.andExpect(status().isNoContent());
		assertEquals(epoch, workspaceManager.currentEpoch());
	}

	@Test
	void unreadableBodyHandlerIsScopedToTheWorkspaceController() throws Exception {
		mockMvc.perform(post("/api/workspace")
				.contentType("application/json")
				.content("{"))
				.andExpect(status().isBadRequest())
				.andExpect(jsonPath("$.code").value("invalid_request"));

		final String metricsBody = mockMvc.perform(post("/api/metrics.json")
						.contentType("application/json")
						.header("X-Query-Channel", "workspace-api-test")
						.header("X-Query-Sequence", "0")
				.content("{"))
				.andExpect(status().isBadRequest())
				.andReturn()
				.getResponse()
				.getContentAsString(StandardCharsets.UTF_8);
		final String prioritizeBody = mockMvc.perform(post("/api/runs/prioritize")
				.contentType("application/json")
				.content("{"))
				.andExpect(status().isBadRequest())
				.andReturn()
				.getResponse()
				.getContentAsString(StandardCharsets.UTF_8);

		assertFalse(metricsBody.contains("invalid_request"));
		assertFalse(prioritizeBody.contains("invalid_request"));
	}

	private List<String> runIds() throws Exception {
		final JsonNode runs = readJson(mockMvc.perform(get("/api/runs.json"))
				.andExpect(status().isOk())
				.andReturn()
				.getResponse()
				.getContentAsString(StandardCharsets.UTF_8));
		final List<String> ids = new ArrayList<>();
		for (JsonNode run : runs.path("runs")) ids.add(run.path("id").asText());
		return ids;
	}

	private static JsonNode readJson(String body) throws Exception {
		return OBJECT_MAPPER.readTree(body);
	}

	private static List<String> strings(JsonNode values) {
		final List<String> result = new ArrayList<>();
		for (JsonNode value : values) result.add(value.asText());
		return result;
	}

	private static Path createFixture() {
		try {
			final Path root = Files.createTempDirectory("metrics-viewer-workspaces-");
			createRun(root, "ws-a", "run-a");
			createRun(root, "ws-b", "run-b");
			Files.createDirectories(root.resolve("config-only").resolve("config"));
			Files.createDirectories(root.resolve("ignored"));
			return root;
		} catch (Exception e) {
			throw new ExceptionInInitializerError(e);
		}
	}

	private static void createRun(Path root, String workspace, String runId) throws Exception {
		final Path runDir = root.resolve(workspace).resolve("runs").resolve(runId);
		Files.createDirectories(runDir);
		Files.writeString(
				runDir.resolve("metrics.jsonl"),
				"{\"type\":\"scalar\",\"tag\":\"loss\",\"step\":0,\"value\":1.0}\n",
				StandardCharsets.UTF_8);
	}
}
