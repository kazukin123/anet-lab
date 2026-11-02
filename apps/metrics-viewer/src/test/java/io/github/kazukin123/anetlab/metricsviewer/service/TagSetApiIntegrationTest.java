package io.github.kazukin123.anetlab.metricsviewer.service;

import static org.junit.jupiter.api.Assertions.*;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
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
class TagSetApiIntegrationTest {
	private static final Path WORKSPACES = createWorkspaces();
	@Autowired MockMvc mvc;
	private final ObjectMapper mapper = new ObjectMapper();

	private static Path createWorkspaces() {
		try { return Files.createTempDirectory("prd111-api-"); }
		catch (java.io.IOException exception) { throw new IllegalStateException(exception); }
	}

	@DynamicPropertySource
	static void properties(DynamicPropertyRegistry registry) {
		registry.add("metricsviewer.workspaces-dir", () -> WORKSPACES.toString());
		registry.add("metricsviewer.cache-memory-mb", () -> "0");
	}

	private JsonNode postJson(String operation, String body, int expectedStatus) throws Exception {
		var response = mvc.perform(post("/api/tag-sets/" + operation).contentType("application/json").content(body))
				.andReturn().getResponse();
		assertEquals(expectedStatus, response.getStatus(), response.getContentAsString());
		return mapper.readTree(response.getContentAsString(StandardCharsets.UTF_8));
	}

	@BeforeEach
	void clearSets() throws Exception {
		var response = mvc.perform(get("/api/tag-sets.json")).andReturn().getResponse();
		for (JsonNode info : mapper.readTree(response.getContentAsString(StandardCharsets.UTF_8)).path("tagSets")) {
			postJson("delete", mapper.writeValueAsString(java.util.Map.of("name", info.path("name").asText())), 200);
		}
	}

	@Test
	void crudNormalizesNamesAndReturnsTheWholeSortedList() throws Exception {
		// 名前はUnicode White_Spaceを除き、tagの順序は保存順に依存しない。
		assertTrue(Files.exists(WORKSPACES.resolve("metrics_viewer_tag_sets.json")));
		JsonNode created = postJson("create", "{\"name\":\"　B\\t\",\"tags\":[\"z\",\"a\"]}", 200);
		assertEquals("B", created.path("tagSets").get(0).path("name").asText());
		assertEquals("a", created.path("tagSets").get(0).path("tags").get(0).asText());
		postJson("create", "{\"name\":\"A\",\"tags\":[]}", 200);
		postJson("update", "{\"name\":\"B\",\"tags\":[\"new\"]}", 200);
		postJson("rename", "{\"name\":\"B\",\"newName\":\" C \"}", 200);
		byte[] before = Files.readAllBytes(WORKSPACES.resolve("metrics_viewer_tag_sets.json"));
		postJson("rename", "{\"name\":\"C\",\"newName\":\"\\tC　\"}", 200);
		assertArrayEquals(before, Files.readAllBytes(WORKSPACES.resolve("metrics_viewer_tag_sets.json")));
		JsonNode remaining = postJson("delete", "{\"name\":\"A\"}", 200).path("tagSets");
		assertEquals(1, remaining.size());
		assertEquals("C", remaining.get(0).path("name").asText());
		assertEquals("new", remaining.get(0).path("tags").get(0).asText());
	}

	@Test
	void invalidRequestsUseClosedSchemaAndTheSpecifiedErrorOrder() throws Exception {
		postJson("create", "{\"name\":\"PER\",\"tags\":[]}", 200);
		// 形→名前→tag→存在→衝突の順に、重なる失敗も同じ応答へ収束させる。
		String[][] cases = {
			{"create", "{", "400", "invalid_request"},
			{"create", "{\"name\":\"X\",\"tags\":[]} {}", "400", "invalid_request"},
			{"create", "null", "400", "invalid_request"},
			{"create", "[]", "400", "invalid_request"},
			{"create", "{\"name\":\"X\"}", "400", "invalid_request"},
			{"rename", "{\"name\":\"PER\"}", "400", "invalid_request"},
			{"create", "{\"name\":\"X\",\"tags\":[],\"extra\":true}", "400", "invalid_request"},
			{"create", "{\"name\":\"X\",\"tags\":[],\"unknownFields\":true}", "400", "invalid_request"},
			{"create", "{\"name\":12,\"tags\":[]}", "400", "invalid_request"},
			{"create", "{\"name\":\"X\",\"tags\":\"tag\"}", "400", "invalid_request"},
			{"create", "{\"name\":\"X\",\"tags\":[12]}", "400", "invalid_request"},
			{"create", "{\"name\":\"X\",\"tags\":[null]}", "400", "invalid_request"},
			{"create", "{\"name\":\"PER\",\"tags\":[\"x\\ty\"]}", "400", "invalid_request"},
			{"update", "{\"name\":\"missing\",\"tags\":[\"x\",\"x\"]}", "400", "invalid_request"},
			{"delete", "{\"name\":\"   \"}", "400", "invalid_request"},
			{"delete", "{\"name\":\"P\\tER\"}", "400", "invalid_request"},
			{"delete", "{\"name\":\"" + "x".repeat(65) + "\"}", "400", "invalid_request"},
			{"create", "{\"name\":\"X\",\"tags\":[\"\"]}", "400", "invalid_request"},
			{"update", "{\"name\":\"missing\",\"tags\":[]}", "404", "unknown_tag_set"},
			{"delete", "{\"name\":\"missing\"}", "404", "unknown_tag_set"},
			{"rename", "{\"name\":\"missing\",\"newName\":\"PER\"}", "404", "unknown_tag_set"},
			{"rename", "{\"name\":\"missing\",\"newName\":\"missing\"}", "404", "unknown_tag_set"},
			{"create", "{\"name\":\"\\tPER　\",\"tags\":[]}", "409", "tag_set_exists"}
		};
		for (String[] entry : cases) {
			assertEquals(entry[3], postJson(entry[0], entry[1], Integer.parseInt(entry[2])).path("code").asText());
		}
		postJson("create", "{\"name\":\"other\",\"tags\":[]}", 200);
		assertEquals("tag_set_exists", postJson("rename", "{\"name\":\"other\",\"newName\":\"PER\"}", 409).path("code").asText());
	}

	@Test
	void unicodeWhitespaceCodePointLimitAndUtf16OrderingAreConsistent() throws Exception {
		String[] whitespaces = {"\t", "\n", "\u000b", "\f", "\r", " ", "\u0085", "\u00a0", "\u1680",
				"\u2000", "\u2001", "\u2002", "\u2003", "\u2004", "\u2005", "\u2006", "\u2007",
				"\u2008", "\u2009", "\u200a", "\u2028", "\u2029", "\u202f", "\u205f", "\u3000"};
		for (int index = 0; index < whitespaces.length; index++) {
			String name = "space" + index;
			postJson("create", mapper.writeValueAsString(java.util.Map.of(
					"name", whitespaces[index] + name + whitespaces[index], "tags", java.util.List.of())), 200);
		}
		String emoji = "\ud83d\ude00";
		postJson("create", mapper.writeValueAsString(java.util.Map.of("name", emoji.repeat(64), "tags", java.util.List.of())), 200);
		assertEquals("invalid_request", postJson("create", mapper.writeValueAsString(java.util.Map.of(
				"name", emoji.repeat(65), "tags", java.util.List.of())), 400).path("code").asText());
		for (String name : java.util.List.of("可塑性", "eval 成績", "PER", "Atari スコア", "per", "\ufeffX", "\ue000")) {
			postJson("create", mapper.writeValueAsString(java.util.Map.of("name", name, "tags", java.util.List.of("z", emoji, "\ue000"))), 200);
		}
		JsonNode all = mapper.readTree(mvc.perform(get("/api/tag-sets.json")).andReturn().getResponse()
				.getContentAsString(StandardCharsets.UTF_8));
		var names = new java.util.ArrayList<String>();
		for (JsonNode info : all.path("tagSets")) names.add(info.path("name").asText());
		assertEquals(names.stream().sorted().toList(), names);
		assertTrue(names.indexOf(emoji.repeat(64)) < names.indexOf("\ue000"));
		assertTrue(names.contains("\ufeffX"));
		assertTrue(names.containsAll(java.util.List.of("PER", "per")));
	}

	@Test
	void writeFailureReturns500AndLeavesGetUnchanged() throws Exception {
		postJson("create", "{\"name\":\"A\",\"tags\":[]}", 200);
		Path file = WORKSPACES.resolve("metrics_viewer_tag_sets.json");
		byte[] before = Files.readAllBytes(file);
		// OSの権限設定に依存せず、置換不能な実ディレクトリを一時的に用意する。
		Files.delete(file);
		Files.createDirectory(file);
		Path block = file.resolve("block");
		Files.writeString(block, "block");
		try {
			JsonNode error = postJson("create", "{\"name\":\"B\",\"tags\":[]}", 500);
			assertEquals("tag_sets_write_failed", error.path("code").asText());
			assertTrue(error.path("message").asText().contains(file.toString()));
			JsonNode result = mapper.readTree(mvc.perform(get("/api/tag-sets.json")).andReturn().getResponse()
					.getContentAsString(StandardCharsets.UTF_8));
			assertEquals(1, result.path("tagSets").size());
			assertEquals("A", result.path("tagSets").get(0).path("name").asText());
		} finally {
			Files.delete(block);
			Files.delete(file);
			Files.write(file, before);
		}
	}

	@Test
	void concurrentChangesToDifferentSetsBothSurvive() throws Exception {
		// 別々のセットへの同時要求を用意し、同じStoreの直列化境界を通す。
		var executor = java.util.concurrent.Executors.newFixedThreadPool(2);
		try {
			var first = executor.submit(() -> postJson("create", "{\"name\":\"A\",\"tags\":[\"a\"]}", 200));
			var second = executor.submit(() -> postJson("create", "{\"name\":\"B\",\"tags\":[\"b\"]}", 200));
			// 両方の完了後にGETし、一方の保存で他方が失われていないことを確かめる。
			first.get();
			second.get();
			JsonNode result = mapper.readTree(mvc.perform(get("/api/tag-sets.json")).andReturn().getResponse()
					.getContentAsString(StandardCharsets.UTF_8));
			assertEquals(2, result.path("tagSets").size());
			// 同じセットの更新は順に確定させ、最後の要求の内容が残ることも確かめる。
			postJson("update", "{\"name\":\"A\",\"tags\":[\"first\"]}", 200);
			postJson("update", "{\"name\":\"A\",\"tags\":[\"last\"]}", 200);
			result = mapper.readTree(mvc.perform(get("/api/tag-sets.json")).andReturn().getResponse()
					.getContentAsString(StandardCharsets.UTF_8));
			assertEquals("last", result.path("tagSets").get(0).path("tags").get(0).asText());
		} finally { executor.shutdownNow(); }
	}
}
