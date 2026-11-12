package io.github.kazukin123.anetlab.metricsviewer.service;

import static org.junit.jupiter.api.Assertions.*;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.stream.Stream;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.core.io.DefaultResourceLoader;
import org.springframework.core.io.Resource;

class TagSetStoreTest {
	@TempDir Path directory;

	static DefaultResourceLoader initial(String json) {
		return new DefaultResourceLoader() {
			@Override public Resource getResource(String location) {
				return new ByteArrayResource(json.getBytes(StandardCharsets.UTF_8), location);
			}
		};
	}

	static Stream<String> malformedFiles() {
		return Stream.of("{", "[]", "null", "{} {}", "{\"A\":[],\"A\":[]}",
				"{\" A\":[]}", "{\"\":[]}", "{\"A\":[1]}", "{\"A\":null}",
				"{\"A\":[\"x\",\"x\"]}", "{\"A\":[\"\"]}", "{\"A\":[\"x\\ty\"]}",
				"{\"A\\tB\":[]}", "{\"" + "a".repeat(65) + "\":[]}");
	}

	@Test
	void createsCanonicalBytesAndPreservesAnExistingWorkingFile() throws Exception {
		// 順序・改行・空集合を公開された保存ファイルのバイト列で確かめる。
		Path file = directory.resolve("new/sets.json");
		var store = new TagSetStore(file.toString(), initial("{\"B\":[\"z\",\"a\"],\"A\":[]}"));
		assertEquals("{\n  \"A\": [],\n  \"B\": [\n    \"a\",\n    \"z\"\n  ]\n}\n", Files.readString(file));
		assertEquals(List.of("A", "B"), store.getTagSets().tagSets().stream().map(info -> info.name()).toList());
		new TagSetStore(file.toString(), initial("{\"different\":[]}"));
		assertEquals(store.getTagSets(), new TagSetStore(file.toString(), initial("{}")).getTagSets());
	}

	@Test
	void failedWriteKeepsMemoryAndExternalEditsAreNotReloaded() throws Exception {
		Path file = directory.resolve("sets.json");
		var store = new TagSetStore(file.toString(), initial("{\"A\":[]}"));
		// 外部の変更はGETへ反映せず、次の保存でメモリ上の一覧から上書きする。
		Files.writeString(file, "{\"external\":[]}");
		store.create("B", List.of());
		assertFalse(Files.readString(file).contains("external"));
		var before = store.getTagSets();
		Files.delete(file);
		Files.createDirectory(file);
		Files.writeString(file.resolve("block"), "prevent replacement");
		assertThrows(MetricsApiException.class, () -> store.create("C", List.of()));
		assertEquals(before, store.getTagSets());
	}

	@Test
	void invalidPathsAndMissingResourceStopInitialization() throws Exception {
		// ファイルの代わりにディレクトリを指定すると、設定キーを含む診断で止まる。
		assertTrue(assertThrows(IllegalArgumentException.class,
				() -> new TagSetStore(directory.toString(), initial("{}"))).getMessage().contains("metricsviewer.tag-sets-file"));
		// 親が通常ファイルの場合も、子の保存先を作れない時点で起動を止める。
		Path parentFile = directory.resolve("not-directory");
		Files.writeString(parentFile, "x");
		assertThrows(IllegalArgumentException.class, () -> new TagSetStore(parentFile.resolve("sets.json").toString(), initial("{}")));
		// 保存先が正しくても初期値が無ければ、リソース名を含む診断で止まる。
		var missing = new DefaultResourceLoader() {
			@Override public Resource getResource(String location) {
				return new org.springframework.core.io.FileSystemResource(directory.resolve("absent"));
			}
		};
		assertTrue(assertThrows(IllegalArgumentException.class,
				() -> new TagSetStore(directory.resolve("sets.json").toString(), missing)).getMessage().contains("initial_tag_sets.json"));
	}

	@Test
	void initialCopyFailureStopsStartupAndIncludesTheWorkingPath() throws Exception {
		Path parent = directory.resolve("parent");
		Path file = parent.resolve("sets.json");
		var resources = new DefaultResourceLoader() {
			@Override public Resource getResource(String location) {
				// パス検証後に親が失われた状況で、初期値の実書き込みを失敗させる。
				try {
					Files.delete(parent);
					Files.writeString(parent, "not a directory");
				} catch (java.io.IOException exception) { throw new IllegalStateException(exception); }
				return new ByteArrayResource("{}".getBytes(StandardCharsets.UTF_8));
			}
		};
		RuntimeException exception = assertThrows(RuntimeException.class,
				() -> new TagSetStore(file.toString(), resources));
		assertTrue(exception.getMessage().contains(file.toString()));
	}

	@ParameterizedTest
	@MethodSource("malformedFiles")
	void startupRejectsMalformedWorkingAndInitialFiles(String json) throws Exception {
		// どちらの入力でも同じ規則を適用し、診断に入力元を残す。
		Path file = directory.resolve("sets.json");
		Files.writeString(file, json);
		RuntimeException working = assertThrows(RuntimeException.class,
				() -> new TagSetStore(file.toString(), initial("{}")));
		assertTrue(working.getMessage().contains(file.toString()), working.getMessage());
		Files.writeString(file, "{}");
		RuntimeException bundled = assertThrows(RuntimeException.class,
				() -> new TagSetStore(file.toString(), initial(json)));
		assertTrue(bundled.getMessage().contains("initial_tag_sets.json"), bundled.getMessage());
	}
}
