package io.github.kazukin123.anetlab.metricsviewer.service;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ResourceLoader;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import io.github.kazukin123.anetlab.metricsviewer.view.model.ApiError;
import io.github.kazukin123.anetlab.metricsviewer.view.model.GetTagSetsResponse;
import io.github.kazukin123.anetlab.metricsviewer.view.model.TagSetInfo;

/** タグセットの永続化とプロセス内の一覧を所有する。 */
@Service
public class TagSetStore {
	private static final Logger LOG = LoggerFactory.getLogger(TagSetStore.class);
	private final Path file;
	private final ObjectMapper mapper = new ObjectMapper()
			.enable(JsonParser.Feature.STRICT_DUPLICATE_DETECTION)
			.enable(DeserializationFeature.FAIL_ON_TRAILING_TOKENS);
	private Map<String, List<String>> tagSets;

	public TagSetStore(@Value("${metricsviewer.tag-sets-file}") String configuredPath,
			ResourceLoader resources) {
		try {
			file = Path.of(configuredPath).toAbsolutePath().normalize();
			if (Files.isDirectory(file)) throw new IOException("path is a directory");
			Files.createDirectories(file.getParent());
		} catch (IOException | java.nio.file.InvalidPathException exception) {
			throw new IllegalArgumentException("Invalid configuration: key=metricsviewer.tag-sets-file value="
					+ configuredPath + " expected=a file with a creatable parent directory", exception);
		}
		try {
			// 初期値と作業用は同じ形式で読み、作業用が無いときだけ写す。
			Map<String, List<String>> initial;
			try (var input = resources.getResource("classpath:initial_tag_sets.json").getInputStream()) {
				initial = read(input.readAllBytes(), "initial_tag_sets.json");
			} catch (IOException exception) {
				throw new IllegalArgumentException("Cannot read initial_tag_sets.json: " + exception.getMessage(), exception);
			}
			if (Files.exists(file)) {
				tagSets = read(Files.readAllBytes(file), file.toString());
			} else {
				write(initial);
				tagSets = initial;
				LOG.info("Created tag sets file {} from initial_tag_sets.json", file);
			}
		} catch (IOException exception) {
			throw new IllegalStateException("Cannot initialize tag sets file " + file, exception);
		}
	}

	private Map<String, List<String>> read(byte[] bytes, String source) {
		try {
			// treeで型を確かめ、文字列への暗黙変換や重複キーの上書きを許さない。
			JsonNode root = mapper.readTree(bytes);
			if (root == null || !root.isObject()) throw new IllegalArgumentException("expected a JSON object");
			var result = new TreeMap<String, List<String>>();
			var fields = root.fields();
			while (fields.hasNext()) {
				var entry = fields.next();
				String name = normalizeName(entry.getKey());
				if (!name.equals(entry.getKey())) throw new IllegalArgumentException("name has surrounding whitespace: " + entry.getKey());
				if (!entry.getValue().isArray()) throw new IllegalArgumentException("expected a tag array for name=" + name);
				var tags = new java.util.ArrayList<String>();
				int index = 0;
				for (JsonNode tag : entry.getValue()) {
					if (!tag.isTextual()) throw new IllegalArgumentException("expected string: name=" + name + " index=" + index);
					tags.add(tag.textValue());
					index++;
				}
				result.put(name, validateTags(tags));
			}
			return result;
		} catch (IOException | IllegalArgumentException exception) {
			throw new IllegalArgumentException("Invalid tag sets file " + source + ": " + exception.getMessage(), exception);
		}
	}

	private static String normalizeName(String name) {
		if (name == null) throw new IllegalArgumentException("name must be a string");
		// JavaのstripとJSのtrimは範囲が異なるため、Unicode White_Spaceで揃える。
		String trimmed = name.replaceAll("^\\p{IsWhite_Space}+|\\p{IsWhite_Space}+$", "");
		if (trimmed.isEmpty() || trimmed.codePointCount(0, trimmed.length()) > 64
				|| trimmed.codePoints().anyMatch(cp -> Character.getType(cp) == Character.CONTROL)) {
			throw new IllegalArgumentException("name=" + name + " expected=1..64 code points without control characters");
		}
		return trimmed;
	}

	private static List<String> validateTags(List<String> tags) {
		if (tags == null) throw new IllegalArgumentException("tags must be a string array");
		var unique = new java.util.TreeSet<String>();
		for (int index = 0; index < tags.size(); index++) {
			String tag = tags.get(index);
			if (tag == null || tag.isEmpty()
					|| tag.codePoints().anyMatch(cp -> Character.getType(cp) == Character.CONTROL)
					|| !unique.add(tag)) {
				throw new IllegalArgumentException("tags[" + index + "]=" + tag
						+ " expected=a nonempty unique string without control characters");
			}
		}
		return List.copyOf(unique);
	}

	public synchronized GetTagSetsResponse getTagSets() {
		return response(tagSets);
	}

	public synchronized GetTagSetsResponse create(String name, List<String> tags) {
		// 名前とtagの規則を先に検証し、衝突よりも入力違反を優先する。
		name = normalizeName(name);
		tags = validateTags(tags);
		if (tagSets.containsKey(name)) throw apiError(HttpStatus.CONFLICT, "tag_set_exists", "Tag set already exists: " + name);
		// 候補を永続化してから公開し、書けなければ手元の一覧を維持する。
		var next = new TreeMap<>(tagSets);
		next.put(name, tags);
		return commit(next);
	}

	public synchronized GetTagSetsResponse update(String name, List<String> tags) {
		// 存在確認より先に入力を検証し、更新できる候補だけを作る。
		name = normalizeName(name);
		tags = validateTags(tags);
		requireExisting(name);
		var next = new TreeMap<>(tagSets);
		next.put(name, tags);
		return commit(next);
	}

	public synchronized GetTagSetsResponse rename(String name, String newName) {
		// 両方の名前、参照元、同名no-op、衝突の順で判定する。
		name = normalizeName(name);
		newName = normalizeName(newName);
		requireExisting(name);
		if (name.equals(newName)) return response(tagSets);
		if (tagSets.containsKey(newName)) throw apiError(HttpStatus.CONFLICT, "tag_set_exists", "Tag set already exists: " + newName);
		var next = new TreeMap<>(tagSets);
		next.put(newName, next.remove(name));
		return commit(next);
	}

	public synchronized GetTagSetsResponse delete(String name) {
		// 不正な名前と不在を区別してから、削除候補を永続化する。
		name = normalizeName(name);
		requireExisting(name);
		var next = new TreeMap<>(tagSets);
		next.remove(name);
		return commit(next);
	}

	private GetTagSetsResponse commit(Map<String, List<String>> next) {
		// 永続化と一覧の公開を同じ同期境界の中で行う。
		try {
			write(next);
		} catch (IOException exception) {
			throw new MetricsApiException(HttpStatus.INTERNAL_SERVER_ERROR,
					new ApiError("tag_sets_write_failed", "Cannot write tag sets file " + file));
		}
		tagSets = next;
		return response(tagSets);
	}

	private void requireExisting(String name) {
		if (!tagSets.containsKey(name)) throw apiError(HttpStatus.NOT_FOUND, "unknown_tag_set", "Unknown tag set: " + name);
	}

	private static MetricsApiException apiError(HttpStatus status, String code, String message) {
		return new MetricsApiException(status, new ApiError(code, message));
	}

	private GetTagSetsResponse response(Map<String, List<String>> values) {
		return new GetTagSetsResponse(new TreeMap<>(values).entrySet().stream()
				.map(entry -> new TagSetInfo(entry.getKey(), entry.getValue())).toList());
	}

	private void write(Map<String, List<String>> values) throws IOException {
		// 整列と改行を固定し、同じディレクトリの一時ファイルから置き換える。
		var entries = new java.util.ArrayList<String>();
		for (var entry : new TreeMap<>(values).entrySet()) {
			var tags = new java.util.ArrayList<String>();
			for (String tag : entry.getValue().stream().sorted().toList()) {
				tags.add("    " + mapper.writeValueAsString(tag));
			}
			entries.add("  " + mapper.writeValueAsString(entry.getKey()) + ": "
					+ (tags.isEmpty() ? "[]" : "[\n" + String.join(",\n", tags) + "\n  ]"));
		}
		String json = entries.isEmpty() ? "{}\n" : "{\n" + String.join(",\n", entries) + "\n}\n";
		Path temporary = Files.createTempFile(file.getParent(), ".tag-sets-", ".tmp");
		try {
			Files.writeString(temporary, json, StandardCharsets.UTF_8);
			Files.move(temporary, file, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
		} finally {
			Files.deleteIfExists(temporary);
		}
	}
}
