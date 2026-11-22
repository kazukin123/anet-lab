package io.github.kazukin123.anetlab.metricsviewer.service;

import java.io.BufferedInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.regex.Pattern;

/** C++ の Properties::Load と同じ行規則で、Run の設定を読む。 */
public final class ConfigTextReader {

	private static final Pattern KEY_WHITESPACE = Pattern.compile("[ \\t\\n\\x0B\\f\\r]");
	private static final Pattern INCLUDE = Pattern.compile("^\\$[ \\t]*include", Pattern.CASE_INSENSITIVE);
	private static final String TRIM_WHITESPACE = " \t\r\n";

	private ConfigTextReader() {
	}

	public static Map<String, String> read(Path path) throws IOException {
		final Map<String, String> values = new TreeMap<>();
		final Set<String> defaults = new HashSet<>();
		try (InputStream input = new BufferedInputStream(Files.newInputStream(path))) {
			// LF だけで分けてから UTF-8 を検証し、読み取れない行の番号を特定する。
			final ByteArrayOutputStream bytes = new ByteArrayOutputStream();
			int lineNumber = 1;
			for (int next; (next = input.read()) != -1;) {
				if (next == '\n') {
					readLine(path, bytes, lineNumber++, values, defaults);
					bytes.reset();
				} else {
					bytes.write(next);
				}
			}
			if (bytes.size() > 0) readLine(path, bytes, lineNumber, values, defaults);
		}
		return values;
	}

	private static void readLine(Path path, ByteArrayOutputStream bytes, int lineNumber,
			Map<String, String> values, Set<String> defaults) throws FormatException {
		final String source;
		try {
			source = StandardCharsets.UTF_8.newDecoder()
					.decode(ByteBuffer.wrap(bytes.toByteArray())).toString();
		} catch (CharacterCodingException e) {
			throw new FormatException(path, lineNumber, "R1", "Invalid UTF-8");
		}

		// ファイル冒頭の BOM とコメントを取り除いてから、ディレクティブと代入を判定する。
		String line = lineNumber == 1 && source.startsWith("\uFEFF") ? source.substring(1) : source;
		if (line.isEmpty() || line.charAt(0) == '#') return;
		line = trim(before(before(line, "#"), "//"));
		if (line.isEmpty()) return;
		if (INCLUDE.matcher(line).find()) {
			throw new FormatException(path, lineNumber, "R5", source);
		}
		final int assignment = line.indexOf('=');
		if (assignment < 0) return;

		// "?=" は連続しているときだけ演算子として取り除く。
		final boolean isDefault = assignment > 0 && line.charAt(assignment - 1) == '?';
		String key = KEY_WHITESPACE.matcher(line.substring(0, assignment - (isDefault ? 1 : 0)))
				.replaceAll("");
		if (key.indexOf('?') >= 0) {
			throw new FormatException(path, lineNumber, "R8", source);
		}
		final int colon = key.indexOf(':');
		if (colon >= 0) {
			if (colon == 0 || colon == key.length() - 1 || colon != key.lastIndexOf(':')) {
				throw new FormatException(path, lineNumber, "R9", source);
			}
			key = key.replace(':', '.');
		}
		if (isDefault && (key.endsWith(".$") || key.equals("$") || key.startsWith("run.@"))) {
			throw new FormatException(path, lineNumber, "R10", source);
		}

		// 個別指定は既定キーに優先し、同じ強さの宣言では後の値を使う。
		if (key.isEmpty() || (isDefault && values.containsKey(key) && !defaults.contains(key))) return;
		final String value = trim(line.substring(assignment + 1));
		int end = value.length();
		while (end > 0 && value.charAt(end - 1) == ';') end--;
		values.put(key, trim(value.substring(0, end)));
		if (isDefault) defaults.add(key);
		else defaults.remove(key);
	}

	private static String before(String text, String delimiter) {
		final int index = text.indexOf(delimiter);
		return index < 0 ? text : text.substring(0, index);
	}

	private static String trim(String text) {
		int begin = 0;
		int end = text.length();
		while (begin < end && TRIM_WHITESPACE.indexOf(text.charAt(begin)) >= 0) begin++;
		while (begin < end && TRIM_WHITESPACE.indexOf(text.charAt(end - 1)) >= 0) end--;
		return text.substring(begin, end);
	}

	static final class FormatException extends IOException {
		FormatException(Path path, int lineNumber, String rule, String source) {
			super(path + ":" + lineNumber + " (" + rule + "): " + source);
		}
	}
}
