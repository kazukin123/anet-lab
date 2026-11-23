package io.github.kazukin123.anetlab.metricsviewer.service;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

class ConfigTextReaderTest {

	@TempDir
	Path directory;

	@Test
	void appliesFrameworkLineRulesToValuesAndDefaults() throws Exception {
		final Path source = directory.resolve("config_data.txt");
		Files.writeString(source, "\uFEFF# comment\r\n\r\n"
				+ " \t // comment\r\nline without assignment\n"
				+ " = ignored\n"
				+ " fa m\t\fil\u000By : a\u3000b = C:\\path=with-equals;;; # comment // ignored\r\n"
				+ "name?= first\nname?= second\n"
				+ "before?= default\nbefore = explicit\n"
				+ "after = explicit\nafter?= default\n"
				+ "individual = first\nindividual = later // comment\n"
				+ "$net.body = 1,048,576\nempty =\ntrailing = left\rright");
		assertEquals(Map.of(
				"family.a\u3000b", "C:\\path=with-equals",
				"name", "second",
				"before", "explicit",
				"after", "explicit",
				"individual", "later",
				"$net.body", "1,048,576",
				"empty", "",
				"trailing", "left\rright"), ConfigTextReader.read(source));
	}

	@Test
	void malformedUtf8ReportsTheActualLineEvenInComments() throws Exception {
		final Path source = directory.resolve("config_data.txt");
		final byte[] prefix = "value=1\n# ".getBytes(StandardCharsets.UTF_8);
		final byte[] bytes = java.util.Arrays.copyOf(prefix, prefix.length + 1);
		bytes[prefix.length] = (byte) 0xFF;
		Files.write(source, bytes);
		final var error = assertThrows(ConfigTextReader.FormatException.class,
				() -> ConfigTextReader.read(source));
		assertTrue(error.getMessage().contains(source + ":2 (R1)"));
	}

	@ParameterizedTest
	@CsvSource(delimiter = '|', value = {
			"$INCLUDE x | R5",
			"$ \tinclude x | R5",
			"invalid? = value | R8",
			"a:b:c = value | R9",
			":a = value | R9",
			"a: = value | R9",
			"run.$?= other | R10",
			"$?= other | R10",
			"run.@x?= other | R10"
	})
	void rejectsInvalidRowsWithPathRuleAndLine(String line, String rule) throws Exception {
		final Path source = directory.resolve("config_data.txt");
		Files.writeString(source, "good = value\n" + line);
		final var error = assertThrows(ConfigTextReader.FormatException.class,
				() -> ConfigTextReader.read(source));
		assertTrue(error.getMessage().contains(source + ":2 (" + rule + "): " + line));
	}
}
