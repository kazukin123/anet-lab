package io.github.kazukin123.anetlab.metricsviewer.infra;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class RunScannerTest {

	@TempDir
	Path runsDir;

	@Test
	void listRunIdIgnoresDirectoriesWithoutMetricsJsonl() throws Exception {
		final Path runA = runsDir.resolve("runA");
		final Path representative = runsDir.resolve("study_t00000");
		Files.createDirectories(runA);
		Files.createDirectories(representative.resolve("trial"));
		Files.writeString(runA.resolve("metrics.jsonl"), "\n", StandardCharsets.UTF_8);
		Files.writeString(runsDir.resolve("plain.txt"), "ignored", StandardCharsets.UTF_8);

		final List<String> runIds = new RunScanner(runsDir.toString()).listRunId();

		assertEquals(List.of("runA"), runIds);
	}
}
