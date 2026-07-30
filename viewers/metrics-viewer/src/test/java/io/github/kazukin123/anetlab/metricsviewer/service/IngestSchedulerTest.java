package io.github.kazukin123.anetlab.metricsviewer.service;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;
import org.springframework.boot.test.system.CapturedOutput;
import org.springframework.boot.test.system.OutputCaptureExtension;

import io.github.kazukin123.anetlab.metricsviewer.infra.RunScanner;

class IngestSchedulerTest {

	@TempDir
	private Path tempDir;

	@Test
	void priorityAndBackgroundRunsReceiveBlocksInAThreeToOneRatio() throws Exception {
		final List<String> runIds = List.of("p1", "p2", "b1", "b2");
		final RunScanner scanner = mock(RunScanner.class);
		when(scanner.listRunId()).thenReturn(runIds);
		for (String runId : runIds) {
			final Path runDir = tempDir.resolve(runId);
			Files.createDirectories(runDir);
			Files.writeString(
					runDir.resolve("metrics.jsonl"),
					"{}\n",
					StandardCharsets.UTF_8);
			when(scanner.resolveRunDir(runId)).thenReturn(runDir);
		}

		final List<String> processed = new ArrayList<>();
		final MetricsIngestor ingestor = mock(MetricsIngestor.class);
		when(ingestor.ingestBlock(anyString(), any(Path.class), any()))
				.thenAnswer(invocation -> {
					processed.add(invocation.getArgument(0));
					return new MetricsIngestor.IngestOutcome(true, "converting");
				});
		final IngestScheduler scheduler =
				new IngestScheduler(scanner, ingestor, new GzipInputSessions());
		scheduler.replacePriority(Set.of("p1", "p2"));

		scheduler.runCycle();
		scheduler.runCycle();

		assertEquals(List.of("p1", "p2", "p1", "b1", "p2", "p1", "p2", "b2"), processed);
	}

	@Test
	void theNonEmptyClassUsesEverySlotWhenTheOtherClassIsEmpty() throws Exception {
		final List<String> runIds = List.of("b1", "b2");
		final RunScanner scanner = mock(RunScanner.class);
		when(scanner.listRunId()).thenReturn(runIds);
		for (String runId : runIds) {
			final Path runDir = tempDir.resolve(runId);
			Files.createDirectories(runDir);
			Files.writeString(
					runDir.resolve("metrics.jsonl"),
					"{}\n",
					StandardCharsets.UTF_8);
			when(scanner.resolveRunDir(runId)).thenReturn(runDir);
		}

		final List<String> processed = new ArrayList<>();
		final MetricsIngestor ingestor = mock(MetricsIngestor.class);
		when(ingestor.ingestBlock(anyString(), any(Path.class), any()))
				.thenAnswer(invocation -> {
					processed.add(invocation.getArgument(0));
					return new MetricsIngestor.IngestOutcome(true, "converting");
				});
		final IngestScheduler scheduler =
				new IngestScheduler(scanner, ingestor, new GzipInputSessions());

		scheduler.runCycle();

		assertEquals(List.of("b1", "b2", "b1", "b2"), processed);
	}

	@Test
	@ExtendWith(OutputCaptureExtension.class)
	void jsonlWinsAndTheDuplicateSourceWarningIsEmittedOncePerRun(CapturedOutput output)
			throws Exception {
		final String runId = "both";
		final Path runDir = tempDir.resolve(runId);
		Files.createDirectories(runDir);
		Files.writeString(runDir.resolve("metrics.jsonl"), "{}\n", StandardCharsets.UTF_8);
		Files.write(runDir.resolve("metrics.jsonl.gz"), new byte[] {31, -117, 8, 0});
		final RunScanner scanner = mock(RunScanner.class);
		when(scanner.listRunId()).thenReturn(List.of(runId));
		when(scanner.resolveRunDir(runId)).thenReturn(runDir);
		final MetricsIngestor ingestor = mock(MetricsIngestor.class);
		when(ingestor.ingestBlock(anyString(), any(Path.class), any()))
				.thenReturn(new MetricsIngestor.IngestOutcome(false, "ready"));
		final IngestScheduler scheduler =
				new IngestScheduler(scanner, ingestor, new GzipInputSessions());

		scheduler.runCycle();
		scheduler.runCycle();

		assertEquals(1, countOccurrences(
				output.getAll(),
				"Both metrics.jsonl and metrics.jsonl.gz exist; using metrics.jsonl: run=both"));
	}

	private static int countOccurrences(String text, String needle) {
		int count = 0;
		int offset = 0;
		while ((offset = text.indexOf(needle, offset)) >= 0) {
			count++;
			offset += needle.length();
		}
		return count;
	}
}
