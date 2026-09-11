package io.github.kazukin123.anetlab.metricsviewer.service;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;
import org.springframework.boot.test.system.CapturedOutput;
import org.springframework.boot.test.system.OutputCaptureExtension;

import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsCacheDatabase;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsCacheDatabase.IngestState;
import io.github.kazukin123.anetlab.metricsviewer.infra.RunScanner;

class IngestSchedulerTest {

	@TempDir
	private Path tempDir;

	@Test
	void terminalNoOpRunIsInspectedOnlyOncePerFourSlotCycle() throws Exception {
		final String runId = "ready";
		final Path runDir = tempDir.resolve(runId);
		Files.createDirectories(runDir);
		Files.writeString(
				runDir.resolve("metrics.jsonl"),
				"{}\n",
				StandardCharsets.UTF_8);
		final RunScanner scanner = mock(RunScanner.class);
		when(scanner.listRunId()).thenReturn(List.of(runId));
		when(scanner.resolveRunDir(runId)).thenReturn(runDir);
		final MetricsIngestor ingestor = mock(MetricsIngestor.class);
		when(ingestor.ingestBlock(anyString(), any(Path.class), any()))
				.thenReturn(new MetricsIngestor.IngestOutcome(
						false, IngestState.READY, false));
		final IngestScheduler scheduler = new IngestScheduler(
				scanner, ingestor, new GzipInputSessions(), new RunWarningRegistry());

		runBlocks(scheduler, 4);

		verify(ingestor, times(1)).ingestBlock(anyString(), any(Path.class), any());
	}

	@Test
	void terminalPriorityRunYieldsEverySlotToActionableBackground() throws Exception {
		final List<String> processed = runMixedCycle("p", "b", Set.of("p"));

		assertEquals(List.of("p", "b", "b", "b", "b"), processed);
	}

	@Test
	void terminalBackgroundRunYieldsItsSlotToActionablePriority() throws Exception {
		final List<String> processed = runMixedCycle("b", "p", Set.of("p"));

		assertEquals(List.of("p", "p", "p", "b", "p"), processed);
	}

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
					return new MetricsIngestor.IngestOutcome(
							true, IngestState.CONVERTING, true);
				});
		final IngestScheduler scheduler =
				new IngestScheduler(
						scanner, ingestor, new GzipInputSessions(), new RunWarningRegistry());
		scheduler.replacePriority(Set.of("p1", "p2"));

		for (int block = 0; block < 8; block++) scheduler.runNextBlock();

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
					return new MetricsIngestor.IngestOutcome(
							true, IngestState.CONVERTING, true);
				});
		final IngestScheduler scheduler =
				new IngestScheduler(
						scanner, ingestor, new GzipInputSessions(), new RunWarningRegistry());

		for (int block = 0; block < 4; block++) scheduler.runNextBlock();

		assertEquals(List.of("b1", "b2", "b1", "b2"), processed);
	}

	@Test
	void priorityAddedDuringScanIsNotPrunedByThatScan() throws Exception {
		final CountDownLatch scanStarted = new CountDownLatch(1);
		final CountDownLatch finishScan = new CountDownLatch(1);
		final RunScanner scanner = mock(RunScanner.class);
		when(scanner.listRunId()).thenAnswer(ignored -> {
			scanStarted.countDown();
			finishScan.await(2, TimeUnit.SECONDS);
			return List.of();
		});
		final IngestScheduler scheduler = new IngestScheduler(
				scanner,
				mock(MetricsIngestor.class),
				new GzipInputSessions(),
				new RunWarningRegistry());
		final ExecutorService executor = Executors.newSingleThreadExecutor();

		try {
			final Future<Boolean> cycle = executor.submit(scheduler::runNextBlock);
			assertTrue(scanStarted.await(2, TimeUnit.SECONDS));
			scheduler.replacePriority(Set.of("new-run"));
			finishScan.countDown();
			cycle.get(2, TimeUnit.SECONDS);
		} finally {
			finishScan.countDown();
			executor.shutdownNow();
		}

		assertEquals(Set.of("new-run"), scheduler.priorityRunIds());
	}

	@Test
	@ExtendWith(OutputCaptureExtension.class)
	void duplicateSourceWarningIsEmittedAgainAfterTheRunReentersTheWorkSet(
			CapturedOutput output)
			throws Exception {
		final String runId = "both";
		final Path runDir = tempDir.resolve(runId);
		Files.createDirectories(runDir);
		Files.writeString(runDir.resolve("metrics.jsonl"), "{}\n", StandardCharsets.UTF_8);
		Files.write(runDir.resolve("metrics.jsonl.gz"), new byte[] {31, -117, 8, 0});
		final RunScanner scanner = mock(RunScanner.class);
		when(scanner.listRunId()).thenReturn(
				List.of(runId),
				List.of(runId),
				List.of(),
				List.of(runId));
		when(scanner.resolveRunDir(runId)).thenReturn(runDir);
		final MetricsIngestor ingestor = mock(MetricsIngestor.class);
		when(ingestor.ingestBlock(anyString(), any(Path.class), any()))
				.thenReturn(new MetricsIngestor.IngestOutcome(false, IngestState.READY, false));
		final IngestScheduler scheduler =
				new IngestScheduler(
						scanner, ingestor, new GzipInputSessions(), new RunWarningRegistry());

		runBlocks(scheduler, 4);
		runBlocks(scheduler, 4);
		runBlocks(scheduler, 4);
		runBlocks(scheduler, 4);

		assertEquals(2, countOccurrences(
				output.getAll(),
				"Both metrics.jsonl and metrics.jsonl.gz exist; using metrics.jsonl: run=both"));
	}

	@Test
	@ExtendWith(OutputCaptureExtension.class)
	void scalarWarningIsEmittedAgainAfterTheRunLeavesAndReentersTheWorkSet(
			CapturedOutput output) throws Exception {
		final String runId = "scalar-warning";
		final Path runDir = tempDir.resolve(runId);
		Files.createDirectories(runDir);
		final Path source = runDir.resolve("metrics.jsonl");
		final Path parkedSource = runDir.resolve("metrics.jsonl.parked");
		Files.writeString(
				source,
				"{\"type\":\"scalar\",\"tag\":\"loss\",\"step\":1,\"value\":null}\n",
				StandardCharsets.UTF_8);
		final GzipInputSessions gzipSessions = new GzipInputSessions();
		final RunWarningRegistry warningRegistry = new RunWarningRegistry();
		final MetricsIngestor ingestor = new MetricsIngestor(
				new MetricsCacheDatabase(),
				gzipSessions,
				new LodIngestWriter(),
				warningRegistry);
		final IngestScheduler scheduler = new IngestScheduler(
				new RunScanner(tempDir.toString()),
				ingestor,
				gzipSessions,
				warningRegistry);

		runBlocks(scheduler, 4);
		runBlocks(scheduler, 4);
		Files.move(source, parkedSource);
		runBlocks(scheduler, 4);
		Files.move(parkedSource, source);
		Files.writeString(
				source,
				"{\"type\":\"scalar\",\"tag\":\"loss\",\"step\":2,\"value\":null}\n",
				StandardCharsets.UTF_8,
				StandardOpenOption.APPEND);
		runBlocks(scheduler, 4);

		assertEquals(2, countOccurrences(
				output.getAll(),
				"run=scalar-warning tag=loss reason=null"));
	}

	private static void runBlocks(IngestScheduler scheduler, int count) {
		for (int block = 0; block < count; block++) scheduler.runNextBlock();
	}

	private List<String> runMixedCycle(
			String terminalRun,
			String convertingRun,
			Set<String> priorityRuns) throws Exception {
		final List<String> runIds = List.of("p", "b");
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
					final String runId = invocation.getArgument(0);
					processed.add(runId);
					if (runId.equals(terminalRun)) {
						return new MetricsIngestor.IngestOutcome(
								false, IngestState.READY, false);
					}
					assertEquals(convertingRun, runId);
					return new MetricsIngestor.IngestOutcome(
							true, IngestState.CONVERTING, true);
				});
		final IngestScheduler scheduler = new IngestScheduler(
				scanner, ingestor, new GzipInputSessions(), new RunWarningRegistry());
		scheduler.replacePriority(priorityRuns);

		runBlocks(scheduler, 4);
		return processed;
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
