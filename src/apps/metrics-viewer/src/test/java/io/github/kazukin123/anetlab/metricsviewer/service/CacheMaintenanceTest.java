package io.github.kazukin123.anetlab.metricsviewer.service;

import static org.junit.jupiter.api.Assertions.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.api.extension.ExtendWith;
import org.springframework.boot.test.system.CapturedOutput;
import org.springframework.boot.test.system.OutputCaptureExtension;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsCacheDatabase;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsSource;

class CacheMaintenanceTest {
	@TempDir
	Path root;

	@Test
	void switchingFromConvertingGzipToRawReleasesTheOldStream() throws Exception {
		final Path run = root.resolve("source-switch/runs/r");
		Files.createDirectories(run);
		try (final var output = new java.util.zip.GZIPOutputStream(
				Files.newOutputStream(run.resolve("metrics.jsonl.gz")))) {
			output.write(("{\"type\":\"scalar\",\"tag\":\"x\",\"step\":1,\"value\":1}\n"
					+ "{\"type\":\"scalar\",\"tag\":\"x\",\"step\":2,\"value\":2}\n")
					.getBytes(java.nio.charset.StandardCharsets.UTF_8));
		}
		final GzipInputSessions sessions = new GzipInputSessions();
		final MetricsIngestor ingestor = new MetricsIngestor(new MetricsCacheDatabase(), sessions, 1);
		final IngestScheduler scheduler = new IngestScheduler(() -> List.of(run), ingestor, sessions,
				new RunWarningRegistry(), () -> 0L, 10_000_000_000L);
		try {
			scheduler.runNextBlock();
			assertEquals(1, sessions.activeRuns().size());
			// rawの優先選択へ移ったRunに旧gzipの枠とhandleを残さない。
			Files.writeString(run.resolve("metrics.jsonl"),
					"{\"type\":\"scalar\",\"tag\":\"x\",\"step\":3,\"value\":3}\n");
			scheduler.runNextBlock();
			assertTrue(sessions.activeRuns().isEmpty());
		}
		finally {
			scheduler.close();
		}
	}

	@Test
	void selectedAppendDuringBackgroundBlockIsDetectedBeforeNextPrioritySlot() throws Exception {
		final Path selected = raw("selected", "live", "{}\n");
		final Path firstBackground = raw("background-a", "r", "{}\n");
		final Path secondBackground = raw("background-b", "r", "{}\n");
		final List<Path> order = new ArrayList<>();
		final java.util.concurrent.atomic.AtomicBoolean appended = new java.util.concurrent.atomic.AtomicBoolean();
		final MetricsIngestor ingestor = org.mockito.Mockito.mock(MetricsIngestor.class);
		org.mockito.Mockito.when(ingestor.ingestBlock(org.mockito.ArgumentMatchers.anyString(),
				org.mockito.ArgumentMatchers.any(), org.mockito.ArgumentMatchers.any())).thenAnswer(call -> {
					final Path run = call.getArgument(1);
					order.add(run);
					if (run.equals(selected))
						return new MetricsIngestor.IngestOutcome(true, MetricsCacheDatabase.IngestState.READY, false);
					// cycle途中の追記は、残りの背景blockを待たず次の優先slotへ戻す。
					if (appended.compareAndSet(false, true))
						Files.writeString(selected.resolve("metrics.jsonl"), "{}\n", StandardOpenOption.APPEND);
					return new MetricsIngestor.IngestOutcome(true, MetricsCacheDatabase.IngestState.CONVERTING, true);
				});
		final IngestScheduler scheduler = new IngestScheduler(
				() -> List.of(selected, firstBackground, secondBackground), ingestor, new GzipInputSessions(),
				new RunWarningRegistry(), () -> 0L, 10_000_000_000L);
		try {
			scheduler.replacePriorityPaths(Set.of(selected));
			scheduler.runNextBlock();
			scheduler.runNextBlock();
			scheduler.runNextBlock();
			assertEquals(selected, order.get(2));
			// 最後のslotは背景へ残し、優先probeの追加で配分を変えない。
			scheduler.runNextBlock();
			assertNotEquals(selected, order.get(3));
		}
		finally {
			scheduler.close();
		}
	}

	@Test
	void masterKindChangeDuringABlockIsNotHiddenByIdenticalFileAttributes() throws Exception {
		final Path run = raw("kind-change", "r", "{}\n");
		final Path raw = run.resolve("metrics.jsonl");
		// parserをmockし、同じinodeの別名でも取り込み対象のkind/pathが変わる境界だけを見る。
		Files.createLink(run.resolve("metrics.jsonl.gz"), raw);
		final MetricsIngestor ingestor = org.mockito.Mockito.mock(MetricsIngestor.class);
		org.mockito.Mockito.when(ingestor.ingestBlock(org.mockito.ArgumentMatchers.anyString(),
				org.mockito.ArgumentMatchers.any(), org.mockito.ArgumentMatchers.any())).thenAnswer(call -> {
					Files.deleteIfExists(raw);
					return new MetricsIngestor.IngestOutcome(true, MetricsCacheDatabase.IngestState.READY, false);
				});
		final IngestScheduler scheduler = new IngestScheduler(() -> List.of(run), ingestor, new GzipInputSessions(),
				new RunWarningRegistry(), () -> 0L, 10_000_000_000L);
		try {
			scheduler.runNextBlock();
			scheduler.runNextBlock();
			org.mockito.Mockito.verify(ingestor, org.mockito.Mockito.times(2)).ingestBlock(
					org.mockito.ArgumentMatchers.anyString(), org.mockito.ArgumentMatchers.any(),
					org.mockito.ArgumentMatchers.any());
		}
		finally {
			scheduler.close();
		}
	}

	@Test
	@ExtendWith(OutputCaptureExtension.class)
	void workspaceSummaryCountsPersistedSourceErrors(CapturedOutput output) throws Exception {
		final Path run = raw("source-error", "bad", "{invalid}\n");
		final GzipInputSessions sessions = new GzipInputSessions();
		final IngestScheduler scheduler = new IngestScheduler(() -> List.of(run),
				new MetricsIngestor(new MetricsCacheDatabase()), sessions, new RunWarningRegistry(), System::nanoTime,
				0L);
		try {
			for (int i = 0; i < 8; i++)
				scheduler.runNextBlock();
			assertTrue(output.getAll().contains("workspace=source-error runs=1 pending=0 failed=1"));
		}
		finally {
			scheduler.close();
		}
	}

	@Test
	@ExtendWith(OutputCaptureExtension.class)
	void actualBatchFailureIsHeldUntilTheDatabaseChangesAndOtherRunsCommit(CapturedOutput output) throws Exception {
		final Path bad = raw("failed", "bad", "{\"type\":\"scalar\",\"tag\":\"s\",\"step\":0,\"value\":0}\n");
		final Path good = raw("healthy", "good", "{\"type\":\"scalar\",\"tag\":\"s\",\"step\":0,\"value\":7}\n");
		final MetricsCacheDatabase database = new MetricsCacheDatabase();
		new MetricsIngestor(database).ingestBlock("bad", bad, MetricsSource.select(bad).orElseThrow());
		try (final var handle = database.openWrite(bad); final var statement = handle.connection().createStatement()) {
			statement.execute(
					"CREATE TRIGGER injected_failure BEFORE INSERT ON scalars BEGIN SELECT RAISE(ABORT,'Injected batch failure'); END");
		}
		Files.writeString(bad.resolve("metrics.jsonl"), "{\"type\":\"scalar\",\"tag\":\"s\",\"step\":1,\"value\":1}\n",
				StandardOpenOption.APPEND);
		final GzipInputSessions sessions = new GzipInputSessions();
		final MetricsIngestor ingestor = org.mockito.Mockito
				.spy(new MetricsIngestor(database, sessions, new LodIngestWriter(), new RunWarningRegistry()));
		final IngestScheduler scheduler = new IngestScheduler(() -> List.of(bad, good), ingestor, sessions,
				new RunWarningRegistry(), System::nanoTime, 0L);
		try {
			for (int i = 0; i < 12; i++)
				scheduler.runNextBlock();
			org.mockito.Mockito.verify(ingestor, org.mockito.Mockito.times(1)).ingestBlock(
					org.mockito.ArgumentMatchers.eq("bad"), org.mockito.ArgumentMatchers.eq(bad),
					org.mockito.ArgumentMatchers.any());
			assertEquals(1, output.getAll().split("Metrics cache failed:", -1).length - 1);
			try (final var handle = database.openRead(good);
					final var statement = handle.connection().createStatement();
					final var rows = statement.executeQuery("SELECT COUNT(*) FROM scalars")) {
				assertTrue(rows.next());
				assertEquals(1, rows.getLong(1));
			}
			try (final var handle = database.openWrite(bad);
					final var statement = handle.connection().createStatement()) {
				statement.execute("DROP TRIGGER injected_failure");
			}
			for (int i = 0; i < 4; i++)
				scheduler.runNextBlock();
			org.mockito.Mockito.verify(ingestor, org.mockito.Mockito.times(2)).ingestBlock(
					org.mockito.ArgumentMatchers.eq("bad"), org.mockito.ArgumentMatchers.eq(bad),
					org.mockito.ArgumentMatchers.any());
			try (final var handle = database.openRead(bad);
					final var statement = handle.connection().createStatement();
					final var rows = statement.executeQuery("SELECT count FROM tag_stats")) {
				assertTrue(rows.next());
				assertEquals(2, rows.getLong(1));
			}
		}
		finally {
			scheduler.close();
		}
	}

	@Test
	@ExtendWith(OutputCaptureExtension.class)
	void workspaceInspectionSummaryIsLoggedOnlyWhenItsResultChanges(CapturedOutput output) throws Exception {
		final Path run = raw("summary", "ready", "{}\n");
		final List<Path> runs = new ArrayList<>(List.of(run));
		final MetricsIngestor ingestor = org.mockito.Mockito.mock(MetricsIngestor.class);
		org.mockito.Mockito
				.when(ingestor.ingestBlock(org.mockito.ArgumentMatchers.anyString(), org.mockito.ArgumentMatchers.any(),
						org.mockito.ArgumentMatchers.any()))
				.thenReturn(new MetricsIngestor.IngestOutcome(true, MetricsCacheDatabase.IngestState.READY, false));
		final IngestScheduler scheduler = new IngestScheduler(() -> runs, ingestor, new GzipInputSessions(),
				new RunWarningRegistry(), System::nanoTime, 0L);
		for (int i = 0; i < 8; i++)
			scheduler.runNextBlock();
		final int first = output.getAll().split("Metrics workspace inspected:", -1).length - 1;
		assertTrue(first > 0);
		for (int i = 0; i < 8; i++)
			scheduler.runNextBlock();
		assertEquals(first, output.getAll().split("Metrics workspace inspected:", -1).length - 1);
		runs.add(raw("summary", "added", "{}\n"));
		scheduler.runNextBlock();
		assertTrue(output.getAll().split("Metrics workspace inspected:", -1).length - 1 > first);
	}

	@Test
	void replacingTheGzipFileInvalidatesItsOpenStreamEvenWithIdenticalSizeAndTime() throws Exception {
		final Path run = root.resolve("file-replaced");
		Files.createDirectories(run);
		final Path gzip = run.resolve("metrics.jsonl.gz");
		try (var output = new java.util.zip.GZIPOutputStream(Files.newOutputStream(gzip))) {
			output.write("{}\n".getBytes(java.nio.charset.StandardCharsets.UTF_8));
		}
		final var source = MetricsSource.select(run).orElseThrow();
		final GzipInputSessions sessions = new GzipInputSessions();
		try {
			sessions.acquire(run, source, "generation");
			final Path replacement = root.resolve("replacement.gz");
			Files.copy(gzip, replacement);
			Files.setLastModifiedTime(replacement, Files.getLastModifiedTime(gzip));
			Files.move(replacement, gzip, StandardCopyOption.REPLACE_EXISTING);
			// WindowsのfileKeyが取得不能でも作成時刻の変化で置換を識別する。
			final var creation = Files.readAttributes(gzip, java.nio.file.attribute.BasicFileAttributes.class)
					.creationTime();
			Files.setAttribute(gzip, "basic:creationTime",
					java.nio.file.attribute.FileTime.fromMillis(creation.toMillis() + 2000));
			assertFalse(sessions.hasSession(run, MetricsSource.select(run).orElseThrow()));
		}
		finally {
			sessions.closeAll();
		}
	}

	@Test
	@ExtendWith(OutputCaptureExtension.class)
	void sameNamedRunsInDifferentWorkspacesHaveIndependentWarnings(CapturedOutput output) throws Exception {
		final Path a = raw("a", "same", "{\"type\":\"scalar\",\"tag\":\"score\",\"step\":0,\"value\":null}\n");
		final Path b = raw("b", "same", "{\"type\":\"scalar\",\"tag\":\"score\",\"step\":0,\"value\":null}\n");
		final GzipInputSessions sessions = new GzipInputSessions();
		final RunWarningRegistry warnings = new RunWarningRegistry();
		final IngestScheduler scheduler = new IngestScheduler(() -> List.of(a, b),
				new MetricsIngestor(new MetricsCacheDatabase(), sessions, new LodIngestWriter(), warnings), sessions,
				warnings, System::nanoTime, 0L);
		for (int i = 0; i < 4; i++)
			scheduler.runNextBlock();
		assertEquals(2, output.getAll().split("Skipping invalid scalar value:", -1).length - 1);
	}

	@Test
	void backgroundGzipAdmissionIsLimitedWithoutDiscardingStartedStreams() throws Exception {
		final List<Path> runs = new ArrayList<>();
		final String line = "{\"type\":\"scalar\",\"tag\":\"s\",\"step\":0,\"value\":1}\n";
		for (int i = 0; i < 5; i++) {
			final Path run = root.resolve("w" + i).resolve("runs/r");
			Files.createDirectories(run);
			try (var gzip = new java.util.zip.GZIPOutputStream(
					Files.newOutputStream(run.resolve("metrics.jsonl.gz")))) {
				gzip.write(line.repeat(100).getBytes(java.nio.charset.StandardCharsets.UTF_8));
			}
			runs.add(run);
		}
		final GzipInputSessions sessions = new GzipInputSessions();
		final RunWarningRegistry warnings = new RunWarningRegistry();
		final IngestScheduler scheduler = new IngestScheduler(() -> runs,
				new MetricsIngestor(new MetricsCacheDatabase(), sessions, 1), sessions, warnings, System::nanoTime, 0L);
		try {
			for (int i = 0; i < 20; i++)
				scheduler.runNextBlock();
			assertEquals(4, runs.stream().filter(p -> {
				try {
					return sessions.hasSession(p, MetricsSource.select(p).orElseThrow());
				}
				catch (Exception e) {
					throw new RuntimeException(e);
				}
			}).count());
		}
		finally {
			sessions.closeAll();
		}
	}

	@Test
	void failedDatabaseRunWaitsForChangedAttributesWhileOtherRunsContinue() throws Exception {
		final Path bad = raw("a", "bad", "{}\n"), good = raw("b", "good", "{}\n");
		final MetricsIngestor ingestor = org.mockito.Mockito.mock(MetricsIngestor.class);
		org.mockito.Mockito
				.when(ingestor.ingestBlock(org.mockito.ArgumentMatchers.eq("bad"), org.mockito.ArgumentMatchers.eq(bad),
						org.mockito.ArgumentMatchers.any()))
				.thenThrow(new java.sql.SQLException("Injected write failure"));
		org.mockito.Mockito
				.when(ingestor.ingestBlock(org.mockito.ArgumentMatchers.eq("good"),
						org.mockito.ArgumentMatchers.eq(good), org.mockito.ArgumentMatchers.any()))
				.thenReturn(new MetricsIngestor.IngestOutcome(true, MetricsCacheDatabase.IngestState.READY, false));
		final IngestScheduler scheduler = new IngestScheduler(() -> List.of(bad, good), ingestor,
				new GzipInputSessions(), new RunWarningRegistry(), System::nanoTime, 0L);
		for (int i = 0; i < 8; i++)
			scheduler.runNextBlock();
		org.mockito.Mockito.verify(ingestor, org.mockito.Mockito.times(1)).ingestBlock(
				org.mockito.ArgumentMatchers.eq("bad"), org.mockito.ArgumentMatchers.eq(bad),
				org.mockito.ArgumentMatchers.any());
		org.mockito.Mockito.verify(ingestor, org.mockito.Mockito.atLeastOnce()).ingestBlock(
				org.mockito.ArgumentMatchers.eq("good"), org.mockito.ArgumentMatchers.eq(good),
				org.mockito.ArgumentMatchers.any());
		Files.writeString(bad.resolve("metrics.jsonl"), "{}\n{}\n");
		for (int i = 0; i < 4; i++)
			scheduler.runNextBlock();
		org.mockito.Mockito.verify(ingestor, org.mockito.Mockito.times(2)).ingestBlock(
				org.mockito.ArgumentMatchers.eq("bad"), org.mockito.ArgumentMatchers.eq(bad),
				org.mockito.ArgumentMatchers.any());
	}

	@Test
	void unchangedTerminalRunIsNotSentBackToIngestorOnEachCycle() throws Exception {
		final Path run = raw("a", "ready", "{}\n");
		final MetricsIngestor ingestor = org.mockito.Mockito.mock(MetricsIngestor.class);
		org.mockito.Mockito
				.when(ingestor.ingestBlock(org.mockito.ArgumentMatchers.anyString(), org.mockito.ArgumentMatchers.any(),
						org.mockito.ArgumentMatchers.any()))
				.thenReturn(new MetricsIngestor.IngestOutcome(false, MetricsCacheDatabase.IngestState.READY, false));
		final IngestScheduler scheduler = new IngestScheduler(() -> List.of(run), ingestor, new GzipInputSessions(),
				new RunWarningRegistry(), System::nanoTime, 0L);
		for (int i = 0; i < 12; i++)
			scheduler.runNextBlock();
		org.mockito.Mockito.verify(ingestor, org.mockito.Mockito.times(1)).ingestBlock(
				org.mockito.ArgumentMatchers.anyString(), org.mockito.ArgumentMatchers.any(),
				org.mockito.ArgumentMatchers.any());
	}

	@Test
	@ExtendWith(OutputCaptureExtension.class)
	void inspectionContinuesAfterIts128RunSegment(CapturedOutput output) throws Exception {
		final List<Path> runs = new ArrayList<>();
		for (int i = 0; i < 129; i++)
			runs.add(raw("w", "r" + i, "{}\n"));
		final MetricsIngestor ingestor = org.mockito.Mockito.mock(MetricsIngestor.class);
		org.mockito.Mockito
				.when(ingestor.ingestBlock(org.mockito.ArgumentMatchers.anyString(), org.mockito.ArgumentMatchers.any(),
						org.mockito.ArgumentMatchers.any()))
				.thenReturn(new MetricsIngestor.IngestOutcome(false, MetricsCacheDatabase.IngestState.READY, false));
		final IngestScheduler scheduler = new IngestScheduler(() -> runs, ingestor, new GzipInputSessions(),
				new RunWarningRegistry(), () -> 0L, 10_000_000_000L);
		scheduler.runNextBlock();
		org.mockito.Mockito.verify(ingestor, org.mockito.Mockito.times(128))
				.inspectCache(org.mockito.ArgumentMatchers.any(), org.mockito.ArgumentMatchers.any());
		scheduler.runNextBlock();
		org.mockito.Mockito.verify(ingestor, org.mockito.Mockito.times(129))
				.inspectCache(org.mockito.ArgumentMatchers.any(), org.mockito.ArgumentMatchers.any());
	}

	@Test
	void fullGzipPoolWaitsUntilAStreamIsReleased() throws Exception {
		final GzipInputSessions sessions = new GzipInputSessions();
		final List<Path> runs = new ArrayList<>();
		for (int i = 0; i < 17; i++) {
			final Path run = root.resolve("w/runs/r" + i);
			Files.createDirectories(run);
			try (var gzip = new java.util.zip.GZIPOutputStream(
					Files.newOutputStream(run.resolve("metrics.jsonl.gz")))) {
				gzip.write("{}\n".getBytes(java.nio.charset.StandardCharsets.UTF_8));
			}
			runs.add(run);
		}
		sessions.replacePriority(Set.copyOf(runs));
		try {
			for (int i = 0; i < 16; i++)
				sessions.acquire(runs.get(i), MetricsSource.select(runs.get(i)).orElseThrow(), "generation");
			final var waiting = assertThrows(GzipInputSessions.ResourceUnavailableException.class, () -> sessions
					.acquire(runs.get(16), MetricsSource.select(runs.get(16)).orElseThrow(), "generation"));
			assertEquals("gzip_stream_limit", waiting.reason());
			sessions.close(runs.get(0));
			sessions.acquire(runs.get(16), MetricsSource.select(runs.get(16)).orElseThrow(), "generation");
			assertTrue(sessions.hasSession(runs.get(16), MetricsSource.select(runs.get(16)).orElseThrow()));
		}
		finally {
			sessions.closeAll();
		}
	}

	@Test
	void newPriorityIsInspectedBeforeAnUnfinishedFullScan() throws Exception {
		final List<Path> runs = new ArrayList<>();
		for (int i = 0; i < 512; i++)
			runs.add(raw("w", "r" + i, "{}\n"));
		final List<Path> processed = new ArrayList<>();
		final MetricsIngestor ingestor = org.mockito.Mockito.mock(MetricsIngestor.class);
		org.mockito.Mockito.when(ingestor.ingestBlock(org.mockito.ArgumentMatchers.anyString(),
				org.mockito.ArgumentMatchers.any(), org.mockito.ArgumentMatchers.any())).thenAnswer(call -> {
					processed.add(call.getArgument(1));
					return new MetricsIngestor.IngestOutcome(true, MetricsCacheDatabase.IngestState.CONVERTING, true);
				});
		final IngestScheduler scheduler = new IngestScheduler(() -> runs, ingestor, new GzipInputSessions(),
				new RunWarningRegistry(), () -> 0L, 10_000_000_000L);
		scheduler.runNextBlock();
		scheduler.replacePriorityPaths(Set.of(runs.get(511)));
		scheduler.runNextBlock();
		assertEquals(runs.get(511), processed.get(1));
	}

	@Test
	void readyPriorityRunObservesAppendBeforeTheNextFullScan() throws Exception {
		final Path priority = raw("a", "priority", "{}\n"), background = raw("b", "background", "{}\n");
		final MetricsIngestor ingestor = org.mockito.Mockito.mock(MetricsIngestor.class);
		org.mockito.Mockito
				.when(ingestor.ingestBlock(org.mockito.ArgumentMatchers.anyString(), org.mockito.ArgumentMatchers.any(),
						org.mockito.ArgumentMatchers.any()))
				.thenAnswer(call -> "priority".equals(call.getArgument(0))
						? new MetricsIngestor.IngestOutcome(true, MetricsCacheDatabase.IngestState.READY, false)
						: new MetricsIngestor.IngestOutcome(true, MetricsCacheDatabase.IngestState.CONVERTING, true));
		final IngestScheduler scheduler = new IngestScheduler(() -> List.of(priority, background), ingestor,
				new GzipInputSessions(), new RunWarningRegistry(), () -> 0L, 10_000_000_000L);
		scheduler.replacePriorityPaths(Set.of(priority));
		for (int i = 0; i < 8; i++)
			scheduler.runNextBlock();
		Files.writeString(priority.resolve("metrics.jsonl"), "{}\n{}\n");
		for (int i = 0; i < 4; i++)
			scheduler.runNextBlock();
		org.mockito.Mockito.verify(ingestor, org.mockito.Mockito.times(2)).ingestBlock(
				org.mockito.ArgumentMatchers.eq("priority"), org.mockito.ArgumentMatchers.eq(priority),
				org.mockito.ArgumentMatchers.any());
	}

	private Path raw(String workspace, String id, String text) throws Exception {
		final Path run = root.resolve(workspace).resolve("runs").resolve(id);
		Files.createDirectories(run);
		Files.writeString(run.resolve("metrics.jsonl"), text);
		return run;
	}
}
