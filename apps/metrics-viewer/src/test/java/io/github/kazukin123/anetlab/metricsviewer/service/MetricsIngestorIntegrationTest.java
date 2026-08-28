package io.github.kazukin123.anetlab.metricsviewer.service;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.FileTime;
import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.Arrays;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.zip.GZIPOutputStream;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;
import org.springframework.boot.test.system.CapturedOutput;
import org.springframework.boot.test.system.OutputCaptureExtension;

import io.github.kazukin123.anetlab.metricsviewer.config.MetricsViewerSettings;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsCacheDatabase;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsCacheDatabase.ConnectionHandle;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsCacheDatabase.IngestState;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsSource;
import io.github.kazukin123.anetlab.metricsviewer.infra.RunScanner;
import io.github.kazukin123.anetlab.metricsviewer.view.model.RunInfo;

class MetricsIngestorIntegrationTest {

	@TempDir
	private Path tempDir;

	@Test
	void unchangedReadyRunSkipsFingerprintAndDatabaseWork() throws Exception {
		final String runId = "run-ready-fast-path";
		final Path runDir = tempDir.resolve(runId);
		Files.createDirectories(runDir);
		Files.writeString(
				runDir.resolve("metrics.jsonl"),
				scalarLines(0, 2),
				StandardCharsets.UTF_8);

		final CountingDatabase database = new CountingDatabase();
		final MetricsIngestor ingestor = new MetricsIngestor(database);
		assertEquals(IngestState.READY, ingestor.ingestBlock(
				runId,
				runDir,
				MetricsSource.select(runDir).orElseThrow()).state());
		database.resetCounts();
		final MetricsSource unchangedSource = spy(MetricsSource.select(runDir).orElseThrow());

		final MetricsIngestor.IngestOutcome idle = ingestor.ingestBlock(
				runId, runDir, unchangedSource);

		assertFalse(idle.didWork());
		assertFalse(idle.immediateRetry());
		assertEquals(IngestState.READY, idle.state());
		assertEquals(0, database.prepareCount());
		assertEquals(0, database.openWriteCount());
		verify(unchangedSource, never()).headSha256();
		verify(unchangedSource, never()).headSha256(org.mockito.ArgumentMatchers.anyLong());
		verify(unchangedSource, never()).sha256Before(org.mockito.ArgumentMatchers.anyLong());
	}

	@Test
	void unchangedErrorRunSkipsFingerprintAndDatabaseWork() throws Exception {
		final String runId = "run-error-fast-path";
		final Path runDir = tempDir.resolve(runId);
		Files.createDirectories(runDir);
		Files.writeString(
				runDir.resolve("metrics.jsonl"),
				"{invalid json}\n",
				StandardCharsets.UTF_8);

		final CountingDatabase database = new CountingDatabase();
		final MetricsIngestor ingestor = new MetricsIngestor(database);
		assertEquals(IngestState.ERROR, ingestor.ingestBlock(
				runId,
				runDir,
				MetricsSource.select(runDir).orElseThrow()).state());
		database.resetCounts();
		final MetricsSource unchangedSource = spy(MetricsSource.select(runDir).orElseThrow());

		final MetricsIngestor.IngestOutcome idle = ingestor.ingestBlock(
				runId, runDir, unchangedSource);

		assertFalse(idle.didWork());
		assertEquals(IngestState.ERROR, idle.state());
		assertEquals(0, database.prepareCount());
		assertEquals(0, database.openWriteCount());
		verify(unchangedSource, never()).headSha256();
		verify(unchangedSource, never()).sha256Before(org.mockito.ArgumentMatchers.anyLong());
	}

	@Test
	void failedErrorPersistenceDoesNotPublishFastPathObservation() throws Exception {
		final String runId = "run-error-persistence-failure";
		final Path runDir = tempDir.resolve(runId);
		Files.createDirectories(runDir);
		Files.writeString(
				runDir.resolve("metrics.jsonl"),
				"{invalid json}\n",
				StandardCharsets.UTF_8);

		final FailErrorPersistenceDatabase database = new FailErrorPersistenceDatabase();
		final MetricsIngestor ingestor = new MetricsIngestor(database);
		assertEquals(IngestState.ERROR, ingestor.ingestBlock(
				runId,
				runDir,
				MetricsSource.select(runDir).orElseThrow()).state());
		database.resetCounts();

		assertEquals(IngestState.ERROR, ingestor.ingestBlock(
				runId,
				runDir,
				MetricsSource.select(runDir).orElseThrow()).state());

		assertEquals(1, database.prepareCount());
		assertTrue(database.openWriteCount() > 0);
	}

	@Test
	void runReentryAfterWorkSetPruneUsesFullValidation() throws Exception {
		final String runId = "run-reentry";
		final Path runDir = tempDir.resolve(runId);
		Files.createDirectories(runDir);
		Files.writeString(
				runDir.resolve("metrics.jsonl"),
				scalarLines(0, 1),
				StandardCharsets.UTF_8);

		final CountingDatabase database = new CountingDatabase();
		final MetricsIngestor ingestor = new MetricsIngestor(database);
		assertEquals(IngestState.READY, ingestor.ingestBlock(
				runId,
				runDir,
				MetricsSource.select(runDir).orElseThrow()).state());
		ingestor.retainRuns(Set.of());
		database.resetCounts();

		assertEquals(IngestState.READY, ingestor.ingestBlock(
				runId,
				runDir,
				MetricsSource.select(runDir).orElseThrow()).state());

		assertEquals(1, database.prepareCount());
		assertEquals(1, database.openWriteCount());
	}

	@Test
	void newIngestorDoesNotReuseAnotherProcessObservation() throws Exception {
		final String runId = "run-process-restart";
		final Path runDir = tempDir.resolve(runId);
		Files.createDirectories(runDir);
		Files.writeString(
				runDir.resolve("metrics.jsonl"),
				scalarLines(0, 1),
				StandardCharsets.UTF_8);

		final CountingDatabase database = new CountingDatabase();
		assertEquals(IngestState.READY, new MetricsIngestor(database).ingestBlock(
				runId,
				runDir,
				MetricsSource.select(runDir).orElseThrow()).state());
		database.resetCounts();

		assertEquals(IngestState.READY, new MetricsIngestor(database).ingestBlock(
				runId,
				runDir,
				MetricsSource.select(runDir).orElseThrow()).state());

		assertEquals(1, database.prepareCount());
		assertEquals(1, database.openWriteCount());
	}

	@Test
	void cacheDatabaseAttributeChangeReturnsToFullValidation() throws Exception {
		final String runId = "run-cache-attribute-change";
		final Path runDir = tempDir.resolve(runId);
		Files.createDirectories(runDir);
		Files.writeString(
				runDir.resolve("metrics.jsonl"),
				scalarLines(0, 1),
				StandardCharsets.UTF_8);

		final CountingDatabase database = new CountingDatabase();
		final MetricsIngestor ingestor = new MetricsIngestor(database);
		assertEquals(IngestState.READY, ingestor.ingestBlock(
				runId,
				runDir,
				MetricsSource.select(runDir).orElseThrow()).state());
		final Path cache = runDir.resolve("metrics_cache.db");
		final FileTime changedMtime = FileTime.fromMillis(
				Files.getLastModifiedTime(cache).toMillis() + 2_000L);
		Files.setLastModifiedTime(cache, changedMtime);
		database.resetCounts();

		assertEquals(IngestState.READY, ingestor.ingestBlock(
				runId,
				runDir,
				MetricsSource.select(runDir).orElseThrow()).state());

		assertEquals(1, database.prepareCount());
		assertEquals(1, database.openWriteCount());
	}

	@Test
	void switchYieldCommitsCompleteLinesAndNextBlockResumesFromCommittedOffset()
			throws Exception {
		final String runId = "run-switch-yield";
		final Path runDir = tempDir.resolve(runId);
		Files.createDirectories(runDir);
		Files.writeString(
				runDir.resolve("metrics.jsonl"),
				scalarLines(0, 5),
				StandardCharsets.UTF_8);

		final MetricsCacheDatabase database = new MetricsCacheDatabase();
		final AtomicBoolean yieldEnabled = new AtomicBoolean(true);
		final AtomicInteger completedLineChecks = new AtomicInteger();
		final MetricsIngestor ingestor = new MetricsIngestor(
				database,
				new GzipInputSessions(),
				new LodIngestWriter(),
				new RunWarningRegistry(),
				MetricsIngestor.MAX_BLOCK_LINES,
				() -> yieldEnabled.get() && completedLineChecks.incrementAndGet() >= 2);
		final MetricsSource source = MetricsSource.select(runDir).orElseThrow();

		final MetricsIngestor.IngestOutcome yielded =
				ingestor.ingestBlock(runId, runDir, source);
		assertEquals(IngestState.CONVERTING, yielded.state());
		assertTrue(yielded.immediateRetry());
		try (ConnectionHandle handle = database.openRead(runDir)) {
			assertEquals(2L, queryLong(handle.connection(), "SELECT COUNT(*) FROM scalars"));
			assertEquals("converting", queryString(
					handle.connection(),
					"SELECT v FROM source_meta WHERE k='state'"));
		}

		yieldEnabled.set(false);
		final MetricsIngestor.IngestOutcome resumed =
				ingestor.ingestBlock(runId, runDir, source);
		assertEquals(IngestState.READY, resumed.state());
		assertFalse(resumed.immediateRetry());
		try (ConnectionHandle handle = database.openRead(runDir)) {
			assertEquals(5L, queryLong(handle.connection(), "SELECT COUNT(*) FROM scalars"));
			assertEquals("ready", queryString(
					handle.connection(),
					"SELECT v FROM source_meta WHERE k='state'"));
		}
	}

	@Test
	void databaseWriteFailureKeepsRunRetryableAndNextAttemptRecovers() throws Exception {
		final String runId = "run-database-retry";
		final Path runDir = tempDir.resolve(runId);
		Files.createDirectories(runDir);
		Files.writeString(
				runDir.resolve("metrics.jsonl"),
				scalarLines(0, 2),
				StandardCharsets.UTF_8);

		final MetricsCacheDatabase database = new FailSecondWriteDatabase();
		final MetricsIngestor ingestor = new MetricsIngestor(
				database,
				new GzipInputSessions(),
				1);
		final MetricsRepository repository = createRepository(database);
		final MetricsSource source = MetricsSource.select(runDir).orElseThrow();

		final MetricsIngestor.IngestOutcome first = ingestor.ingestBlock(runId, runDir, source);
		assertEquals(IngestState.CONVERTING, first.state());
		final RunInfo beforeFailure = repository.findRunInfo(runId);
		assertEquals("converting", beforeFailure.getIngest().state());
		assertEquals(50, beforeFailure.getIngest().percentage());
		assertNull(beforeFailure.getIngest().error());

		assertThrows(SQLException.class, () -> ingestor.ingestBlock(runId, runDir, source));

		final RunInfo afterFailure = repository.findRunInfo(runId);
		assertEquals(beforeFailure.getGeneration(), afterFailure.getGeneration());
		assertEquals("converting", afterFailure.getIngest().state());
		assertEquals(50, afterFailure.getIngest().percentage());
		assertNull(afterFailure.getIngest().error());

		final MetricsIngestor.IngestOutcome retry = ingestor.ingestBlock(runId, runDir, source);
		assertEquals(IngestState.READY, retry.state());
		final RunInfo recovered = repository.findRunInfo(runId);
		assertEquals(beforeFailure.getGeneration(), recovered.getGeneration());
		assertEquals("ready", recovered.getIngest().state());
		assertEquals(100, recovered.getIngest().percentage());
		assertNull(recovered.getIngest().error());
		assertEquals(2L, recovered.getTags().get(0).getStats().getCount());
	}

	@Test
	@ExtendWith(OutputCaptureExtension.class)
	void invalidCompletedJsonRollsBackTheWholeBlock(CapturedOutput output) throws Exception {
		final Path runDir = tempDir.resolve("run-invalid-json");
		Files.createDirectories(runDir);
		Files.writeString(
				runDir.resolve("metrics.jsonl"),
				"""
				{"type":"scalar","tag":"loss","step":1,"value":2.0}
				{invalid json}
				""",
				StandardCharsets.UTF_8);

		final MetricsCacheDatabase database = new MetricsCacheDatabase();
		final MetricsIngestor ingestor = new MetricsIngestor(database);
		final MetricsIngestor.IngestOutcome outcome = ingestor.ingestBlock(
				"run-invalid-json",
				runDir,
				MetricsSource.select(runDir).orElseThrow());

		assertEquals(IngestState.ERROR, outcome.state());
		try (ConnectionHandle handle = database.openRead(runDir)) {
			final Connection connection = handle.connection();
			assertEquals(0L, queryLong(connection, "SELECT COUNT(*) FROM scalars"));
			assertEquals(0L, queryLong(connection, "SELECT COUNT(*) FROM tag_stats"));
			assertEquals(0L, queryLong(connection,
					"SELECT CAST(v AS INTEGER) FROM source_meta WHERE k='committed_offset'"));
			assertEquals("invalid_json", queryString(connection,
					"SELECT v FROM source_meta WHERE k='error_code'"));
		}
		assertEquals(1, countOccurrences(
				output.getAll(),
				"Run ingest entered error state: run=run-invalid-json code=invalid_json"));
	}

	@Test
	void blankCompletedLineIsInvalidJsonAndRollsBackTheWholeBlock() throws Exception {
		final Path runDir = tempDir.resolve("run-blank-line");
		Files.createDirectories(runDir);
		Files.writeString(
				runDir.resolve("metrics.jsonl"),
				"{\"type\":\"scalar\",\"tag\":\"loss\",\"step\":1,\"value\":2.0}\n\n",
				StandardCharsets.UTF_8);

		final MetricsCacheDatabase database = new MetricsCacheDatabase();
		final MetricsIngestor.IngestOutcome outcome = new MetricsIngestor(database).ingestBlock(
				"run-blank-line",
				runDir,
				MetricsSource.select(runDir).orElseThrow());

		assertEquals(IngestState.ERROR, outcome.state());
		try (ConnectionHandle handle = database.openRead(runDir)) {
			assertEquals(0L, queryLong(handle.connection(), "SELECT COUNT(*) FROM scalars"));
			assertEquals("invalid_json", queryString(
					handle.connection(),
					"SELECT v FROM source_meta WHERE k='error_code'"));
		}
	}

	@Test
	void fatalRunErrorIsRetriedFromANewGenerationAfterTheSourceChanges() throws Exception {
		final Path runDir = tempDir.resolve("run-error-retry");
		Files.createDirectories(runDir);
		final Path jsonl = runDir.resolve("metrics.jsonl");
		Files.writeString(jsonl, "{invalid json}\n", StandardCharsets.UTF_8);

		final MetricsCacheDatabase database = new MetricsCacheDatabase();
		final MetricsIngestor ingestor = new MetricsIngestor(database);
		assertEquals(IngestState.ERROR, ingestor.ingestBlock(
				"run-error-retry",
				runDir,
				MetricsSource.select(runDir).orElseThrow()).state());
		final String failedGeneration;
		try (ConnectionHandle handle = database.openRead(runDir)) {
			failedGeneration = queryString(
					handle.connection(),
					"SELECT v FROM source_meta WHERE k='generation'");
		}

		Files.writeString(
				jsonl,
				"{\"type\":\"scalar\",\"tag\":\"loss\",\"step\":1,\"value\":2.0}\n",
				StandardCharsets.UTF_8);
		assertEquals(IngestState.READY, ingestor.ingestBlock(
				"run-error-retry",
				runDir,
				MetricsSource.select(runDir).orElseThrow()).state());
		try (ConnectionHandle handle = database.openRead(runDir)) {
			assertNotEquals(failedGeneration, queryString(
					handle.connection(),
					"SELECT v FROM source_meta WHERE k='generation'"));
			assertEquals(1L, queryLong(handle.connection(), "SELECT COUNT(*) FROM scalars"));
		}
	}

	@Test
	void everyFatalRecordShapeRollsBackWithAStableErrorCode() throws Exception {
		final String[][] cases = {
				{"[]\n", "invalid_record"},
				{"{\"type\":\"scalar\",\"tag\":\"loss\",\"step\":1}\n", "invalid_record"},
				{"{\"type\":\"scalar\",\"tag\":\"loss\",\"step\":1.5,\"value\":1}\n", "invalid_step"},
				{"{\"type\":\"scalar\",\"tag\":\"loss\",\"step\":9007199254740992,\"value\":1}\n",
						"invalid_step"},
				{"{\"tag\":\"weights\",\"values\":[1,2]}\n", "invalid_record"}
		};

		for (int i = 0; i < cases.length; i++) {
			final Path runDir = tempDir.resolve("run-fatal-" + i);
			Files.createDirectories(runDir);
			Files.writeString(runDir.resolve("metrics.jsonl"), cases[i][0], StandardCharsets.UTF_8);
			final MetricsCacheDatabase database = new MetricsCacheDatabase();

			final MetricsIngestor.IngestOutcome outcome = new MetricsIngestor(database).ingestBlock(
					"run-fatal-" + i,
					runDir,
					MetricsSource.select(runDir).orElseThrow());

			assertEquals(IngestState.ERROR, outcome.state());
			try (ConnectionHandle handle = database.openRead(runDir)) {
				assertEquals(cases[i][1], queryString(
						handle.connection(),
						"SELECT v FROM source_meta WHERE k='error_code'"));
				assertEquals(0L, queryLong(handle.connection(), "SELECT COUNT(*) FROM scalars"));
			}
		}
	}

	@Test
	@ExtendWith(OutputCaptureExtension.class)
	void invalidValuesAreSkippedAndOnlyTheRegressingTagIsQuarantined(CapturedOutput output)
			throws Exception {
		final Path runDir = tempDir.resolve("run-quarantine");
		Files.createDirectories(runDir);
		Files.writeString(
				runDir.resolve("metrics.jsonl"),
				"""
				{"type":"scalar","tag":"loss","step":100,"value":null}
				{"type":"scalar","tag":"loss","step":100,"value":"not-a-number"}
				{"type":"scalar","tag":"loss","step":100,"value":NaN}
				{"type":"scalar","tag":"loss","step":100,"value":1e100}
				{"type":"scalar","tag":"loss","step":2,"value":2.0}
				{"type":"scalar","tag":"other","step":1,"value":10.0}
				{"type":"scalar","tag":"loss","step":1,"value":3.0}
				{"type":"scalar","tag":"loss","step":3,"value":4.0}
				{"type":"histogram","tag":"weights","step":5,"values":[1,2]}
				{"type":"scalar","tag":"other","step":2,"value":14.0}
				""",
				StandardCharsets.UTF_8);

		final MetricsCacheDatabase database = new MetricsCacheDatabase();
		final MetricsIngestor ingestor = new MetricsIngestor(database);
		final MetricsIngestor.IngestOutcome outcome = ingestor.ingestBlock(
				"run-quarantine",
				runDir,
				MetricsSource.select(runDir).orElseThrow());

		assertEquals(IngestState.READY, outcome.state());
		try (ConnectionHandle handle = database.openRead(runDir)) {
			final Connection connection = handle.connection();
			assertEquals(3L, queryLong(connection, "SELECT COUNT(*) FROM scalars"));
			assertEquals(1L, queryLong(connection, """
					SELECT s.count
					FROM tag_stats s JOIN tags t ON t.id=s.tag_id
					WHERE t.key='loss'
					"""));
			assertEquals("error", queryString(connection,
					"SELECT status FROM tags WHERE key='loss'"));
			assertEquals("tag_step_regression", queryString(connection,
					"SELECT error_code FROM tags WHERE key='loss'"));
			assertEquals(12.0, queryDouble(connection, """
					SELECT s.mean
					FROM tag_stats s JOIN tags t ON t.id=s.tag_id
					WHERE t.key='other'
					"""));
			assertEquals(1L, queryLong(connection, "SELECT COUNT(*) FROM json_lines"));
			assertEquals(
					"{\"type\":\"histogram\",\"tag\":\"weights\",\"step\":5,\"values\":[1,2]}",
					queryString(connection, "SELECT json FROM json_lines"));
		}
		assertEquals(1, countOccurrences(
				output.getAll(),
				"Scalar tag quarantined: run=run-quarantine tag=loss"
						+ " reason=tag_step_regression previousStep=2 step=1"));
	}

	@Test
	@ExtendWith(OutputCaptureExtension.class)
	void invalidValueWarningIsLoggedOnceWhileTheRunRemainsInTheWorkSet(
			CapturedOutput output) throws Exception {
		final Path runDir = tempDir.resolve("run-warning-once");
		Files.createDirectories(runDir);
		final Path jsonl = runDir.resolve("metrics.jsonl");
		Files.writeString(
				jsonl,
				"""
				{"type":"scalar","tag":"loss","step":1,"value":null}
				{"type":"scalar","tag":"loss","step":2,"value":null}
				""",
				StandardCharsets.UTF_8);

		final MetricsCacheDatabase database = new MetricsCacheDatabase();
		final MetricsIngestor ingestor = new MetricsIngestor(database);
		ingestor.ingestBlock(
				"run-warning-once",
				runDir,
				MetricsSource.select(runDir).orElseThrow());
		final String firstGeneration;
		try (ConnectionHandle handle = database.openRead(runDir)) {
			firstGeneration = queryString(
					handle.connection(),
					"SELECT v FROM source_meta WHERE k='generation'");
		}
		Files.writeString(
				jsonl,
				"{\"type\":\"scalar\",\"tag\":\"loss\",\"step\":3,\"value\":null}\n",
				StandardCharsets.UTF_8,
				java.nio.file.StandardOpenOption.APPEND);
		ingestor.ingestBlock(
				"run-warning-once",
				runDir,
				MetricsSource.select(runDir).orElseThrow());
		Files.writeString(
				jsonl,
				"{\"type\":\"scalar\",\"tag\":\"loss\",\"step\":1000,\"value\":null}\n",
				StandardCharsets.UTF_8);
		ingestor.ingestBlock(
				"run-warning-once",
				runDir,
				MetricsSource.select(runDir).orElseThrow());
		try (ConnectionHandle handle = database.openRead(runDir)) {
			assertNotEquals(firstGeneration, queryString(
					handle.connection(),
					"SELECT v FROM source_meta WHERE k='generation'"));
		}

		assertEquals(1, countOccurrences(
				output.getAll(),
				"run=run-warning-once tag=loss reason=null"));
	}

	@Test
	void unterminatedLineIsHeldUntilItsNewlineArrives() throws Exception {
		final Path runDir = tempDir.resolve("run-tail");
		Files.createDirectories(runDir);
		final Path jsonl = runDir.resolve("metrics.jsonl");
		final String committed =
				"{\"type\":\"scalar\",\"tag\":\"loss\",\"step\":1,\"value\":1.0}\n";
		final String pending =
				"{\"type\":\"scalar\",\"tag\":\"loss\",\"step\":2,\"value\":3.0}";
		Files.writeString(jsonl, committed + pending, StandardCharsets.UTF_8);

		final MetricsCacheDatabase database = new MetricsCacheDatabase();
		final MetricsIngestor ingestor = new MetricsIngestor(database);
		ingestor.ingestBlock(
				"run-tail", runDir, MetricsSource.select(runDir).orElseThrow());
		try (ConnectionHandle handle = database.openRead(runDir)) {
			assertEquals(1L, queryLong(handle.connection(), "SELECT COUNT(*) FROM scalars"));
			assertEquals(committed.getBytes(StandardCharsets.UTF_8).length, queryLong(
					handle.connection(),
					"SELECT CAST(v AS INTEGER) FROM source_meta WHERE k='committed_offset'"));
			assertEquals("ready", queryString(
					handle.connection(),
					"SELECT v FROM source_meta WHERE k='state'"));
		}

		final MetricsIngestor.IngestOutcome idle = ingestor.ingestBlock(
				"run-tail", runDir, MetricsSource.select(runDir).orElseThrow());
		assertFalse(idle.didWork());
		assertEquals(IngestState.READY, idle.state());

		Files.writeString(jsonl, "\n", StandardCharsets.UTF_8, java.nio.file.StandardOpenOption.APPEND);
		ingestor.ingestBlock(
				"run-tail", runDir, MetricsSource.select(runDir).orElseThrow());
		try (ConnectionHandle handle = database.openRead(runDir)) {
			assertEquals(2L, queryLong(handle.connection(), "SELECT COUNT(*) FROM scalars"));
			assertEquals("ready", queryString(
					handle.connection(),
					"SELECT v FROM source_meta WHERE k='state'"));
		}
	}

	@Test
	void jsonlGrowthAfterSourceSnapshotIsDeferredWithoutChangingGeneration() throws Exception {
		final Path runDir = tempDir.resolve("run-growing-source");
		Files.createDirectories(runDir);
		final Path jsonl = runDir.resolve("metrics.jsonl");
		Files.writeString(jsonl, scalarLines(0, 1), StandardCharsets.UTF_8);
		final MetricsSource firstSnapshot = MetricsSource.select(runDir).orElseThrow();
		Files.writeString(
				jsonl,
				scalarLines(1, 2),
				StandardCharsets.UTF_8,
				java.nio.file.StandardOpenOption.APPEND);

		final MetricsCacheDatabase database = new MetricsCacheDatabase();
		final MetricsIngestor ingestor = new MetricsIngestor(database);
		ingestor.ingestBlock("run-growing-source", runDir, firstSnapshot);
		final String firstGeneration;
		try (ConnectionHandle handle = database.openRead(runDir)) {
			firstGeneration = queryString(
					handle.connection(),
					"SELECT v FROM source_meta WHERE k='generation'");
			assertEquals(1L, queryLong(handle.connection(), "SELECT COUNT(*) FROM scalars"));
			assertEquals(firstSnapshot.size(), queryLong(
					handle.connection(),
					"SELECT CAST(v AS INTEGER) FROM source_meta WHERE k='committed_offset'"));
		}

		ingestor.ingestBlock(
				"run-growing-source",
				runDir,
				MetricsSource.select(runDir).orElseThrow());
		try (ConnectionHandle handle = database.openRead(runDir)) {
			assertEquals(firstGeneration, queryString(
					handle.connection(),
					"SELECT v FROM source_meta WHERE k='generation'"));
			assertEquals(2L, queryLong(handle.connection(), "SELECT COUNT(*) FROM scalars"));
			assertEquals(Files.size(jsonl), queryLong(
					handle.connection(),
					"SELECT CAST(v AS INTEGER) FROM source_meta WHERE k='committed_offset'"));
		}
	}

	@Test
	void incompleteLodStateIsRestoredFromCommittedRowsAfterRestart() throws Exception {
		final Path runDir = tempDir.resolve("run-lod-resume");
		Files.createDirectories(runDir);
		final Path jsonl = runDir.resolve("metrics.jsonl");
		Files.writeString(jsonl, scalarLines(0, 15), StandardCharsets.UTF_8);

		final MetricsCacheDatabase database = new MetricsCacheDatabase();
		new MetricsIngestor(database).ingestBlock(
				"run-lod-resume", runDir, MetricsSource.select(runDir).orElseThrow());
		final String generation;
		try (ConnectionHandle handle = database.openRead(runDir)) {
			generation = queryString(
					handle.connection(),
					"SELECT v FROM source_meta WHERE k='generation'");
			assertEquals(0L, queryLong(handle.connection(), "SELECT COUNT(*) FROM scalars_lod"));
		}

		Files.writeString(
				jsonl,
				scalarLines(15, 16),
				StandardCharsets.UTF_8,
				java.nio.file.StandardOpenOption.APPEND);
		new MetricsIngestor(database).ingestBlock(
				"run-lod-resume", runDir, MetricsSource.select(runDir).orElseThrow());

		try (ConnectionHandle handle = database.openRead(runDir)) {
			assertEquals(generation, queryString(
					handle.connection(),
					"SELECT v FROM source_meta WHERE k='generation'"));
			assertEquals(1L, queryLong(handle.connection(), """
					SELECT COUNT(*) FROM scalars_lod
					WHERE level=1 AND bucket=0 AND cnt=16
					"""));
			assertEquals(16L, queryLong(handle.connection(), "SELECT count FROM tag_stats"));
			assertNotEquals(0.0, queryDouble(handle.connection(), "SELECT m2 FROM tag_stats"));
		}
	}

	@Test
	void everyCompleteFactorSixteenLevelIsPersisted() throws Exception {
		final Path runDir = tempDir.resolve("run-all-lod-levels");
		Files.createDirectories(runDir);
		Files.writeString(
				runDir.resolve("metrics.jsonl"),
				scalarLines(0, 4096),
				StandardCharsets.UTF_8);

		final MetricsCacheDatabase database = new MetricsCacheDatabase();
		new MetricsIngestor(database).ingestBlock(
				"run-all-lod-levels",
				runDir,
				MetricsSource.select(runDir).orElseThrow());

		try (ConnectionHandle handle = database.openRead(runDir)) {
			assertEquals(256L, queryLong(handle.connection(),
					"SELECT COUNT(*) FROM scalars_lod WHERE level=1"));
			assertEquals(16L, queryLong(handle.connection(),
					"SELECT COUNT(*) FROM scalars_lod WHERE level=2"));
			assertEquals(1L, queryLong(handle.connection(),
					"SELECT COUNT(*) FROM scalars_lod WHERE level=3"));
			assertEquals(0L, queryLong(handle.connection(),
					"SELECT COUNT(*) FROM scalars_lod WHERE level=4"));
		}
	}

	@Test
	void gzipProducesTheSameCommittedDataAsJsonl() throws Exception {
		final String content = scalarLines(0, 32);
		final Path jsonlRun = tempDir.resolve("run-jsonl");
		final Path gzipRun = tempDir.resolve("run-gzip");
		Files.createDirectories(jsonlRun);
		Files.createDirectories(gzipRun);
		Files.writeString(jsonlRun.resolve("metrics.jsonl"), content, StandardCharsets.UTF_8);
		try (GZIPOutputStream output = new GZIPOutputStream(
				Files.newOutputStream(gzipRun.resolve("metrics.jsonl.gz")))) {
			output.write(content.getBytes(StandardCharsets.UTF_8));
		}

		final MetricsCacheDatabase database = new MetricsCacheDatabase();
		ingestUntilStopped(new MetricsIngestor(database), jsonlRun);
		ingestUntilStopped(new MetricsIngestor(database), gzipRun);

		try (ConnectionHandle jsonl = database.openRead(jsonlRun);
				ConnectionHandle gzip = database.openRead(gzipRun)) {
			assertEquals(queryLong(jsonl.connection(), "SELECT COUNT(*) FROM scalars"),
					queryLong(gzip.connection(), "SELECT COUNT(*) FROM scalars"));
			assertEquals(queryLong(jsonl.connection(), "SELECT COUNT(*) FROM scalars_lod"),
					queryLong(gzip.connection(), "SELECT COUNT(*) FROM scalars_lod"));
			assertEquals(queryDouble(jsonl.connection(), "SELECT mean FROM tag_stats"),
					queryDouble(gzip.connection(), "SELECT mean FROM tag_stats"));
			assertEquals(queryDouble(jsonl.connection(), "SELECT m2 FROM tag_stats"),
					queryDouble(gzip.connection(), "SELECT m2 FROM tag_stats"));
			assertEquals("ready", queryString(
					gzip.connection(),
					"SELECT v FROM source_meta WHERE k='state'"));
		}
	}

	@Test
	void gzipSessionContinuesBufferedLinesAfterCompressedOffsetReachesEof()
			throws Exception {
		final Path runDir = tempDir.resolve("run-gzip-buffered-resume");
		Files.createDirectories(runDir);
		final Path gzip = runDir.resolve("metrics.jsonl.gz");
		writeGzip(gzip, scalarLines(0, 5));
		final MetricsSource source = MetricsSource.select(runDir).orElseThrow();
		assertTrue(source.size() < 64 * 1024);

		final MetricsCacheDatabase database = new MetricsCacheDatabase();
		final GzipInputSessions sessions = new GzipInputSessions();
		final MetricsIngestor ingestor = new MetricsIngestor(database, sessions, 2);

		final MetricsIngestor.IngestOutcome first = ingestor.ingestBlock(
				"run-gzip-buffered-resume", runDir, source);
		assertTrue(first.didWork());
		assertEquals(IngestState.CONVERTING, first.state());
		final String generation;
		final long committedOffset;
		try (ConnectionHandle handle = database.openRead(runDir)) {
			generation = queryString(
					handle.connection(),
					"SELECT v FROM source_meta WHERE k='generation'");
			committedOffset = queryLong(
					handle.connection(),
					"SELECT CAST(v AS INTEGER) FROM source_meta WHERE k='committed_offset'");
			assertEquals(2L, queryLong(handle.connection(), "SELECT COUNT(*) FROM scalars"));
			assertEquals(source.size(), committedOffset);
		}
		assertTrue(sessions.hasSession(runDir, source));

		final MetricsIngestor.IngestOutcome second = ingestor.ingestBlock(
				"run-gzip-buffered-resume", runDir, source);
		assertFalse(second.didWork());
		assertEquals(IngestState.CONVERTING, second.state());
		try (ConnectionHandle handle = database.openRead(runDir)) {
			assertEquals(generation, queryString(
					handle.connection(),
					"SELECT v FROM source_meta WHERE k='generation'"));
			assertEquals(committedOffset, queryLong(
					handle.connection(),
					"SELECT CAST(v AS INTEGER) FROM source_meta WHERE k='committed_offset'"));
			assertEquals(4L, queryLong(handle.connection(), "SELECT COUNT(*) FROM scalars"));
		}
		assertTrue(sessions.hasSession(runDir, source));

		final MetricsIngestor.IngestOutcome third = ingestor.ingestBlock(
				"run-gzip-buffered-resume", runDir, source);
		assertTrue(third.didWork());
		assertEquals(IngestState.READY, third.state());
		try (ConnectionHandle handle = database.openRead(runDir)) {
			assertEquals(generation, queryString(
					handle.connection(),
					"SELECT v FROM source_meta WHERE k='generation'"));
			assertEquals(committedOffset, queryLong(
					handle.connection(),
					"SELECT CAST(v AS INTEGER) FROM source_meta WHERE k='committed_offset'"));
			assertEquals(5L, queryLong(handle.connection(), "SELECT COUNT(*) FROM scalars"));
		}
		assertFalse(sessions.hasSession(runDir, source));
	}

	@Test
	void interruptedGzipConversionRestartsWithANewGeneration() throws Exception {
		final Path runDir = tempDir.resolve("run-gzip-restart");
		Files.createDirectories(runDir);
		writeGzip(runDir.resolve("metrics.jsonl.gz"), scalarLines(0, 5));

		final MetricsCacheDatabase database = new MetricsCacheDatabase();
		final GzipInputSessions firstSessions = new GzipInputSessions();
		final MetricsIngestor firstProcess = new MetricsIngestor(database, firstSessions, 2);
		assertEquals(IngestState.CONVERTING, firstProcess.ingestBlock(
				"run-gzip-restart",
				runDir,
				MetricsSource.select(runDir).orElseThrow()).state());
		final String interruptedGeneration;
		try (ConnectionHandle handle = database.openRead(runDir)) {
			interruptedGeneration = queryString(
					handle.connection(),
					"SELECT v FROM source_meta WHERE k='generation'");
			assertEquals(2L, queryLong(handle.connection(), "SELECT COUNT(*) FROM scalars"));
		}
		firstSessions.closeAll();

		final GzipInputSessions restartedSessions = new GzipInputSessions();
		final MetricsIngestor restartedProcess =
				new MetricsIngestor(database, restartedSessions, 2);
		assertEquals(IngestState.CONVERTING, restartedProcess.ingestBlock(
				"run-gzip-restart",
				runDir,
				MetricsSource.select(runDir).orElseThrow()).state());
		try (ConnectionHandle handle = database.openRead(runDir)) {
			assertNotEquals(interruptedGeneration, queryString(
					handle.connection(),
					"SELECT v FROM source_meta WHERE k='generation'"));
			assertEquals(2L, queryLong(handle.connection(), "SELECT COUNT(*) FROM scalars"));
		}
		ingestUntilStopped(restartedProcess, runDir);
		try (ConnectionHandle handle = database.openRead(runDir)) {
			assertEquals(5L, queryLong(handle.connection(), "SELECT COUNT(*) FROM scalars"));
			assertEquals("ready", queryString(
					handle.connection(),
					"SELECT v FROM source_meta WHERE k='state'"));
		}
	}

	@Test
	void corruptGzipKeepsCommittedBlocksAndIsNotRetriedUntilTheSourceChanges()
			throws Exception {
		final Path runDir = tempDir.resolve("run-gzip-corrupt");
		Files.createDirectories(runDir);
		final Path gzip = runDir.resolve("metrics.jsonl.gz");
		writeGzip(gzip, scalarLines(0, 5));
		final byte[] complete = Files.readAllBytes(gzip);
		Files.write(gzip, Arrays.copyOf(complete, complete.length - 4));

		final MetricsCacheDatabase database = new MetricsCacheDatabase();
		final MetricsIngestor ingestor =
				new MetricsIngestor(database, new GzipInputSessions(), 2);
		ingestUntilStopped(ingestor, runDir);

		final String generation;
		final long committedCount;
		try (ConnectionHandle handle = database.openRead(runDir)) {
			generation = queryString(
					handle.connection(),
					"SELECT v FROM source_meta WHERE k='generation'");
			committedCount = queryLong(handle.connection(), "SELECT COUNT(*) FROM scalars");
			assertEquals(4L, committedCount);
			assertEquals("error", queryString(
					handle.connection(),
					"SELECT v FROM source_meta WHERE k='state'"));
			assertEquals("gzip_corrupt", queryString(
					handle.connection(),
					"SELECT v FROM source_meta WHERE k='error_code'"));
		}

		final MetricsIngestor.IngestOutcome retry = ingestor.ingestBlock(
				"run-gzip-corrupt",
				runDir,
				MetricsSource.select(runDir).orElseThrow());
		assertEquals(false, retry.didWork());
		try (ConnectionHandle handle = database.openRead(runDir)) {
			assertEquals(generation, queryString(
					handle.connection(),
					"SELECT v FROM source_meta WHERE k='generation'"));
			assertEquals(committedCount, queryLong(
					handle.connection(),
					"SELECT COUNT(*) FROM scalars"));
		}

		writeGzip(gzip, scalarLines(0, 5));
		ingestUntilStopped(ingestor, runDir);
		try (ConnectionHandle handle = database.openRead(runDir)) {
			assertNotEquals(generation, queryString(
					handle.connection(),
					"SELECT v FROM source_meta WHERE k='generation'"));
			assertEquals(5L, queryLong(handle.connection(), "SELECT COUNT(*) FROM scalars"));
			assertEquals("ready", queryString(
					handle.connection(),
					"SELECT v FROM source_meta WHERE k='state'"));
		}
	}

	private static void ingestUntilStopped(MetricsIngestor ingestor, Path runDir) throws Exception {
		for (int i = 0; i < 10; i++) {
			final MetricsIngestor.IngestOutcome outcome = ingestor.ingestBlock(
					runDir.getFileName().toString(),
					runDir,
					MetricsSource.select(runDir).orElseThrow());
			if (outcome.state() == IngestState.READY || outcome.state() == IngestState.ERROR) return;
		}
		throw new AssertionError("Ingest did not stop");
	}

	private static void writeGzip(Path path, String content) throws Exception {
		try (GZIPOutputStream output = new GZIPOutputStream(Files.newOutputStream(path))) {
			output.write(content.getBytes(StandardCharsets.UTF_8));
		}
	}

	private static String scalarLines(int fromInclusive, int toExclusive) {
		final StringBuilder lines = new StringBuilder();
		for (int step = fromInclusive; step < toExclusive; step++) {
			lines.append("{\"type\":\"scalar\",\"tag\":\"loss\",\"step\":")
					.append(step)
					.append(",\"value\":")
					.append(step)
					.append(".0}\n");
		}
		return lines.toString();
	}

	private static long queryLong(Connection connection, String sql) throws Exception {
		try (Statement statement = connection.createStatement();
				ResultSet result = statement.executeQuery(sql)) {
			result.next();
			return result.getLong(1);
		}
	}

	private static String queryString(Connection connection, String sql) throws Exception {
		try (Statement statement = connection.createStatement();
				ResultSet result = statement.executeQuery(sql)) {
			result.next();
			return result.getString(1);
		}
	}

	private static double queryDouble(Connection connection, String sql) throws Exception {
		try (Statement statement = connection.createStatement();
				ResultSet result = statement.executeQuery(sql)) {
			result.next();
			return result.getDouble(1);
		}
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

	private MetricsRepository createRepository(MetricsCacheDatabase database) {
		final MetricsViewerSettings settings = new MetricsViewerSettings(3, 100, 0, 1);
		return new MetricsRepository(
				new RunScanner(tempDir.toString()),
				database,
				settings,
				new MetricsRangeProjector(new LodPageCache(settings)));
	}

	private static final class FailSecondWriteDatabase extends MetricsCacheDatabase {
		private final AtomicInteger writeCount = new AtomicInteger();

		@Override
		public ConnectionHandle openWrite(Path runDir) throws SQLException {
			if (writeCount.incrementAndGet() == 2) {
				throw new SQLException("Injected Metrics cache write failure");
			}
			return super.openWrite(runDir);
		}
	}

	private static class CountingDatabase extends MetricsCacheDatabase {
		private final AtomicInteger prepareCount = new AtomicInteger();
		private final AtomicInteger openWriteCount = new AtomicInteger();

		@Override
		public CachePreparation prepare(
				Path runDir,
				MetricsSource source,
				boolean activeGzipSession) throws java.io.IOException, SQLException {
			prepareCount.incrementAndGet();
			return super.prepare(runDir, source, activeGzipSession);
		}

		@Override
		public ConnectionHandle openWrite(Path runDir) throws SQLException {
			openWriteCount.incrementAndGet();
			return super.openWrite(runDir);
		}

		void resetCounts() {
			prepareCount.set(0);
			openWriteCount.set(0);
		}

		int prepareCount() {
			return prepareCount.get();
		}

		int openWriteCount() {
			return openWriteCount.get();
		}
	}

	private static final class FailErrorPersistenceDatabase extends CountingDatabase {
		private final AtomicInteger totalOpenWriteCount = new AtomicInteger();

		@Override
		public ConnectionHandle openWrite(Path runDir) throws SQLException {
			if (totalOpenWriteCount.incrementAndGet() == 2) {
				throw new SQLException("Injected error persistence failure");
			}
			return super.openWrite(runDir);
		}
	}
}
