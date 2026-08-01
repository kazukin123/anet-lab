package io.github.kazukin123.anetlab.metricsviewer.infra;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.FileTime;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.Statement;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.NullSource;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.boot.test.system.CapturedOutput;
import org.springframework.boot.test.system.OutputCaptureExtension;

import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsCacheDatabase.IngestState;
import io.github.kazukin123.anetlab.metricsviewer.service.MetricsIngestor;

class MetricsCacheDatabaseIntegrationTest {

	@TempDir
	private Path tempDir;

	@Test
	void ingestStateKeepsStableExternalNamesAndRejectsUnknownDatabaseValues() {
		for (IngestState state : IngestState.values()) {
			assertTrue(IngestState.isValidDbValue(state.externalName()));
			assertEquals(state, IngestState.fromDb(state.externalName()));
		}
		assertEquals("pending", IngestState.PENDING.externalName());
		assertEquals("converting", IngestState.CONVERTING.externalName());
		assertEquals("ready", IngestState.READY.externalName());
		assertEquals("error", IngestState.ERROR.externalName());
		assertFalse(IngestState.isValidDbValue(null));
		assertFalse(IngestState.isValidDbValue("unknown"));
		assertThrows(IllegalArgumentException.class, () -> IngestState.fromDb(null));
		assertThrows(IllegalArgumentException.class, () -> IngestState.fromDb("unknown"));
		assertTrue(IngestState.PENDING.isStillIngesting());
		assertTrue(IngestState.CONVERTING.isStillIngesting());
		assertFalse(IngestState.READY.isStillIngesting());
		assertFalse(IngestState.ERROR.isStillIngesting());
	}

	@Test
	void schemaWithMissingColumnsIsRebuiltAndOnlyLegacyKryoIsRemoved() throws Exception {
		final Path runDir = tempDir.resolve("run-old-schema");
		Files.createDirectories(runDir);
		Files.writeString(
				runDir.resolve("metrics.jsonl"),
				"{\"type\":\"scalar\",\"tag\":\"loss\",\"step\":1,\"value\":1.0}\n",
				StandardCharsets.UTF_8);
		final Path databasePath = runDir.resolve(MetricsCacheDatabase.DATABASE_FILENAME);
		try (Connection connection = DriverManager.getConnection("jdbc:sqlite:" + databasePath);
				Statement statement = connection.createStatement()) {
			statement.execute("PRAGMA application_id=" + MetricsCacheDatabase.APPLICATION_ID);
			for (String table : Set.of(
					"tags", "scalars", "scalars_lod", "tag_stats", "json_lines", "source_meta")) {
				statement.execute("CREATE TABLE " + table + "(wrong_column TEXT)");
			}
			statement.execute("PRAGMA user_version=" + MetricsCacheDatabase.SCHEMA_VERSION);
		}
		Files.writeString(runDir.resolve("metrics_cache.kryo"), "legacy", StandardCharsets.UTF_8);
		Files.writeString(runDir.resolve("metrics_cache.parquet"), "owned elsewhere", StandardCharsets.UTF_8);
		Files.writeString(runDir.resolve("metrics_cache.db-wal"), "stale", StandardCharsets.UTF_8);
		Files.writeString(runDir.resolve("metrics_cache.db-shm"), "stale", StandardCharsets.UTF_8);

		final MetricsCacheDatabase database = new MetricsCacheDatabase();
		final MetricsCacheDatabase.CachePreparation preparation = database.prepare(
				runDir,
				MetricsSource.select(runDir).orElseThrow(),
				false);

		assertNotNull(preparation.generation());
		assertFalse(Files.exists(runDir.resolve("metrics_cache.kryo")));
		assertTrue(Files.exists(runDir.resolve("metrics_cache.parquet")));
		try (MetricsCacheDatabase.ConnectionHandle handle = database.openRead(runDir);
				Statement statement = handle.connection().createStatement()) {
			assertEquals(MetricsCacheDatabase.APPLICATION_ID, queryInt(statement, "PRAGMA application_id"));
			assertEquals(MetricsCacheDatabase.SCHEMA_VERSION, queryInt(statement, "PRAGMA user_version"));
			assertTrue(columnNames(statement, "scalars").containsAll(
					Set.of("tag_id", "ordinal", "step", "value")));
			assertTrue(columnNames(statement, "source_meta").containsAll(Set.of("k", "v")));
		}
	}

	@Test
	@ExtendWith(OutputCaptureExtension.class)
	void generationChangesOnlyWhenTheSourceIdentityRequiresARebuild(CapturedOutput output)
			throws Exception {
		final Path runDir = tempDir.resolve("run-source-identity");
		Files.createDirectories(runDir);
		final Path jsonl = runDir.resolve("metrics.jsonl");
		Files.writeString(jsonl, scalarLine(1, 1.0), StandardCharsets.UTF_8);

		final MetricsCacheDatabase database = new MetricsCacheDatabase();
		final MetricsIngestor ingestor = new MetricsIngestor(database);
		ingest(ingestor, runDir);
		final String initialGeneration = generation(database, runDir);

		Files.writeString(
				jsonl,
				scalarLine(2, 2.0),
				StandardCharsets.UTF_8,
				java.nio.file.StandardOpenOption.APPEND);
		ingest(ingestor, runDir);
		assertEquals(initialGeneration, generation(database, runDir));
		assertFalse(output.getAll().contains("Rebuilding Metrics cache: run=run-source-identity"));

		Files.setLastModifiedTime(
				jsonl,
				FileTime.fromMillis(Files.getLastModifiedTime(jsonl).toMillis() + 2_000L));
		ingest(ingestor, runDir);
		assertEquals(initialGeneration, generation(database, runDir));
		assertFalse(output.getAll().contains("reason=source_mtime_changed_without_growth"));

		Files.writeString(jsonl, scalarLine(9, 9.0), StandardCharsets.UTF_8);
		ingest(ingestor, runDir);
		final String overwrittenGeneration = generation(database, runDir);
		assertNotEquals(initialGeneration, overwrittenGeneration);
		assertTrue(output.getAll().contains(
				"Rebuilding Metrics cache: run=run-source-identity"
						+ " reason=source_truncated_below_committed_offset"));

		Files.writeString(
				jsonl,
				scalarLine(4, 4.0) + scalarLine(5, 5.0) + scalarLine(6, 6.0),
				StandardCharsets.UTF_8);
		ingest(ingestor, runDir);
		final String replacedGeneration = generation(database, runDir);
		assertNotEquals(overwrittenGeneration, replacedGeneration);

		Files.writeString(jsonl, "", StandardCharsets.UTF_8);
		ingest(ingestor, runDir);
		final String truncatedGeneration = generation(database, runDir);
		assertNotEquals(replacedGeneration, truncatedGeneration);

		Files.delete(jsonl);
		Files.write(runDir.resolve("metrics.jsonl.gz"), new byte[] {31, -117, 8, 0});
		final MetricsCacheDatabase.CachePreparation gzipPreparation = database.prepare(
				runDir,
				MetricsSource.select(runDir).orElseThrow(),
				false);
		assertNotEquals(truncatedGeneration, gzipPreparation.generation());
	}

	@Test
	@ExtendWith(OutputCaptureExtension.class)
	void corruptDatabaseAndInvalidGenerationMetadataAreRebuilt(CapturedOutput output)
			throws Exception {
		final Path runDir = tempDir.resolve("run-corrupt-cache");
		Files.createDirectories(runDir);
		Files.writeString(
				runDir.resolve("metrics.jsonl"),
				scalarLine(1, 1.0),
				StandardCharsets.UTF_8);
		final Path databasePath = runDir.resolve(MetricsCacheDatabase.DATABASE_FILENAME);
		Files.write(databasePath, new byte[] {1, 2, 3, 4, 5});

		final MetricsCacheDatabase database = new MetricsCacheDatabase();
		final MetricsSource source = MetricsSource.select(runDir).orElseThrow();
		final String firstGeneration = database.prepare(runDir, source, false).generation();
		assertNotNull(firstGeneration);
		assertTrue(output.getAll().contains(
				"Rebuilding Metrics cache: run=run-corrupt-cache reason=database_open_failed"));

		try (MetricsCacheDatabase.ConnectionHandle handle = database.openWrite(runDir);
				Statement statement = handle.connection().createStatement()) {
			statement.execute("""
					UPDATE source_meta
					SET v='not-a-uuid'
					WHERE k='generation'
					""");
		}
		final String rebuiltGeneration = database.prepare(runDir, source, false).generation();

		assertNotEquals(firstGeneration, rebuiltGeneration);
		assertNotNull(java.util.UUID.fromString(rebuiltGeneration));
		assertTrue(output.getAll().contains(
				"Rebuilding Metrics cache: run=run-corrupt-cache reason=generation_invalid"));

		try (MetricsCacheDatabase.ConnectionHandle handle = database.openWrite(runDir);
				Statement statement = handle.connection().createStatement()) {
			statement.execute("UPDATE source_meta SET v='unknown' WHERE k='state'");
		}
		final String invalidStateGeneration = database.prepare(runDir, source, false).generation();
		assertNotEquals(rebuiltGeneration, invalidStateGeneration);
		assertTrue(output.getAll().contains(
				"Rebuilding Metrics cache: run=run-corrupt-cache reason=state_invalid"));

		try (MetricsCacheDatabase.ConnectionHandle handle = database.openWrite(runDir);
				Statement statement = handle.connection().createStatement()) {
			statement.execute("DROP TABLE scalars");
		}
		final String schemaRebuiltGeneration = database.prepare(runDir, source, false).generation();
		assertNotEquals(invalidStateGeneration, schemaRebuiltGeneration);
		assertTrue(output.getAll().contains(
				"Rebuilding Metrics cache: run=run-corrupt-cache reason=required_table_missing"));
	}

	@Test
	@ExtendWith(OutputCaptureExtension.class)
	void failedDeepValidationRebuildsWithStableReason(CapturedOutput output) throws Exception {
		final Path runDir = tempDir.resolve("run-deep-validation-failure");
		Files.createDirectories(runDir);
		Files.writeString(
				runDir.resolve("metrics.jsonl"),
				scalarLine(1, 1.0),
				StandardCharsets.UTF_8);
		final MetricsSource source = MetricsSource.select(runDir).orElseThrow();
		final String initialGeneration =
				new MetricsCacheDatabase().prepare(runDir, source, false).generation();

		final MetricsCacheDatabase failingDatabase = new MetricsCacheDatabase(statement -> false);
		final String rebuiltGeneration = failingDatabase.prepare(runDir, source, false).generation();

		assertNotEquals(initialGeneration, rebuiltGeneration);
		assertTrue(output.getAll().contains(
				"Rebuilding Metrics cache: run=run-deep-validation-failure"
						+ " reason=deep_validation_failed"));
	}

	@Test
	@ExtendWith(OutputCaptureExtension.class)
	void invalidCommittedOffsetMetadataRebuildsWithStableReason(CapturedOutput output)
			throws Exception {
		final Path runDir = tempDir.resolve("run-invalid-committed-offset");
		Files.createDirectories(runDir);
		Files.writeString(
				runDir.resolve("metrics.jsonl"),
				scalarLine(1, 1.0),
				StandardCharsets.UTF_8);

		final MetricsCacheDatabase database = new MetricsCacheDatabase();
		final MetricsIngestor ingestor = new MetricsIngestor(database);
		ingest(ingestor, runDir);
		final String initialGeneration = generation(database, runDir);
		try (MetricsCacheDatabase.ConnectionHandle handle = database.openWrite(runDir);
				Statement statement = handle.connection().createStatement()) {
			statement.execute(
					"UPDATE source_meta SET v='not-a-number' WHERE k='committed_offset'");
		}

		final String rebuiltGeneration = database.prepare(
				runDir,
				MetricsSource.select(runDir).orElseThrow(),
				false).generation();

		assertNotEquals(initialGeneration, rebuiltGeneration);
		assertTrue(output.getAll().contains(
				"Rebuilding Metrics cache: run=run-invalid-committed-offset"
						+ " reason=source_metadata_invalid"));
	}

	@ParameterizedTest
	@NullSource
	@ValueSource(strings = "not-a-number")
	@ExtendWith(OutputCaptureExtension.class)
	void invalidRawSourceMtimeMetadataRebuildsWithStableReason(
			String invalidMtime,
			CapturedOutput output) throws Exception {
		final Path runDir = tempDir.resolve("run-invalid-source-mtime");
		Files.createDirectories(runDir);
		Files.writeString(
				runDir.resolve("metrics.jsonl"),
				scalarLine(1, 1.0),
				StandardCharsets.UTF_8);

		final MetricsCacheDatabase database = new MetricsCacheDatabase();
		final MetricsIngestor ingestor = new MetricsIngestor(database);
		ingest(ingestor, runDir);
		final String initialGeneration = generation(database, runDir);
		try (MetricsCacheDatabase.ConnectionHandle handle = database.openWrite(runDir);
				Statement statement = handle.connection().createStatement()) {
			if (invalidMtime == null) {
				statement.execute("DELETE FROM source_meta WHERE k='source_mtime'");
			} else {
				statement.execute(
						"UPDATE source_meta SET v='not-a-number' WHERE k='source_mtime'");
			}
		}

		final String rebuiltGeneration = database.prepare(
				runDir,
				MetricsSource.select(runDir).orElseThrow(),
				false).generation();

		assertNotEquals(initialGeneration, rebuiltGeneration);
		assertTrue(output.getAll().contains(
				"Rebuilding Metrics cache: run=run-invalid-source-mtime"
						+ " reason=source_metadata_invalid"));
	}

	@Test
	void startupDeepValidationDoesNotBlockReadSnapshots() throws Exception {
		final Path runDir = tempDir.resolve("run-deep-validation-concurrency");
		Files.createDirectories(runDir);
		Files.writeString(
				runDir.resolve("metrics.jsonl"),
				scalarLine(1, 1.0),
				StandardCharsets.UTF_8);
		final MetricsSource source = MetricsSource.select(runDir).orElseThrow();
		new MetricsCacheDatabase().prepare(runDir, source, false);

		final CountDownLatch validationStarted = new CountDownLatch(1);
		final CountDownLatch releaseValidation = new CountDownLatch(1);
		final MetricsCacheDatabase database = new MetricsCacheDatabase(statement -> {
			validationStarted.countDown();
			if (!releaseValidation.await(5, TimeUnit.SECONDS)) {
				throw new IllegalStateException("Timed out waiting to release deep validation");
			}
			return true;
		});
		final ExecutorService executor = Executors.newFixedThreadPool(2);
		try {
			final Future<MetricsCacheDatabase.CachePreparation> preparation =
					executor.submit(() -> database.prepare(runDir, source, false));
			assertTrue(validationStarted.await(1, TimeUnit.SECONDS));
			final Future<String> state = executor.submit(() -> {
				try (MetricsCacheDatabase.ConnectionHandle reader = database.openRead(runDir);
						Statement statement = reader.connection().createStatement();
						ResultSet result = statement.executeQuery(
								"SELECT v FROM source_meta WHERE k='state'")) {
					result.next();
					return result.getString(1);
				}
			});
			try {
				assertEquals("pending", state.get(1, TimeUnit.SECONDS));
			} finally {
				releaseValidation.countDown();
			}
			assertEquals(IngestState.PENDING, preparation.get(2, TimeUnit.SECONDS).state());
		} finally {
			releaseValidation.countDown();
			executor.shutdownNow();
		}
	}

	@Test
	void readSnapshotRemainsAvailableWhileAWriteTransactionIsOpen() throws Exception {
		final Path runDir = tempDir.resolve("run-wal-concurrency");
		Files.createDirectories(runDir);
		Files.writeString(
				runDir.resolve("metrics.jsonl"),
				scalarLine(1, 1.0),
				StandardCharsets.UTF_8);

		final MetricsCacheDatabase database = new MetricsCacheDatabase();
		database.prepare(runDir, MetricsSource.select(runDir).orElseThrow(), false);
		final ExecutorService executor = Executors.newSingleThreadExecutor();
		try (MetricsCacheDatabase.ConnectionHandle writer = database.openWrite(runDir)) {
			writer.connection().setAutoCommit(false);
			try (Statement statement = writer.connection().createStatement()) {
				statement.executeUpdate(
						"INSERT INTO json_lines(type, json) VALUES('probe', '{}')");
			}

			final Future<Long> readCount = executor.submit(() -> {
				try (MetricsCacheDatabase.ConnectionHandle reader = database.openRead(runDir);
						Statement statement = reader.connection().createStatement();
						ResultSet result = statement.executeQuery("SELECT COUNT(*) FROM json_lines")) {
					result.next();
					return result.getLong(1);
				}
			});
			assertEquals(0L, readCount.get(1, TimeUnit.SECONDS));
			writer.connection().rollback();
		} finally {
			executor.shutdownNow();
		}
	}

	private static void ingest(MetricsIngestor ingestor, Path runDir) throws Exception {
		ingestor.ingestBlock(
				runDir.getFileName().toString(),
				runDir,
				MetricsSource.select(runDir).orElseThrow());
	}

	private static String generation(MetricsCacheDatabase database, Path runDir) throws Exception {
		try (MetricsCacheDatabase.ConnectionHandle handle = database.openRead(runDir);
				Statement statement = handle.connection().createStatement();
				ResultSet result = statement.executeQuery(
						"SELECT v FROM source_meta WHERE k='generation'")) {
			result.next();
			return result.getString(1);
		}
	}

	private static String scalarLine(long step, double value) {
		return "{\"type\":\"scalar\",\"tag\":\"loss\",\"step\":"
				+ step + ",\"value\":" + value + "}\n";
	}

	private static int queryInt(Statement statement, String sql) throws Exception {
		try (ResultSet result = statement.executeQuery(sql)) {
			result.next();
			return result.getInt(1);
		}
	}

	private static Set<String> columnNames(Statement statement, String table) throws Exception {
		final Set<String> columns = new HashSet<>();
		try (ResultSet result = statement.executeQuery("PRAGMA table_info(" + table + ")")) {
			while (result.next()) columns.add(result.getString("name"));
		}
		return columns;
	}
}
