package io.github.kazukin123.anetlab.metricsviewer.infra;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

@Component
public class MetricsCacheDatabase {

	public static final String DATABASE_FILENAME = "metrics_cache.db";
	public static final int APPLICATION_ID = 0x414E4554;
	public static final int SCHEMA_VERSION = 1;

	private static final Logger log = LoggerFactory.getLogger(MetricsCacheDatabase.class);
	private static final Set<String> REQUIRED_TABLES = Set.of(
			"tags", "scalars", "scalars_lod", "tag_stats", "json_lines", "source_meta");
	private static final Map<String, Set<String>> REQUIRED_COLUMNS = Map.of(
			"tags", Set.of(
					"id", "key", "type", "status", "error_code", "error_message",
					"error_source_offset", "error_previous_step", "error_step"),
			"scalars", Set.of("tag_id", "ordinal", "step", "value"),
			"scalars_lod", Set.of(
					"tag_id", "level", "bucket", "cnt", "step_first", "step_last",
					"min_ordinal", "min_step", "vmin",
					"max_ordinal", "max_step", "vmax", "vmean", "vlast"),
			"tag_stats", Set.of(
					"tag_id", "count", "mean", "m2", "min_value", "max_value",
					"min_step", "max_step", "last_value"),
			"json_lines", Set.of("ordinal", "type", "tag", "step", "timestamp", "json"),
			"source_meta", Set.of("k", "v"));

	private final Map<Path, ReentrantReadWriteLock> lifecycleLocks = new ConcurrentHashMap<>();
	private final Set<Path> deepValidatedDatabases = ConcurrentHashMap.newKeySet();
	private final DeepDatabaseValidator deepDatabaseValidator;

	public MetricsCacheDatabase() {
		this(MetricsCacheDatabase::passesQuickCheck);
	}

	MetricsCacheDatabase(DeepDatabaseValidator deepDatabaseValidator) {
		this.deepDatabaseValidator = deepDatabaseValidator;
	}

	@FunctionalInterface
	interface DeepDatabaseValidator {
		boolean isValid(Statement statement) throws Exception;
	}

	public enum IngestState {
		PENDING("pending"),
		CONVERTING("converting"),
		READY("ready"),
		ERROR("error");

		private final String externalName;

		IngestState(String externalName) {
			this.externalName = externalName;
		}

		public String externalName() {
			return externalName;
		}

		public boolean isStillIngesting() {
			return this == PENDING || this == CONVERTING;
		}

		public static IngestState fromDb(String value) {
			for (IngestState state : values()) {
				if (state.externalName.equals(value)) return state;
			}
			throw new IllegalArgumentException("Unknown Metrics cache ingest state: " + value);
		}

		public static boolean isValidDbValue(String value) {
			for (IngestState state : values()) {
				if (state.externalName.equals(value)) return true;
			}
			return false;
		}
	}

	public record CachePreparation(
			String generation,
			IngestState state,
			long committedOffset,
			boolean sourceUnchangedError) {
	}

	public record CacheMetadata(
			String generation,
			IngestState state,
			long committedOffset,
			long sourceSize,
			String errorCode,
			String errorMessage) {
	}

	public CachePreparation prepare(Path runDir, MetricsSource source, boolean activeGzipSession)
			throws IOException, SQLException {
		final Path normalizedRunDir = runDir.toAbsolutePath().normalize();
		final Path database = normalizedRunDir.resolve(DATABASE_FILENAME);
		final ReentrantReadWriteLock rwLock = lifecycleLockFor(normalizedRunDir);
		boolean deepValidationFailed = false;
		final Lock readLock = rwLock.readLock();
		readLock.lock();
		try {
			if (Files.exists(database) && !deepValidatedDatabases.contains(database)) {
				final long startedAt = System.nanoTime();
				log.info("Metrics cache validation started: run={} dbBytes={}",
						normalizedRunDir.getFileName(), Files.size(database));
				final boolean valid = isValidDatabase(database, true);
				log.info("Metrics cache validation completed: run={} valid={} durationMs={}",
						normalizedRunDir.getFileName(),
						valid,
						(System.nanoTime() - startedAt) / 1_000_000L);
				if (valid) deepValidatedDatabases.add(database);
				else deepValidationFailed = true;
			}
		} finally {
			readLock.unlock();
		}

		rwLock.writeLock().lock();
		try {
			Files.deleteIfExists(normalizedRunDir.resolve("metrics_cache.kryo"));
			final boolean failedCurrentDeepCheck =
					deepValidationFailed && !deepValidatedDatabases.contains(database);
			if (Files.exists(database)) {
				final String invalidReason;
				if (failedCurrentDeepCheck) {
					final String diagnosedReason = diagnoseInvalidDatabase(database);
					invalidReason = "database_validation_failed".equals(diagnosedReason)
							? "deep_validation_failed"
							: diagnosedReason;
				} else if (!isValidDatabase(database, false)) {
					invalidReason = diagnoseInvalidDatabase(database);
				} else {
					invalidReason = null;
				}
				if (invalidReason != null) {
					logCacheRebuild(
							normalizedRunDir,
							invalidReason,
							readSourceMetaIfAvailable(database),
							source);
					deepValidatedDatabases.remove(database);
					deleteCacheFiles(database);
				}
			}
			if (!Files.exists(database)) {
				initialize(database, source);
			}
			deepValidatedDatabases.add(database);

			Map<String, String> meta;
			try (Connection connection = openConnection(database, false)) {
				meta = readSourceMeta(connection);
			}
			if (!matchesSource(meta, source, activeGzipSession)) {
				logCacheRebuild(
						normalizedRunDir,
						describeSourceMismatch(meta, source, activeGzipSession),
						meta,
						source);
				deepValidatedDatabases.remove(database);
				deleteCacheFiles(database);
				initialize(database, source);
				deepValidatedDatabases.add(database);
				try (Connection connection = openConnection(database, false)) {
					meta = readSourceMeta(connection);
				}
			}

			final IngestState state = IngestState.fromDb(
					meta.getOrDefault("state", IngestState.PENDING.externalName()));
			return new CachePreparation(
					meta.get("generation"),
					state,
					parseLong(meta.get("committed_offset"), 0L),
					state == IngestState.ERROR);
		} finally {
			rwLock.writeLock().unlock();
		}
	}

	public ConnectionHandle openWrite(Path runDir) throws SQLException {
		final Path normalizedRunDir = runDir.toAbsolutePath().normalize();
		// 通常transaction同士はWALへ委ね、DB破棄・再構築だけを排他的にする。
		final Lock lock = lifecycleLockFor(normalizedRunDir).readLock();
		lock.lock();
		try {
			return new ConnectionHandle(
					openConnection(normalizedRunDir.resolve(DATABASE_FILENAME), false), lock);
		} catch (SQLException e) {
			lock.unlock();
			throw e;
		}
	}

	public ConnectionHandle openRead(Path runDir) throws SQLException {
		final Path normalizedRunDir = runDir.toAbsolutePath().normalize();
		final Lock lock = lifecycleLockFor(normalizedRunDir).readLock();
		lock.lock();
		try {
			final Path database = normalizedRunDir.resolve(DATABASE_FILENAME);
			if (!Files.isRegularFile(database)) {
				lock.unlock();
				return null;
			}
			return new ConnectionHandle(openConnection(database, true), lock);
		} catch (SQLException e) {
			lock.unlock();
			throw e;
		}
	}

	private ReentrantReadWriteLock lifecycleLockFor(Path runDir) {
		return lifecycleLocks.computeIfAbsent(runDir, ignored -> new ReentrantReadWriteLock(true));
	}

	private boolean matchesSource(
			Map<String, String> meta,
			MetricsSource source,
			boolean activeGzipSession) throws IOException {
		if (!source.kind().equals(meta.get("source_kind"))) return false;
		try {
			UUID.fromString(meta.get("generation"));
		} catch (Exception e) {
			return false;
		}
		if (!IngestState.isValidDbValue(meta.get("state"))) return false;
		final IngestState state = IngestState.fromDb(meta.get("state"));

		final long storedSize = parseLong(meta.get("source_size"), -1L);
		final long committedOffset = parseLong(meta.get("committed_offset"), -1L);
		if (storedSize < 0L || committedOffset < 0L) return false;
		if (!source.headSha256(storedSize).equals(meta.get("source_head_sha256"))) return false;
		if (!source.sha256Before(committedOffset).equals(meta.get("source_commit_tail_sha256"))) return false;

		if (state == IngestState.ERROR) {
			return source.size() == storedSize
					&& source.modifiedTime() == parseLong(meta.get("source_mtime"), -1L);
		}
		if ("jsonl.gz".equals(source.kind())) {
			if (state == IngestState.CONVERTING && !activeGzipSession) return false;
			return source.size() == storedSize
					&& source.modifiedTime() == parseLong(meta.get("source_mtime"), -1L);
		}

		if (source.size() < committedOffset || source.size() < storedSize) return false;
		return true;
	}

	private String describeSourceMismatch(
			Map<String, String> meta,
			MetricsSource source,
			boolean activeGzipSession) {
		if (!source.kind().equals(meta.get("source_kind"))) return "source_kind_changed";
		try {
			UUID.fromString(meta.get("generation"));
		} catch (Exception e) {
			return "generation_invalid";
		}
		if (!IngestState.isValidDbValue(meta.get("state"))) return "state_invalid";
		final IngestState state = IngestState.fromDb(meta.get("state"));

		final long storedSize = parseLong(meta.get("source_size"), -1L);
		final long committedOffset = parseLong(meta.get("committed_offset"), -1L);
		if (storedSize < 0L || committedOffset < 0L) return "source_metadata_invalid";
		if (source.size() < committedOffset) return "source_truncated_below_committed_offset";
		if (source.size() < storedSize) return "source_truncated_below_previous_size";
		if ("jsonl.gz".equals(source.kind())
				&& state == IngestState.CONVERTING
				&& !activeGzipSession) {
			return "gzip_conversion_session_missing";
		}

		try {
			if (!source.headSha256(storedSize).equals(meta.get("source_head_sha256"))) {
				return "source_head_changed";
			}
			if (!source.sha256Before(committedOffset).equals(
					meta.get("source_commit_tail_sha256"))) {
				return "committed_source_tail_changed";
			}
		} catch (IOException e) {
			return "source_fingerprint_diagnosis_failed";
		}

		final long storedMtime = parseLong(meta.get("source_mtime"), -1L);
		if (state == IngestState.ERROR) {
			if (source.size() != storedSize) return "errored_source_size_changed";
			if (source.modifiedTime() != storedMtime) return "errored_source_mtime_changed";
		}
		if ("jsonl.gz".equals(source.kind())) {
			if (source.size() != storedSize) return "gzip_source_size_changed";
			if (source.modifiedTime() != storedMtime) return "gzip_source_mtime_changed";
		}
		return "source_identity_changed";
	}

	private void initialize(Path database, MetricsSource source) throws IOException, SQLException {
		Files.createDirectories(database.getParent());
		try (Connection connection = openConnection(database, false);
				Statement statement = connection.createStatement()) {
			connection.setAutoCommit(false);
			try {
				statement.execute("PRAGMA application_id = " + APPLICATION_ID);
				statement.execute("""
						CREATE TABLE tags(
						  id INTEGER PRIMARY KEY,
						  key TEXT UNIQUE NOT NULL,
						  type TEXT NOT NULL CHECK(type = 'scalar'),
						  status TEXT NOT NULL CHECK(status IN ('ok', 'error')),
						  error_code TEXT,
						  error_message TEXT,
						  error_source_offset INTEGER,
						  error_previous_step INTEGER,
						  error_step INTEGER
						)
						""");
				statement.execute("""
						CREATE TABLE scalars(
						  tag_id INTEGER NOT NULL,
						  ordinal INTEGER NOT NULL,
						  step INTEGER NOT NULL,
						  value REAL NOT NULL,
						  PRIMARY KEY(tag_id, ordinal)
						) WITHOUT ROWID
						""");
				statement.execute("""
						CREATE TABLE scalars_lod(
						  tag_id INTEGER NOT NULL,
						  level INTEGER NOT NULL,
						  bucket INTEGER NOT NULL,
						  cnt INTEGER NOT NULL,
						  step_first INTEGER NOT NULL,
						  step_last INTEGER NOT NULL,
						  min_ordinal INTEGER NOT NULL,
						  min_step INTEGER NOT NULL,
						  vmin REAL NOT NULL,
						  max_ordinal INTEGER NOT NULL,
						  max_step INTEGER NOT NULL,
						  vmax REAL NOT NULL,
						  vmean REAL NOT NULL,
						  vlast REAL NOT NULL,
						  PRIMARY KEY(tag_id, level, bucket)
						) WITHOUT ROWID
						""");
				statement.execute("""
						CREATE TABLE tag_stats(
						  tag_id INTEGER PRIMARY KEY,
						  count INTEGER NOT NULL,
						  mean REAL NOT NULL,
						  m2 REAL NOT NULL,
						  min_value REAL NOT NULL,
						  max_value REAL NOT NULL,
						  min_step INTEGER NOT NULL,
						  max_step INTEGER NOT NULL,
						  last_value REAL NOT NULL
						) WITHOUT ROWID
						""");
				statement.execute("""
						CREATE TABLE json_lines(
						  ordinal INTEGER PRIMARY KEY,
						  type TEXT NOT NULL,
						  tag TEXT,
						  step INTEGER,
						  timestamp TEXT,
						  json TEXT NOT NULL
						)
						""");
				statement.execute("""
						CREATE TABLE source_meta(
						  k TEXT PRIMARY KEY,
						  v TEXT NOT NULL
						) WITHOUT ROWID
						""");

				writeMeta(statement, "generation", UUID.randomUUID().toString());
				writeMeta(statement, "source_kind", source.kind());
				writeMeta(statement, "source_size", Long.toString(source.size()));
				writeMeta(statement, "source_mtime", Long.toString(source.modifiedTime()));
				writeMeta(statement, "source_head_sha256", source.headSha256());
				writeMeta(statement, "source_commit_tail_sha256", source.sha256Before(0L));
				writeMeta(statement, "committed_offset", "0");
				writeMeta(statement, "state", IngestState.PENDING.externalName());
				statement.execute("PRAGMA user_version = " + SCHEMA_VERSION);
				connection.commit();
			} catch (Exception e) {
				connection.rollback();
				throw e;
			}
		} catch (Exception e) {
			deleteCacheFiles(database);
			if (e instanceof IOException ioException) throw ioException;
			if (e instanceof SQLException sqlException) throw sqlException;
			throw new SQLException("Failed to initialize Metrics cache", e);
		}
	}

	private boolean isValidDatabase(Path database, boolean deepCheck) {
		try (Connection connection = openConnection(database, true);
				Statement statement = connection.createStatement()) {
			if (queryInt(statement, "PRAGMA application_id") != APPLICATION_ID) return false;
			if (queryInt(statement, "PRAGMA user_version") != SCHEMA_VERSION) return false;
			if (deepCheck && !deepDatabaseValidator.isValid(statement)) return false;
			final Set<String> tables = new HashSet<>();
			try (ResultSet result = statement.executeQuery(
					"SELECT name FROM sqlite_master WHERE type='table'")) {
				while (result.next()) tables.add(result.getString(1));
			}
			if (!tables.containsAll(REQUIRED_TABLES)) return false;
			for (Map.Entry<String, Set<String>> requirement : REQUIRED_COLUMNS.entrySet()) {
				final Set<String> columns = new HashSet<>();
				try (ResultSet result = statement.executeQuery(
						"PRAGMA table_info(" + requirement.getKey() + ")")) {
					while (result.next()) columns.add(result.getString("name"));
				}
				if (!columns.containsAll(requirement.getValue())) return false;
			}
			return true;
		} catch (Exception e) {
			return false;
		}
	}

	private String diagnoseInvalidDatabase(Path database) {
		try (Connection connection = openConnection(database, true);
				Statement statement = connection.createStatement()) {
			if (queryInt(statement, "PRAGMA application_id") != APPLICATION_ID) {
				return "application_id_mismatch";
			}
			if (queryInt(statement, "PRAGMA user_version") != SCHEMA_VERSION) {
				return "schema_version_mismatch";
			}
			final Set<String> tables = new HashSet<>();
			try (ResultSet result = statement.executeQuery(
					"SELECT name FROM sqlite_master WHERE type='table'")) {
				while (result.next()) tables.add(result.getString(1));
			}
			if (!tables.containsAll(REQUIRED_TABLES)) return "required_table_missing";
			for (Map.Entry<String, Set<String>> requirement : REQUIRED_COLUMNS.entrySet()) {
				final Set<String> columns = new HashSet<>();
				try (ResultSet result = statement.executeQuery(
						"PRAGMA table_info(" + requirement.getKey() + ")")) {
					while (result.next()) columns.add(result.getString("name"));
				}
				if (!columns.containsAll(requirement.getValue())) {
					return "required_column_missing";
				}
			}
			return "database_validation_failed";
		} catch (Exception e) {
			return "database_open_failed";
		}
	}

	private static boolean passesQuickCheck(Statement statement) throws SQLException {
		try (ResultSet result = statement.executeQuery("PRAGMA quick_check")) {
			return result.next() && "ok".equalsIgnoreCase(result.getString(1));
		}
	}

	private Connection openConnection(Path database, boolean readOnly) throws SQLException {
		final Connection connection = DriverManager.getConnection(
				"jdbc:sqlite:" + database.toAbsolutePath().normalize());
		try {
			try (Statement statement = connection.createStatement()) {
				statement.execute("PRAGMA busy_timeout = 5000");
				if (readOnly) {
					statement.execute("PRAGMA query_only = ON");
				} else {
					statement.execute("PRAGMA journal_mode = WAL");
					statement.execute("PRAGMA synchronous = NORMAL");
				}
			}
			return connection;
		} catch (SQLException | RuntimeException e) {
			try {
				connection.close();
			} catch (SQLException closeError) {
				e.addSuppressed(closeError);
			}
			throw e;
		}
	}

	private static int queryInt(Statement statement, String sql) throws SQLException {
		try (ResultSet result = statement.executeQuery(sql)) {
			return result.next() ? result.getInt(1) : Integer.MIN_VALUE;
		}
	}

	private static Map<String, String> readSourceMeta(Connection connection) throws SQLException {
		final Map<String, String> meta = new HashMap<>();
		try (Statement statement = connection.createStatement();
				ResultSet result = statement.executeQuery("SELECT k, v FROM source_meta")) {
			while (result.next()) meta.put(result.getString(1), result.getString(2));
		}
		return meta;
	}

	private Map<String, String> readSourceMetaIfAvailable(Path database) {
		try (Connection connection = openConnection(database, true)) {
			return readSourceMeta(connection);
		} catch (Exception e) {
			return Map.of();
		}
	}

	private static void logCacheRebuild(
			Path runDir,
			String reason,
			Map<String, String> meta,
			MetricsSource source) {
		log.warn(
				"Rebuilding Metrics cache: run={} reason={} oldGeneration={} oldState={}"
						+ " committedOffset={} storedSourceKind={} currentSourceKind={}"
						+ " storedSourceBytes={} currentSourceBytes={}"
						+ " storedSourceMtime={} currentSourceMtime={}",
				runDir.getFileName(),
				reason,
				meta.getOrDefault("generation", "unknown"),
				meta.getOrDefault("state", "unknown"),
				meta.getOrDefault("committed_offset", "unknown"),
				meta.getOrDefault("source_kind", "unknown"),
				source.kind(),
				meta.getOrDefault("source_size", "unknown"),
				source.size(),
				meta.getOrDefault("source_mtime", "unknown"),
				source.modifiedTime());
	}

	private static void writeMeta(Statement statement, String key, String value) throws SQLException {
		final String escapedKey = key.replace("'", "''");
		final String escapedValue = value.replace("'", "''");
		statement.execute("INSERT INTO source_meta(k, v) VALUES('"
				+ escapedKey + "', '" + escapedValue + "')");
	}

	private static long parseLong(String value, long fallback) {
		if (value == null) return fallback;
		try {
			return Long.parseLong(value);
		} catch (NumberFormatException e) {
			return fallback;
		}
	}

	private static void deleteCacheFiles(Path database) throws IOException {
		Files.deleteIfExists(database.resolveSibling(database.getFileName() + "-wal"));
		Files.deleteIfExists(database.resolveSibling(database.getFileName() + "-shm"));
		Files.deleteIfExists(database);
	}

	private static boolean cacheFilesExist(Path database) {
		return Files.exists(database)
				|| Files.exists(database.resolveSibling(database.getFileName() + "-wal"))
				|| Files.exists(database.resolveSibling(database.getFileName() + "-shm"));
	}

	public static final class ConnectionHandle implements AutoCloseable {
		private final Connection connection;
		private final Lock lock;

		private ConnectionHandle(Connection connection, Lock lock) {
			this.connection = connection;
			this.lock = lock;
		}

		public Connection connection() {
			return connection;
		}

		@Override
		public void close() throws SQLException {
			try {
				connection.close();
			} finally {
				lock.unlock();
			}
		}
	}
}
