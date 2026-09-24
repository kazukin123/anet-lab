package io.github.kazukin123.anetlab.metricsviewer.infra;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
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

	/**
	 * externalNameはsource_meta.stateの永続値かつHTTP JSONのingest state公開値であり、
	 * 互換性なく変更しない。
	 */
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
			throw new IllegalArgumentException("Unknown Metrics cache ingest state: " + value
					+ "; expected one of: pending, converting, ready, error");
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

	/**
	 * stateとstateParseErrorは排他的で、既知stateではstateのみ、未知stateでは解析エラーのみを保持する。
	 */
	public record CacheMetadata(
			String generation,
			IngestState state,
			String stateParseError,
			long committedOffset,
			long sourceSize,
			String errorCode,
			String errorMessage) {

		public IngestState stateOrThrow() {
			if (stateParseError != null) throw new IllegalArgumentException(stateParseError);
			return state;
		}
	}

	public CachePreparation prepare(Path runDir, MetricsSource source, boolean activeGzipSession)
			throws IOException, SQLException {
		final Path normalizedRunDir = runDir.toAbsolutePath().normalize();
		final Path database = normalizedRunDir.resolve(DATABASE_FILENAME);
		final ReentrantReadWriteLock rwLock = lifecycleLockFor(normalizedRunDir);

		rwLock.writeLock().lock();
		try {
			Files.deleteIfExists(normalizedRunDir.resolve("metrics_cache.kryo"));
			if (Files.exists(database)) {
				// 起動時にDB全体を走査せず、headerとschemaの軽量検査だけを行う。
				final String invalidReason = checkDatabaseInvalid(database);
				if (invalidReason != null) {
					logCacheRebuild(
							normalizedRunDir,
							invalidReason,
							readSourceMetaIfAvailable(database),
							source);
					deleteCacheFiles(database);
				}
			}
			if (!Files.exists(database)) {
				initialize(database, source);
			}

			SourceMeta.Values meta;
			try (Connection connection = openConnection(database, false)) {
				meta = SourceMeta.readValues(connection);
			}
			final String sourceMismatch = checkSourceMismatch(meta, source, activeGzipSession);
			if (sourceMismatch != null) {
				logCacheRebuild(
						normalizedRunDir,
						sourceMismatch,
						meta,
						source);
				deleteCacheFiles(database);
				initialize(database, source);
				try (Connection connection = openConnection(database, false)) {
					meta = SourceMeta.readValues(connection);
				}
			}

			final IngestState state = IngestState.fromDb(
					meta.getOrDefault(SourceMeta.STATE, IngestState.PENDING.externalName()));
			return new CachePreparation(
					meta.get(SourceMeta.GENERATION),
					state,
					SourceMeta.parseLong(meta.get(SourceMeta.COMMITTED_OFFSET), 0L),
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

	private String checkSourceMismatch(
			SourceMeta.Values meta,
			MetricsSource source,
			boolean activeGzipSession) throws IOException {
		if (!source.kind().equals(meta.get(SourceMeta.SOURCE_KIND))) return "source_kind_changed";
		try {
			UUID.fromString(meta.get(SourceMeta.GENERATION));
		} catch (Exception e) {
			return "generation_invalid";
		}
		if (!IngestState.isValidDbValue(meta.get(SourceMeta.STATE))) return "state_invalid";
		final IngestState state = IngestState.fromDb(meta.get(SourceMeta.STATE));

		final long storedSize = SourceMeta.parseLong(meta.get(SourceMeta.SOURCE_SIZE), -1L);
		final long committedOffset =
				SourceMeta.parseLong(meta.get(SourceMeta.COMMITTED_OFFSET), -1L);
		final long storedMtime = SourceMeta.parseLong(meta.get(SourceMeta.SOURCE_MTIME), -1L);
		if (storedSize < 0L || committedOffset < 0L || storedMtime < 0L) {
			return "source_metadata_invalid";
		}
		if (source.size() < committedOffset) return "source_truncated_below_committed_offset";
		if (source.size() < storedSize) return "source_truncated_below_previous_size";
		if ("jsonl.gz".equals(source.kind())
				&& state == IngestState.CONVERTING
				&& !activeGzipSession) {
			return "gzip_conversion_session_missing";
		}

		// fingerprintのI/O失敗はキャッシュ不一致へ丸めず、呼び出し元へIOExceptionを伝播する。
		if (!source.headSha256(storedSize).equals(meta.get(SourceMeta.SOURCE_HEAD_SHA256))) {
			return "source_head_changed";
		}
		if (!source.sha256Before(committedOffset).equals(
				meta.get(SourceMeta.SOURCE_COMMIT_TAIL_SHA256))) {
			return "committed_source_tail_changed";
		}

		if (state == IngestState.ERROR) {
			if (source.size() != storedSize) return "errored_source_size_changed";
			if (source.modifiedTime() != storedMtime) return "errored_source_mtime_changed";
			return null;
		}
		if ("jsonl.gz".equals(source.kind())) {
			if (source.size() != storedSize) return "gzip_source_size_changed";
			if (source.modifiedTime() != storedMtime) return "gzip_source_mtime_changed";
		}
		return null;
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

				SourceMeta.initialize(connection, source);
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

	private String checkDatabaseInvalid(Path database) {
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
			return null;
		} catch (Exception e) {
			return "database_open_failed";
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

	private SourceMeta.Values readSourceMetaIfAvailable(Path database) {
		try (Connection connection = openConnection(database, true)) {
			return SourceMeta.readValues(connection);
		} catch (Exception e) {
			return SourceMeta.Values.empty();
		}
	}

	private static void logCacheRebuild(
			Path runDir,
			String reason,
			SourceMeta.Values meta,
			MetricsSource source) {
		log.warn(
				"Rebuilding Metrics cache: run={} reason={} oldGeneration={} oldState={}"
						+ " committedOffset={} storedSourceKind={} currentSourceKind={}"
						+ " storedSourceBytes={} currentSourceBytes={}"
						+ " storedSourceMtime={} currentSourceMtime={}",
				runDir.getFileName(),
				reason,
				meta.getOrDefault(SourceMeta.GENERATION, "unknown"),
				meta.getOrDefault(SourceMeta.STATE, "unknown"),
				meta.getOrDefault(SourceMeta.COMMITTED_OFFSET, "unknown"),
				meta.getOrDefault(SourceMeta.SOURCE_KIND, "unknown"),
				source.kind(),
				meta.getOrDefault(SourceMeta.SOURCE_SIZE, "unknown"),
				source.size(),
				meta.getOrDefault(SourceMeta.SOURCE_MTIME, "unknown"),
				source.modifiedTime());
	}

	private static void deleteCacheFiles(Path database) throws IOException {
		Files.deleteIfExists(database.resolveSibling(database.getFileName() + "-wal"));
		Files.deleteIfExists(database.resolveSibling(database.getFileName() + "-shm"));
		Files.deleteIfExists(database);
	}

	public static final class SourceMeta {
		private static final String GENERATION = "generation";
		private static final String SOURCE_KIND = "source_kind";
		private static final String SOURCE_SIZE = "source_size";
		private static final String SOURCE_MTIME = "source_mtime";
		private static final String SOURCE_HEAD_SHA256 = "source_head_sha256";
		private static final String SOURCE_COMMIT_TAIL_SHA256 = "source_commit_tail_sha256";
		private static final String COMMITTED_OFFSET = "committed_offset";
		private static final String STATE = "state";
		private static final String ERROR_CODE = "error_code";
		private static final String ERROR_MESSAGE = "error_message";

		private SourceMeta() {
		}

		public static CacheMetadata read(Connection connection, long numericFallback)
				throws SQLException {
			final Values values = readValues(connection);
			final String stateValue = values.getOrDefault(
					STATE, IngestState.PENDING.externalName());
			IngestState state = null;
			String stateParseError = null;
			try {
				state = IngestState.fromDb(stateValue);
			} catch (IllegalArgumentException e) {
				stateParseError = e.getMessage();
			}
			return new CacheMetadata(
					values.get(GENERATION),
					state,
					stateParseError,
					parseLong(values.get(COMMITTED_OFFSET), numericFallback),
					parseLong(values.get(SOURCE_SIZE), numericFallback),
					values.get(ERROR_CODE),
					values.get(ERROR_MESSAGE));
		}

		public static void initialize(Connection connection, MetricsSource source)
				throws IOException, SQLException {
			try (PreparedStatement statement = upsertStatement(connection)) {
				upsert(statement, GENERATION, UUID.randomUUID().toString());
				upsert(statement, SOURCE_KIND, source.kind());
				upsert(statement, SOURCE_SIZE, Long.toString(source.size()));
				upsert(statement, SOURCE_MTIME, Long.toString(source.modifiedTime()));
				upsert(statement, SOURCE_HEAD_SHA256, source.headSha256());
				upsert(statement, SOURCE_COMMIT_TAIL_SHA256, source.sha256Before(0L));
				upsert(statement, COMMITTED_OFFSET, "0");
				upsert(statement, STATE, IngestState.PENDING.externalName());
			}
		}

		public static void updateProgress(
				Connection connection,
				MetricsSource source,
				long committedOffset,
				IngestState state) throws IOException, SQLException {
			try (PreparedStatement statement = upsertStatement(connection)) {
				upsert(statement, SOURCE_SIZE, Long.toString(source.size()));
				upsert(statement, SOURCE_MTIME, Long.toString(source.modifiedTime()));
				upsert(statement, SOURCE_HEAD_SHA256, source.headSha256());
				upsert(statement, SOURCE_COMMIT_TAIL_SHA256, source.sha256Before(committedOffset));
				upsert(statement, COMMITTED_OFFSET, Long.toString(committedOffset));
				upsert(statement, STATE, state.externalName());
			}
			try (PreparedStatement statement = connection.prepareStatement(
					"DELETE FROM source_meta WHERE k IN (?, ?)")) {
				statement.setString(1, ERROR_CODE);
				statement.setString(2, ERROR_MESSAGE);
				statement.executeUpdate();
			}
		}

		public static void markError(
				Connection connection,
				MetricsSource source,
				String code,
				String message) throws IOException, SQLException {
			try (PreparedStatement statement = upsertStatement(connection)) {
				upsert(statement, SOURCE_SIZE, Long.toString(source.size()));
				upsert(statement, SOURCE_MTIME, Long.toString(source.modifiedTime()));
				upsert(statement, SOURCE_HEAD_SHA256, source.headSha256());
				upsert(statement, STATE, IngestState.ERROR.externalName());
				upsert(statement, ERROR_CODE, code);
				upsert(statement, ERROR_MESSAGE, message == null ? code : message);
			}
		}

		private static PreparedStatement upsertStatement(Connection connection) throws SQLException {
			return connection.prepareStatement("""
					INSERT INTO source_meta(k, v) VALUES(?, ?)
					ON CONFLICT(k) DO UPDATE SET v=excluded.v
					""");
		}

		private static void upsert(PreparedStatement statement, String key, String value)
				throws SQLException {
			statement.setString(1, key);
			statement.setString(2, value);
			statement.executeUpdate();
		}

		private static Values readValues(Connection connection) throws SQLException {
			final Map<String, String> values = new HashMap<>();
			try (PreparedStatement statement = connection.prepareStatement(
					"SELECT k, v FROM source_meta");
					ResultSet result = statement.executeQuery()) {
				while (result.next()) values.put(result.getString(1), result.getString(2));
			}
			return new Values(values);
		}

		private static long parseLong(String value, long fallback) {
			if (value == null) return fallback;
			try {
				return Long.parseLong(value);
			} catch (NumberFormatException e) {
				return fallback;
			}
		}

		private record Values(Map<String, String> entries) {
			private static Values empty() {
				return new Values(Map.of());
			}

			private String get(String key) {
				return entries.get(key);
			}

			private String getOrDefault(String key, String fallback) {
				return entries.getOrDefault(key, fallback);
			}
		}
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
