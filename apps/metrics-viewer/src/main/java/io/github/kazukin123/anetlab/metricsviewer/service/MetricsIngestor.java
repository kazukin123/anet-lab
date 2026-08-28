package io.github.kazukin123.anetlab.metricsviewer.service;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.nio.file.attribute.BasicFileAttributes;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.function.BooleanSupplier;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.fasterxml.jackson.core.json.JsonReadFeature;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.json.JsonMapper;

import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsCacheDatabase;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsCacheDatabase.CachePreparation;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsCacheDatabase.ConnectionHandle;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsCacheDatabase.IngestState;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsCacheDatabase.SourceMeta;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsSource;
import io.github.kazukin123.anetlab.metricsviewer.service.SourceReader.GzipCorruptException;
import io.github.kazukin123.anetlab.metricsviewer.service.SourceReader.ReadResult;

public class MetricsIngestor {

	public static final int MAX_BLOCK_LINES = 1_000_000;
	private static final long MAX_SAFE_INTEGER = 9_007_199_254_740_991L;
	private static final Logger log = LoggerFactory.getLogger(MetricsIngestor.class);
	private static final ObjectMapper OBJECT_MAPPER = JsonMapper.builder()
			.enable(JsonReadFeature.ALLOW_NON_NUMERIC_NUMBERS)
			.build();

	private final MetricsCacheDatabase database;
	private final GzipInputSessions gzipSessions;
	private final LodIngestWriter lodWriter;
	private final RunWarningRegistry warningRegistry;
	private final int maxBlockLines;
	private final BooleanSupplier yieldRequested;
	private final Map<Path, ValidatedRunObservation> validatedObservations = new HashMap<>();

	public record IngestOutcome(
			boolean didWork,
			IngestState state,
			boolean immediateRetry) {
	}

	public MetricsIngestor(MetricsCacheDatabase database) {
		this(
				database,
				new GzipInputSessions(),
				new LodIngestWriter(),
				new RunWarningRegistry(),
				MAX_BLOCK_LINES,
				() -> false);
	}

	public MetricsIngestor(
			MetricsCacheDatabase database,
			GzipInputSessions gzipSessions,
			LodIngestWriter lodWriter,
			RunWarningRegistry warningRegistry) {
		this(database, gzipSessions, lodWriter, warningRegistry, MAX_BLOCK_LINES, () -> false);
	}

	MetricsIngestor(
			MetricsCacheDatabase database,
			GzipInputSessions gzipSessions,
			int maxBlockLines) {
		this(
				database,
				gzipSessions,
				new LodIngestWriter(),
				new RunWarningRegistry(),
				maxBlockLines,
				() -> false);
	}

	MetricsIngestor(
			MetricsCacheDatabase database,
			GzipInputSessions gzipSessions,
			LodIngestWriter lodWriter,
			RunWarningRegistry warningRegistry,
			int maxBlockLines,
			BooleanSupplier yieldRequested) {
		this.database = database;
		this.gzipSessions = gzipSessions;
		this.lodWriter = lodWriter;
		this.warningRegistry = warningRegistry;
		this.maxBlockLines = maxBlockLines;
		this.yieldRequested = yieldRequested;
	}

	void retainRuns(Set<Path> runDirs) {
		final Set<Path> retained = new HashSet<>();
		for (Path runDir : runDirs) retained.add(normalize(runDir));
		validatedObservations.keySet().retainAll(retained);
	}

	void forgetRun(Path runDir) {
		validatedObservations.remove(normalize(runDir));
	}

	public IngestOutcome ingestBlock(String runId, Path runDir, MetricsSource source) throws Exception {
		final Path runKey = normalize(runDir);
		final MetricsSource sourceSnapshot = normalize(source);
		final IngestOutcome fastPath = fastPath(runKey, sourceSnapshot);
		if (fastPath != null) return fastPath;

		final SourceReader reader = SourceReader.forSource(runDir, source, gzipSessions);
		final IngestAttempt attempt;
		try (reader) {
			attempt = ingestWithReader(runId, runDir, source, reader);
		}
		if (attempt.publishObservation()) {
			publishValidatedObservation(runKey, sourceSnapshot, attempt.outcome().state());
		}
		return attempt.outcome();
	}

	private IngestAttempt ingestWithReader(
			String runId,
			Path runDir,
			MetricsSource source,
			SourceReader reader) throws Exception {
		final SourceReader.Preparation sourcePreparation;
		try {
			// ソース同一性を検証し、必要なら破棄可能cacheを先頭から作り直す。
			sourcePreparation = reader.prepare(database);
		} catch (GzipCorruptException e) {
			final boolean persisted =
					markRunError(runDir, source, "gzip_corrupt", e.getMessage());
			reader.fail();
			return new IngestAttempt(
					new IngestOutcome(true, IngestState.ERROR, false), persisted);
		}
		final CachePreparation preparation = sourcePreparation.cache();
		if (!sourcePreparation.readRequired()) {
			final IngestOutcome outcome = new IngestOutcome(
					false,
					preparation.state(),
					preparation.state() == IngestState.CONVERTING);
			return new IngestAttempt(outcome, isTerminal(outcome.state()));
		}

		final long startOffset = preparation.committedOffset();

		try (ConnectionHandle handle = database.openWrite(runDir)) {
			final Connection connection = handle.connection();
			connection.setAutoCommit(false);

			try {
				final ReadResult readResult;
				final BlockWriteSession writeSession = new BlockWriteSession(connection, runId);
				try (writeSession) {
					// 1行ずつparseし、巨大な中間Listを作らず同じtransactionへ反映する。
					readResult = reader.readCompleteLines(
							maxBlockLines,
							yieldRequested,
							writeSession::writeLine);
				}

				// block内のL0、LOD、TagStats、source位置を同じcommit境界で確定する。
				writeSession.flushTagStats();
				reader.validate(readResult);
				final IngestState state = readResult.logicalEof()
						? IngestState.READY
						: IngestState.CONVERTING;
				updateSourceMeta(
						connection,
						source,
						readResult.committedSourceOffset(),
						state);
				connection.commit();

				writeSession.emitWarnings();
				reader.finish(state);
				return new IngestAttempt(
						new IngestOutcome(
								readResult.committedSourceOffset() != startOffset
										|| state != preparation.state(),
								state,
								state == IngestState.CONVERTING),
						state == IngestState.READY);
			} catch (FatalRecordException e) {
				connection.rollback();
				final boolean persisted =
						markRunError(runDir, source, e.code(), e.getMessage());
				reader.fail();
				return new IngestAttempt(
						new IngestOutcome(true, IngestState.ERROR, false), persisted);
			} catch (GzipCorruptException e) {
				connection.rollback();
				final boolean persisted =
						markRunError(runDir, source, "gzip_corrupt", e.getMessage());
				reader.fail();
				return new IngestAttempt(
						new IngestOutcome(true, IngestState.ERROR, false), persisted);
			} catch (IOException e) {
				connection.rollback();
				if (reader.treatsBlockIOExceptionAsCorrupt()) {
					final boolean persisted =
							markRunError(runDir, source, "gzip_corrupt", e.getMessage());
					reader.fail();
					return new IngestAttempt(
							new IngestOutcome(true, IngestState.ERROR, false), persisted);
				}
				throw e;
			} catch (Exception e) {
				connection.rollback();
				throw e;
			}
		} catch (IOException e) {
			// Metricsマスタの読取失敗だけをsource errorとして記録する。
			if (reader.marksSourceReadFailureAsError()) {
				markRunError(runDir, source, "source_read_error", e.getMessage());
			}
			throw e;
		} catch (SQLException e) {
			// 開始済みcache transactionはrollbackされる。読取位置だけが先行しないようsessionを破棄し、
			// pending/convertingを維持したまま次cycleの再試行へ委ねる。
			reader.fail();
			throw e;
		}
	}

	private static boolean isTerminal(IngestState state) {
		return state == IngestState.READY || state == IngestState.ERROR;
	}

	private IngestOutcome fastPath(Path runKey, MetricsSource source) {
		final ValidatedRunObservation observed = validatedObservations.get(runKey);
		if (observed == null) return null;
		try {
			final CacheFileObservation cache = CacheFileObservation.capture(runKey);
			if (observed.matches(source, cache)) {
				return new IngestOutcome(false, observed.state(), false);
			}
		} catch (IOException ignored) {
			// attributeを安定して読めない場合は観測を信用せず、既存の完全検証へ戻る。
		}
		validatedObservations.remove(runKey, observed);
		return null;
	}

	private void publishValidatedObservation(
			Path runKey,
			MetricsSource sourceSnapshot,
			IngestState state) {
		try {
			// resource close後にsource/cacheを再観測し、block中の差替えをfast pathへ持ち越さない。
			final MetricsSource currentSource = MetricsSource.select(runKey)
					.map(MetricsIngestor::normalize)
					.orElse(null);
			if (!sourceSnapshot.equals(currentSource)) {
				validatedObservations.remove(runKey);
				return;
			}
			final CacheFileObservation cache = CacheFileObservation.capture(runKey);
			if (!cache.exists()) {
				validatedObservations.remove(runKey);
				return;
			}
			validatedObservations.put(
					runKey,
					new ValidatedRunObservation(sourceSnapshot, cache, state));
		} catch (IOException ignored) {
			validatedObservations.remove(runKey);
		}
	}

	private static Path normalize(Path path) {
		return path.toAbsolutePath().normalize();
	}

	private static MetricsSource normalize(MetricsSource source) {
		return new MetricsSource(
				normalize(source.path()),
				source.kind(),
				source.size(),
				source.modifiedTime());
	}

	private static ParsedRecord parseRecord(String rawLine, long sourceOffset)
			throws FatalRecordException {
		if (rawLine.isBlank()) {
			throw new FatalRecordException(
					"invalid_json",
					"Blank JSON line at source offset " + sourceOffset);
		}
		final JsonNode root;
		try {
			root = OBJECT_MAPPER.readTree(rawLine);
		} catch (Exception e) {
			throw new FatalRecordException(
					"invalid_json",
					"Invalid JSON at source offset " + sourceOffset + ": " + e.getMessage());
		}
		if (root == null || !root.isObject()) {
			throw new FatalRecordException(
					"invalid_record",
					"Top-level JSON value must be an object at source offset " + sourceOffset);
		}

		final JsonNode typeNode = root.get("type");
		if (typeNode == null || !typeNode.isTextual()) {
			throw new FatalRecordException(
					"invalid_record",
					"Missing or invalid field 'type' at source offset " + sourceOffset);
		}
		final String type = typeNode.textValue();
		final boolean scalar = "scalar".equals(type);
		if (!scalar) {
			return new ParsedRecord(
					false,
					type,
					textOrNull(root.get("tag")),
					optionalStep(root.get("step"), sourceOffset),
					textOrNull(root.get("timestamp")),
					null);
		}

		final JsonNode tagNode = root.get("tag");
		final JsonNode stepNode = root.get("step");
		final JsonNode valueNode = root.get("value");
		if (tagNode == null || !tagNode.isTextual()) {
			throw new FatalRecordException(
					"invalid_record",
					"Missing or invalid field 'tag' at source offset " + sourceOffset);
		}
		if (valueNode == null) {
			throw new FatalRecordException(
					"invalid_record",
					"Missing field 'value' at source offset " + sourceOffset);
		}
		return new ParsedRecord(
				true,
				type,
				tagNode.textValue(),
				requiredStep(stepNode, sourceOffset),
				textOrNull(root.get("timestamp")),
				valueNode);
	}

	private static Long optionalStep(JsonNode stepNode, long sourceOffset)
			throws FatalRecordException {
		if (stepNode == null || stepNode.isNull()) return null;
		return requiredStep(stepNode, sourceOffset);
	}

	private static long requiredStep(JsonNode stepNode, long sourceOffset)
			throws FatalRecordException {
		if (stepNode == null || !stepNode.isIntegralNumber() || !stepNode.canConvertToLong()) {
			throw new FatalRecordException(
					"invalid_step",
					"Missing or invalid field 'step' at source offset " + sourceOffset);
		}
		final long step = stepNode.longValue();
		if (step < -MAX_SAFE_INTEGER || step > MAX_SAFE_INTEGER) {
			throw new FatalRecordException(
					"invalid_step",
					"Unsafe field 'step' at source offset " + sourceOffset + ": " + step);
		}
		return step;
	}

	private static String textOrNull(JsonNode node) {
		return node != null && node.isTextual() ? node.textValue() : null;
	}

	private void writeScalar(
			BlockWriteSession session,
			ParsedRecord record,
			long sourceOffset) throws SQLException {
		final JsonNode valueNode = record.valueNode();
		if (valueNode.isNull()) {
			session.warnings.add(new WarningKey(session.runId, record.tag(), "null"));
			return;
		}
		if (!valueNode.isNumber()) {
			session.warnings.add(new WarningKey(session.runId, record.tag(), "not_numeric"));
			return;
		}
		final double value = valueNode.doubleValue();
		if (!Double.isFinite(value)) {
			session.warnings.add(new WarningKey(session.runId, record.tag(), "non_finite"));
			return;
		}
		if (!Float.isFinite((float) value)) {
			session.warnings.add(new WarningKey(session.runId, record.tag(), "f32_overflow"));
			return;
		}

		TagWriteState state = session.tagStates.get(record.tag());
		if (state == null) {
			state = loadTagState(session.connection, record.tag());
			session.tagStates.put(record.tag(), state);
		}
		if (state.quarantined) return;
		if (state.count > 0L && record.step() < state.previousStep) {
			session.quarantineWarnings.add(new TagQuarantineWarning(
					session.runId,
					record.tag(),
					state.previousStep,
					record.step(),
					sourceOffset));
			quarantineTag(session.connection, state, sourceOffset, record.step());
			state.quarantined = true;
			return;
		}

		final long ordinal = state.nextOrdinal;
		session.insertScalar.setLong(1, state.tagId);
		session.insertScalar.setLong(2, ordinal);
		session.insertScalar.setLong(3, record.step());
		session.insertScalar.setDouble(4, value);
		session.insertScalar.executeUpdate();
		state.record(record.step(), value);
		lodWriter.append(
				session.lodInsert,
				state.tagId,
				state.lodState,
				LodBucket.point(ordinal, record.step(), value));
	}

	private TagWriteState loadTagState(Connection connection, String tag) throws SQLException {
		try (PreparedStatement statement = connection.prepareStatement("""
					SELECT t.id, t.status, s.count, s.mean, s.m2, s.min_value, s.max_value,
					       s.min_step, s.max_step, s.last_value,
					       COALESCE((SELECT MAX(ordinal) + 1 FROM scalars WHERE tag_id=t.id), 0)
					FROM tags t
					LEFT JOIN tag_stats s ON s.tag_id=t.id
					WHERE t.key=?
					""")) {
			statement.setString(1, tag);
			try (ResultSet result = statement.executeQuery()) {
				if (result.next()) {
					final TagWriteState state = TagWriteState.fromResult(result);
					state.lodState = lodWriter.restore(connection, state.tagId, state.count);
					return state;
				}
			}
		}

		try (PreparedStatement statement = connection.prepareStatement(
				"INSERT INTO tags(key, type, status) VALUES(?, 'scalar', 'ok')",
				PreparedStatement.RETURN_GENERATED_KEYS)) {
			statement.setString(1, tag);
			statement.executeUpdate();
			try (ResultSet keys = statement.getGeneratedKeys()) {
				if (!keys.next()) throw new SQLException("Missing generated tag id");
				final TagWriteState state = TagWriteState.empty(keys.getLong(1));
				state.lodState = lodWriter.restore(connection, state.tagId, 0L);
				return state;
			}
		}
	}

	private static void quarantineTag(
			Connection connection,
			TagWriteState state,
			long sourceOffset,
			long step) throws SQLException {
		try (PreparedStatement statement = connection.prepareStatement("""
				UPDATE tags
				SET status='error', error_code='tag_step_regression',
				    error_message=?, error_source_offset=?,
				    error_previous_step=?, error_step=?
				WHERE id=?
				""")) {
			statement.setString(1, "step regressed from " + state.previousStep + " to " + step);
			statement.setLong(2, sourceOffset);
			statement.setLong(3, state.previousStep);
			statement.setLong(4, step);
			statement.setLong(5, state.tagId);
			statement.executeUpdate();
		}
	}

	private static void writeJsonLine(
			PreparedStatement statement,
			ParsedRecord record,
			String rawLine)
			throws SQLException {
		statement.setString(1, record.type());
		statement.setString(2, record.tag());
		if (record.step() == null) statement.setNull(3, java.sql.Types.BIGINT);
		else statement.setLong(3, record.step());
		statement.setString(4, record.timestamp());
		statement.setString(5, rawLine);
		statement.executeUpdate();
	}

	private static void flushTagStats(
			Connection connection,
			Map<String, TagWriteState> states) throws SQLException {
		try (PreparedStatement statement = connection.prepareStatement("""
				INSERT INTO tag_stats(
				  tag_id, count, mean, m2, min_value, max_value, min_step, max_step, last_value
				) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
				ON CONFLICT(tag_id) DO UPDATE SET
				  count=excluded.count, mean=excluded.mean, m2=excluded.m2,
				  min_value=excluded.min_value, max_value=excluded.max_value,
				  min_step=excluded.min_step, max_step=excluded.max_step,
				  last_value=excluded.last_value
				""")) {
			for (TagWriteState state : states.values()) {
				if (!state.dirty) continue;
				statement.setLong(1, state.tagId);
				statement.setLong(2, state.count);
				statement.setDouble(3, state.mean);
				statement.setDouble(4, state.m2);
				statement.setDouble(5, state.minValue);
				statement.setDouble(6, state.maxValue);
				statement.setLong(7, state.minStep);
				statement.setLong(8, state.maxStep);
				statement.setDouble(9, state.lastValue);
				statement.addBatch();
			}
			statement.executeBatch();
		}
	}

	private static void updateSourceMeta(
			Connection connection,
			MetricsSource source,
			long committedOffset,
			IngestState state) throws SQLException, IOException {
		SourceMeta.updateProgress(connection, source, committedOffset, state);
	}

	private boolean markRunError(
			Path runDir,
			MetricsSource source,
			String code,
			String message) {
		try (ConnectionHandle handle = database.openWrite(runDir)) {
			final Connection connection = handle.connection();
			connection.setAutoCommit(false);
			SourceMeta.markError(connection, source, code, message);
			connection.commit();
			log.warn("Run ingest entered error state: run={} code={} message={}",
					runDir.getFileName(), code, message == null ? code : message);
			return true;
		} catch (Exception error) {
			log.error("Failed to persist Run error: run={} code={} message={}",
					runDir.getFileName(), code, error.getMessage());
			return false;
		}
	}

	private final class BlockWriteSession implements AutoCloseable {
		private final Connection connection;
		private final String runId;
		private final Map<String, TagWriteState> tagStates = new HashMap<>();
		private final Set<WarningKey> warnings = new HashSet<>();
		private final Set<TagQuarantineWarning> quarantineWarnings = new HashSet<>();
		private PreparedStatement insertScalar;
		private PreparedStatement insertJson;
		private LodIngestWriter.InsertSession lodInsert;

		private BlockWriteSession(Connection connection, String runId) throws SQLException {
			this.connection = connection;
			this.runId = runId;
			try {
				insertScalar = connection.prepareStatement(
						"INSERT INTO scalars(tag_id, ordinal, step, value) VALUES(?, ?, ?, ?)");
				insertJson = connection.prepareStatement("""
						INSERT INTO json_lines(type, tag, step, timestamp, json)
						VALUES(?, ?, ?, ?, ?)
						""");
				lodInsert = lodWriter.openInsertSession(connection);
			} catch (SQLException e) {
				try {
					close();
				} catch (SQLException closeError) {
					e.addSuppressed(closeError);
				}
				throw e;
			}
		}

		private void writeLine(String line, long sourceOffset) throws Exception {
			final ParsedRecord record = parseRecord(line, sourceOffset);
			if (record.scalar()) {
				writeScalar(this, record, sourceOffset);
			} else {
				writeJsonLine(insertJson, record, line);
			}
		}

		private void flushTagStats() throws SQLException {
			MetricsIngestor.flushTagStats(connection, tagStates);
		}

		private void emitWarnings() {
			for (WarningKey warning : warnings) {
				if (warningRegistry.firstScalarSkip(
						warning.runId(), warning.tag(), warning.reason())) {
					log.warn(
							"Skipping invalid scalar value: run={} tag={} reason={}",
							warning.runId(),
							warning.tag(),
							warning.reason());
				}
			}
			for (TagQuarantineWarning warning : quarantineWarnings) {
				log.warn(
						"Scalar tag quarantined: run={} tag={}"
								+ " reason=tag_step_regression previousStep={} step={}"
								+ " sourceOffset={}",
						warning.runId(),
						warning.tag(),
						warning.previousStep(),
						warning.step(),
						warning.sourceOffset());
			}
		}

		@Override
		public void close() throws SQLException {
			SQLException failure = null;
			failure = closeResource(lodInsert, failure);
			failure = closeResource(insertJson, failure);
			failure = closeResource(insertScalar, failure);
			if (failure != null) throw failure;
		}

		private SQLException closeResource(AutoCloseable resource, SQLException failure) {
			if (resource == null) return failure;
			try {
				resource.close();
			} catch (Exception e) {
				final SQLException closeError = e instanceof SQLException sqlError
						? sqlError
						: new SQLException("Failed to close block write resource", e);
				if (failure == null) return closeError;
				failure.addSuppressed(closeError);
			}
			return failure;
		}
	}

	private record IngestAttempt(IngestOutcome outcome, boolean publishObservation) {
	}

	private record ValidatedRunObservation(
			MetricsSource source,
			CacheFileObservation cache,
			IngestState state) {
		private boolean matches(MetricsSource currentSource, CacheFileObservation currentCache) {
			return source.equals(currentSource) && cache.equals(currentCache);
		}
	}

	private record CacheFileObservation(boolean exists, long size, long modifiedTime) {
		private static CacheFileObservation capture(Path runDir) throws IOException {
			final Path cache = runDir.resolve("metrics_cache.db");
			try {
				final BasicFileAttributes attributes =
						Files.readAttributes(cache, BasicFileAttributes.class);
				return new CacheFileObservation(
						true,
						attributes.size(),
						attributes.lastModifiedTime().toMillis());
			} catch (NoSuchFileException e) {
				return new CacheFileObservation(false, 0L, 0L);
			}
		}
	}

	private record ParsedRecord(
			boolean scalar,
			String type,
			String tag,
			Long step,
			String timestamp,
			JsonNode valueNode) {
	}

	private record WarningKey(String runId, String tag, String reason) {
	}

	private record TagQuarantineWarning(
			String runId,
			String tag,
			long previousStep,
			long step,
			long sourceOffset) {
	}

	private static final class TagWriteState {
		private final long tagId;
		private long nextOrdinal;
		private long count;
		private double mean;
		private double m2;
		private double minValue;
		private double maxValue;
		private long minStep;
		private long maxStep;
		private double lastValue;
		private long previousStep;
		private boolean quarantined;
		private boolean dirty;
		private LodIngestWriter.State lodState;

		private TagWriteState(long tagId) {
			this.tagId = tagId;
		}

		private static TagWriteState empty(long tagId) {
			final TagWriteState state = new TagWriteState(tagId);
			state.minValue = Double.POSITIVE_INFINITY;
			state.maxValue = Double.NEGATIVE_INFINITY;
			return state;
		}

		private static TagWriteState fromResult(ResultSet result) throws SQLException {
			final TagWriteState state = new TagWriteState(result.getLong(1));
			state.quarantined = "error".equals(result.getString(2));
			state.count = result.getLong(3);
			if (!result.wasNull()) {
				state.mean = result.getDouble(4);
				state.m2 = result.getDouble(5);
				state.minValue = result.getDouble(6);
				state.maxValue = result.getDouble(7);
				state.minStep = result.getLong(8);
				state.maxStep = result.getLong(9);
				state.lastValue = result.getDouble(10);
				state.previousStep = state.maxStep;
			} else {
				state.minValue = Double.POSITIVE_INFINITY;
				state.maxValue = Double.NEGATIVE_INFINITY;
			}
			state.nextOrdinal = result.getLong(11);
			return state;
		}

		private void record(long step, double value) {
			final long nextCount = count + 1L;
			final double delta = value - mean;
			mean += delta / nextCount;
			m2 += delta * (value - mean);
			count = nextCount;
			minValue = Math.min(minValue, value);
			maxValue = Math.max(maxValue, value);
			if (count == 1L) {
				minStep = step;
				maxStep = step;
			} else {
				minStep = Math.min(minStep, step);
				maxStep = Math.max(maxStep, step);
			}
			lastValue = value;
			previousStep = step;
			nextOrdinal++;
			dirty = true;
		}

	}

	private static final class FatalRecordException extends Exception {
		private final String code;

		private FatalRecordException(String code, String message) {
			super(message);
			this.code = code;
		}

		private String code() {
			return code;
		}
	}

}
