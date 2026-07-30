package io.github.kazukin123.anetlab.metricsviewer.service;

import java.io.BufferedInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.LongSupplier;

import org.springframework.beans.factory.annotation.Autowired;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import com.fasterxml.jackson.core.json.JsonReadFeature;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.json.JsonMapper;

import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsCacheDatabase;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsCacheDatabase.CachePreparation;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsCacheDatabase.ConnectionHandle;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsSource;

@Component
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
	private final int maxBlockLines;
	private final Set<WarningKey> issuedWarnings = ConcurrentHashMap.newKeySet();

	public record IngestOutcome(boolean didWork, String state) {
	}

	public MetricsIngestor(MetricsCacheDatabase database) {
		this(database, new GzipInputSessions(), new LodIngestWriter(), MAX_BLOCK_LINES);
	}

	@Autowired
	public MetricsIngestor(
			MetricsCacheDatabase database,
			GzipInputSessions gzipSessions,
			LodIngestWriter lodWriter) {
		this(database, gzipSessions, lodWriter, MAX_BLOCK_LINES);
	}

	MetricsIngestor(
			MetricsCacheDatabase database,
			GzipInputSessions gzipSessions,
			int maxBlockLines) {
		this(database, gzipSessions, new LodIngestWriter(), maxBlockLines);
	}

	private MetricsIngestor(
			MetricsCacheDatabase database,
			GzipInputSessions gzipSessions,
			LodIngestWriter lodWriter,
			int maxBlockLines) {
		this.database = database;
		this.gzipSessions = gzipSessions;
		this.lodWriter = lodWriter;
		this.maxBlockLines = maxBlockLines;
	}

	public IngestOutcome ingestBlock(String runId, Path runDir, MetricsSource source) throws Exception {
		// ソース同一性を検証し、必要なら破棄可能cacheを先頭から作り直す。
		final boolean gzip = "jsonl.gz".equals(source.kind());
		final CachePreparation preparation = database.prepare(
				runDir,
				source,
				gzip && gzipSessions.hasSession(runDir, source));
		if (preparation.sourceUnchangedError()) {
			if (gzip) gzipSessions.close(runDir);
			return new IngestOutcome(false, "error");
		}
		if (gzip && "ready".equals(preparation.state())) {
			gzipSessions.close(runDir);
			return new IngestOutcome(false, "ready");
		}

		final long startOffset = preparation.committedOffset();
		final BlockCursor cursor = new BlockCursor(startOffset);
		final Map<String, TagWriteState> tagStates = new HashMap<>();
		final Set<WarningKey> warnings = new HashSet<>();
		final Set<TagQuarantineWarning> quarantineWarnings = new HashSet<>();
		final InputStream input;
		final LongSupplier consumedBytes;
		final boolean closeInput;
		if (gzip) {
			final GzipInputSessions.Session session;
			try {
				session = gzipSessions.acquire(runDir, source, preparation.generation());
			} catch (IOException e) {
				markRunError(runDir, source, "gzip_corrupt", e.getMessage());
				return new IngestOutcome(true, "error");
			}
			input = session.input();
			consumedBytes = session::compressedBytes;
			closeInput = false;
		} else {
			input = new BufferedInputStream(
					java.nio.file.Files.newInputStream(source.path()), 64 * 1024);
			input.skipNBytes(startOffset);
			consumedBytes = () -> cursor.bytesRead;
			closeInput = true;
		}

		try (ConnectionHandle handle = database.openWrite(runDir)) {
			final Connection connection = handle.connection();
			connection.setAutoCommit(false);

			try {
				try (PreparedStatement insertScalar = connection.prepareStatement(
							"INSERT INTO scalars(tag_id, ordinal, step, value) VALUES(?, ?, ?, ?)");
						PreparedStatement insertJson = connection.prepareStatement("""
								INSERT INTO json_lines(type, tag, step, timestamp, json)
								VALUES(?, ?, ?, ?, ?)
								""");
						LodIngestWriter.InsertSession lodInsert =
								lodWriter.openInsertSession(connection)) {
					// 1行ずつparseし、巨大な中間Listを作らず同じtransactionへ反映する。
					// JSONLはMetricsSource取得後も追記され得るため、取得時点の末尾を論理EOFとする。
					final long readLimitOffset = gzip ? Long.MAX_VALUE : source.size();
					readCompleteLines(input, consumedBytes, gzip, readLimitOffset, cursor, line -> {
						final ParsedRecord record = parseRecord(line, cursor.bytesRead);
						if (record.scalar()) {
							writeScalar(
									connection,
									insertScalar,
									lodInsert,
									runId,
									record,
									cursor.bytesRead,
									tagStates,
									warnings,
									quarantineWarnings);
						} else {
							writeJsonLine(insertJson, record, line);
						}
					});
				}

				// block内で更新したTagStatsとsource位置を、L0と同じcommit境界で確定する。
				flushTagStats(connection, tagStates);
				if (gzip && cursor.eof && cursor.trailingBytes != 0) {
					throw new GzipCorruptException("gzip ended with an unterminated JSON line");
				}
				// raw JSONLの未終端末尾は次回追記まで保持するが、現snapshotの完全行には
				// 追いついている。gzipだけはimmutableなので未終端末尾をcorruptとする。
				final String state = cursor.eof && (!gzip || cursor.trailingBytes == 0)
						? "ready"
						: "converting";
				updateSourceMeta(connection, source, cursor.committedOffset, state);
				connection.commit();

				for (WarningKey warning : warnings) {
					if (issuedWarnings.add(warning)) {
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
				if (gzip && "ready".equals(state)) gzipSessions.close(runDir);
				return new IngestOutcome(
						cursor.committedOffset != startOffset || !state.equals(preparation.state()),
						state);
			} catch (FatalRecordException e) {
				connection.rollback();
				markRunError(runDir, source, e.code(), e.getMessage());
				if (gzip) gzipSessions.close(runDir);
				return new IngestOutcome(true, "error");
			} catch (GzipCorruptException e) {
				connection.rollback();
				markRunError(runDir, source, "gzip_corrupt", e.getMessage());
				gzipSessions.close(runDir);
				return new IngestOutcome(true, "error");
			} catch (IOException e) {
				connection.rollback();
				if (gzip) {
					markRunError(runDir, source, "gzip_corrupt", e.getMessage());
					gzipSessions.close(runDir);
					return new IngestOutcome(true, "error");
				}
				throw e;
			} catch (Exception e) {
				connection.rollback();
				throw e;
			}
		} catch (IOException | SQLException e) {
			if (!gzip) markRunError(runDir, source, "source_read_error", e.getMessage());
			throw e;
		} finally {
			if (closeInput) input.close();
		}
	}

	private void readCompleteLines(
			InputStream input,
			LongSupplier consumedBytes,
			boolean compressed,
			long readLimitOffset,
			BlockCursor cursor,
			LineConsumer consumer) throws Exception {
		final ByteArrayOutputStream lineBuffer = new ByteArrayOutputStream(4096);
		int value = -1;
		while (cursor.completedLines < maxBlockLines
				&& cursor.bytesRead < readLimitOffset
				&& (value = input.read()) >= 0) {
			if (compressed) cursor.bytesRead = consumedBytes.getAsLong();
			else cursor.bytesRead++;
			if (value == '\n') {
				cursor.completedLines++;
				cursor.committedOffset = compressed ? consumedBytes.getAsLong() : cursor.bytesRead;
				String line = lineBuffer.toString(StandardCharsets.UTF_8);
				lineBuffer.reset();
				if (line.endsWith("\r")) line = line.substring(0, line.length() - 1);
				consumer.accept(line);
			} else {
				lineBuffer.write(value);
			}
		}
		cursor.trailingBytes = lineBuffer.size();
		cursor.eof = value < 0 || cursor.bytesRead >= readLimitOffset;
		if (compressed) cursor.bytesRead = consumedBytes.getAsLong();
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
			Connection connection,
			PreparedStatement insertScalar,
			LodIngestWriter.InsertSession lodInsert,
			String runId,
			ParsedRecord record,
			long sourceOffset,
			Map<String, TagWriteState> tagStates,
			Set<WarningKey> warnings,
			Set<TagQuarantineWarning> quarantineWarnings) throws SQLException {
		final JsonNode valueNode = record.valueNode();
		if (valueNode.isNull()) {
			warnings.add(new WarningKey(runId, record.tag(), "null"));
			return;
		}
		if (!valueNode.isNumber()) {
			warnings.add(new WarningKey(runId, record.tag(), "not_numeric"));
			return;
		}
		final double value = valueNode.doubleValue();
		if (!Double.isFinite(value)) {
			warnings.add(new WarningKey(runId, record.tag(), "non_finite"));
			return;
		}
		if (!Float.isFinite((float) value)) {
			warnings.add(new WarningKey(runId, record.tag(), "f32_overflow"));
			return;
		}

		final TagWriteState state = tagStates.computeIfAbsent(
				record.tag(),
				tag -> loadTagState(connection, tag));
		if (state.quarantined) return;
		if (state.count > 0L && record.step() < state.previousStep) {
			quarantineWarnings.add(new TagQuarantineWarning(
					runId,
					record.tag(),
					state.previousStep,
					record.step(),
					sourceOffset));
			quarantineTag(connection, state, sourceOffset, record.step());
			state.quarantined = true;
			return;
		}

		final long ordinal = state.nextOrdinal;
		insertScalar.setLong(1, state.tagId);
		insertScalar.setLong(2, ordinal);
		insertScalar.setLong(3, record.step());
		insertScalar.setDouble(4, value);
		insertScalar.executeUpdate();
		state.record(record.step(), value);
		lodWriter.append(
				lodInsert,
				state.tagId,
				state.lodState,
				LodBucket.point(ordinal, record.step(), value));
	}

	private TagWriteState loadTagState(Connection connection, String tag) {
		try {
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
		} catch (SQLException e) {
			throw new DatabaseWriteRuntimeException(e);
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
			String state) throws SQLException, IOException {
		upsertMeta(connection, "source_size", Long.toString(source.size()));
		upsertMeta(connection, "source_mtime", Long.toString(source.modifiedTime()));
		upsertMeta(connection, "source_head_sha256", source.headSha256());
		upsertMeta(connection, "source_commit_tail_sha256", source.sha256Before(committedOffset));
		upsertMeta(connection, "committed_offset", Long.toString(committedOffset));
		upsertMeta(connection, "state", state);
		try (PreparedStatement statement = connection.prepareStatement(
				"DELETE FROM source_meta WHERE k IN ('error_code', 'error_message')")) {
			statement.executeUpdate();
		}
	}

	private void markRunError(
			Path runDir,
			MetricsSource source,
			String code,
			String message) {
		try (ConnectionHandle handle = database.openWrite(runDir)) {
			if (handle == null) return;
			final Connection connection = handle.connection();
			connection.setAutoCommit(false);
			upsertMeta(connection, "source_size", Long.toString(source.size()));
			upsertMeta(connection, "source_mtime", Long.toString(source.modifiedTime()));
			upsertMeta(connection, "source_head_sha256", source.headSha256());
			upsertMeta(connection, "state", "error");
			upsertMeta(connection, "error_code", code);
			upsertMeta(connection, "error_message", message == null ? code : message);
			connection.commit();
			log.warn("Run ingest entered error state: run={} code={} message={}",
					runDir.getFileName(), code, message == null ? code : message);
		} catch (Exception error) {
			log.error("Failed to persist Run error: run={} code={} message={}",
					runDir.getFileName(), code, error.getMessage());
		}
	}

	private static void upsertMeta(Connection connection, String key, String value) throws SQLException {
		try (PreparedStatement statement = connection.prepareStatement("""
				INSERT INTO source_meta(k, v) VALUES(?, ?)
				ON CONFLICT(k) DO UPDATE SET v=excluded.v
				""")) {
			statement.setString(1, key);
			statement.setString(2, value);
			statement.executeUpdate();
		}
	}

	@FunctionalInterface
	private interface LineConsumer {
		void accept(String line) throws Exception;
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

	private static final class BlockCursor {
		private long bytesRead;
		private long committedOffset;
		private int completedLines;
		private int trailingBytes;
		private boolean eof;

		private BlockCursor(long startOffset) {
			this.bytesRead = startOffset;
			this.committedOffset = startOffset;
		}
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

	private static final class GzipCorruptException extends IOException {
		private GzipCorruptException(String message) {
			super(message);
		}
	}

	private static final class DatabaseWriteRuntimeException extends RuntimeException {
		private DatabaseWriteRuntimeException(SQLException cause) {
			super(cause);
		}
	}
}
