package io.github.kazukin123.anetlab.metricsviewer.service;

import java.nio.file.Path;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import io.github.kazukin123.anetlab.metricsviewer.config.MetricsViewerSettings;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsCacheDatabase;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsCacheDatabase.CacheMetadata;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsCacheDatabase.ConnectionHandle;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsCacheDatabase.IngestState;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsCacheDatabase.SourceMeta;
import io.github.kazukin123.anetlab.metricsviewer.infra.RunScanner;
import io.github.kazukin123.anetlab.metricsviewer.view.model.ApiError;
import io.github.kazukin123.anetlab.metricsviewer.view.model.IngestInfo;
import io.github.kazukin123.anetlab.metricsviewer.view.model.Issue;
import io.github.kazukin123.anetlab.metricsviewer.view.model.MetricsSeriesRequest;
import io.github.kazukin123.anetlab.metricsviewer.view.model.MetricsSeriesResult;
import io.github.kazukin123.anetlab.metricsviewer.view.model.RunInfo;
import io.github.kazukin123.anetlab.metricsviewer.view.model.RunStats;
import io.github.kazukin123.anetlab.metricsviewer.view.model.TagInfo;
import io.github.kazukin123.anetlab.metricsviewer.view.model.TagStats;

public class MetricsRepository {

	private static final Logger log = LoggerFactory.getLogger(MetricsRepository.class);
	private static final long SOURCE_META_NUMERIC_FALLBACK = 0L;

	private final RunScanner runScanner;
	private final MetricsCacheDatabase database;
	private final MetricsRangeProjector rangeProjector;
	private final MetricsQueryPlanner queryPlanner;

	public MetricsRepository(
			RunScanner runScanner,
			MetricsCacheDatabase database,
			MetricsViewerSettings settings,
			MetricsRangeProjector rangeProjector) {
		this.runScanner = runScanner;
		this.database = database;
		this.rangeProjector = rangeProjector;
		this.queryPlanner = new MetricsQueryPlanner(settings);
	}

	public RunInfo findRunInfo(String runId) {
		final Path runDir = runScanner.resolveRunDir(runId);
		try (ConnectionHandle handle = database.openRead(runDir)) {
			if (handle == null) return pendingRun(runId);

			final CacheMetadata metadata = SourceMeta.read(
					handle.connection(), SOURCE_META_NUMERIC_FALLBACK);
			final IngestState state = metadata.stateOrThrow();
			final List<TagInfo> tags = readTags(handle.connection());
			final Long maxStep = tags.stream()
					.map(TagInfo::getStats)
					.filter(stats -> stats != null)
					.map(TagStats::getMaxStep)
					.max(Long::compareTo)
					.orElse(null);
			final ApiError error = metadata.errorCode() == null
					? null
					: new ApiError(metadata.errorCode(), metadata.errorMessage());

			return RunInfo.builder()
					.id(runId)
					.generation(UUID.fromString(metadata.generation()))
					.stats(RunStats.builder().maxStep(maxStep).build())
					.ingest(new IngestInfo(
							state.externalName(),
							calculatePercentage(metadata),
							error))
					.tags(tags)
					.build();
		} catch (Exception e) {
			log.warn("Failed to read Run metadata: run={} message={}", runId, e.getMessage());
			return pendingRun(runId);
		}
	}

	public List<MetricsSeriesResult> query(
			List<MetricsSeriesRequest> requests,
			QueryExecution query) {
		query.checkpoint();
		final Set<String> existingRuns = new HashSet<>(runScanner.listRunId());
		final MetricsQueryPlanner.SeriesInput[] inputs =
				new MetricsQueryPlanner.SeriesInput[requests.size()];
		final Map<String, List<Integer>> indicesByRun = new LinkedHashMap<>();
		for (int i = 0; i < requests.size(); i++) {
			query.checkpoint();
			final MetricsSeriesRequest request = requests.get(i);
			if (!existingRuns.contains(request.getRunId())) {
				inputs[i] = new MetricsQueryPlanner.SeriesInput(
						i, request, false, null, null, null);
				continue;
			}
			indicesByRun.computeIfAbsent(request.getRunId(), ignored -> new ArrayList<>()).add(i);
		}
		// Runごとに1本のread connectionを計画から射影完了まで保持する。lifecycle read lockで
		// 全再構築を排他し、SQLite transactionにより通常commitをまたいでも同じsnapshotを参照する。
		final Map<String, RunQueryContext> contexts = new LinkedHashMap<>();
		try {
			for (Map.Entry<String, List<Integer>> entry : indicesByRun.entrySet()) {
				query.checkpoint();
				final RunQueryContext context = openQueryContext(entry.getKey(), query);
				contexts.put(entry.getKey(), context);
				readSeriesInputs(context, entry.getValue(), requests, inputs, query);
			}

			final MetricsQueryPlanner.SeriesPlan[] plans =
					queryPlanner.plan(Arrays.asList(inputs));
			final MetricsSeriesResult[] results = new MetricsSeriesResult[plans.length];
			for (MetricsQueryPlanner.SeriesPlan plan : plans) {
				query.checkpoint();
				final RunQueryContext context = plan.availability() == SeriesAvailability.OK
						? contexts.get(plan.request().getRunId())
						: null;
				results[plan.index()] = buildResult(plan, context, query);
			}
			query.checkpoint();
			return Arrays.asList(results);
		} finally {
			for (RunQueryContext context : contexts.values()) {
				if (context != null) context.close();
			}
		}
	}

	private void readSeriesInputs(
			RunQueryContext context,
			List<Integer> indices,
			List<MetricsSeriesRequest> requests,
			MetricsQueryPlanner.SeriesInput[] inputs,
			QueryExecution query) {
		if (context == null) {
			for (int index : indices) {
				query.checkpoint();
				inputs[index] = new MetricsQueryPlanner.SeriesInput(
						index, requests.get(index), true, null, null, null);
			}
			return;
		}
		for (int index : indices) {
			query.checkpoint();
			final MetricsSeriesRequest request = requests.get(index);
			try {
				inputs[index] = new MetricsQueryPlanner.SeriesInput(
						index,
						request,
						true,
						context.metadata,
						readTagInput(context.connection, request, query),
						null);
			} catch (QueryCancelledException e) {
				throw e;
			} catch (Exception e) {
				inputs[index] = new MetricsQueryPlanner.SeriesInput(
						index, request, true, context.metadata, null, e.getMessage());
			}
		}
	}

	private RunQueryContext openQueryContext(String runId, QueryExecution query) {
		ConnectionHandle handle = null;
		try {
			query.checkpoint();
			handle = database.openRead(runScanner.resolveRunDir(runId));
			if (handle == null) return null;
			handle.connection().setAutoCommit(false);
			final CacheMetadata metadata = SourceMeta.read(
					handle.connection(), SOURCE_META_NUMERIC_FALLBACK);
			query.checkpoint();
			return new RunQueryContext(handle, metadata);
		} catch (QueryCancelledException e) {
			if (handle != null) {
				try {
					handle.close();
				} catch (Exception closeError) {
					e.addSuppressed(closeError);
				}
			}
			throw e;
		} catch (Exception e) {
			if (handle != null) {
				try {
					handle.close();
				} catch (Exception closeError) {
					e.addSuppressed(closeError);
				}
			}
			log.warn("Failed to open Metrics query snapshot: run={} message={}", runId, e.getMessage());
			return null;
		}
	}

	private static MetricsQueryPlanner.TagInput readTagInput(
			Connection connection,
			MetricsSeriesRequest request,
			QueryExecution query) throws Exception {
		final long tagId;
		final String status;
		final String errorCode;
		final String errorMessage;
		final long count;
		final boolean empty;
		try (PreparedStatement statement = connection.prepareStatement("""
				SELECT t.id, t.status, t.error_code, t.error_message, s.count
				FROM tags t
				LEFT JOIN tag_stats s ON s.tag_id=t.id
				WHERE t.key=?
				""");
				StatementRegistration ignored = query.registerStatement(statement)) {
			statement.setString(1, request.getTagKey());
			query.checkpoint();
			try (ResultSet result = statement.executeQuery()) {
				query.checkpoint();
				if (!result.next()) return null;
				tagId = result.getLong(1);
				status = result.getString(2);
				errorCode = "error".equals(status) ? result.getString(3) : null;
				errorMessage = "error".equals(status) ? result.getString(4) : null;
				count = result.getLong(5);
				empty = result.wasNull() || count == 0L;
			}
		}
		if (empty) {
			return new MetricsQueryPlanner.TagInput(
					tagId, null, 0L, 0L, "error".equals(status), errorCode, errorMessage);
		}
		final long ordinalFrom = lowerBound(
				connection, tagId, count, request.getFromStep(), false, query);
		final long ordinalTo = lowerBound(
				connection, tagId, count, request.getToStep(), true, query);
		return new MetricsQueryPlanner.TagInput(
				tagId,
				count,
				ordinalFrom,
				ordinalTo,
				"error".equals(status),
				errorCode,
				errorMessage);
	}

	private MetricsSeriesResult buildResult(
			MetricsQueryPlanner.SeriesPlan plan,
			RunQueryContext context,
			QueryExecution query) {
		query.checkpoint();
		if (plan.availability() != SeriesAvailability.OK) {
			return baseResult(plan)
					.level(null)
					.bucketWidth(null)
					.projection(null)
					.build();
		}

		try {
			final MetricsRangeProjector.Projection projection = rangeProjector.project(
					context.connection,
					plan.generationValue(),
					plan.request().getRunId(),
					plan.tagId(),
					plan.ordinalFrom(),
					plan.ordinalTo(),
					plan.pointBudget(),
					query);
			return baseResult(plan)
					.level(projection.level())
					.bucketWidth(projection.bucketWidth())
					.projection(projection.body())
					.build();
		} catch (QueryCancelledException e) {
			throw e;
		} catch (Exception e) {
			plan.issues().add(new Issue("run", "query_error", e.getMessage()));
			return baseResult(plan)
					.availability(SeriesAvailability.PENDING.externalName())
					.level(null)
					.bucketWidth(null)
					.projection(null)
					.build();
		}
	}

	private static MetricsSeriesResult.MetricsSeriesResultBuilder baseResult(
			MetricsQueryPlanner.SeriesPlan plan) {
		return MetricsSeriesResult.builder()
				.runId(plan.request().getRunId())
				.tagKey(plan.request().getTagKey())
				.generation(plan.generation())
				.fromStep(plan.request().getFromStep())
				.toStep(plan.request().getToStep())
				.availability(plan.availability().externalName())
				.pointBudget(plan.pointBudget())
				.issues(List.copyOf(plan.issues()));
	}

	private static long lowerBound(
			Connection connection,
			long tagId,
			long count,
			long target,
			boolean upperBound,
			QueryExecution query) throws Exception {
		long low = 0L;
		long high = count;
		try (PreparedStatement statement = connection.prepareStatement(
				"SELECT step FROM scalars WHERE tag_id=? AND ordinal=?");
				StatementRegistration ignored = query.registerStatement(statement)) {
			while (low < high) {
				query.checkpoint();
				final long middle = low + (high - low) / 2L;
				statement.setLong(1, tagId);
				statement.setLong(2, middle);
				try (ResultSet result = statement.executeQuery()) {
					if (!result.next()) throw new IllegalStateException("Missing L0 ordinal " + middle);
					final long step = result.getLong(1);
					if (step < target || (upperBound && step == target)) low = middle + 1L;
					else high = middle;
				}
			}
		}
		return low;
	}

	private static List<TagInfo> readTags(Connection connection) throws Exception {
		final List<TagInfo> tags = new ArrayList<>();
		try (PreparedStatement statement = connection.prepareStatement("""
				SELECT t.key, t.type, t.status, t.error_code, t.error_message,
				       s.count, s.mean, s.m2, s.min_value, s.max_value,
				       s.min_step, s.max_step, s.last_value
				FROM tags t
				LEFT JOIN tag_stats s ON s.tag_id=t.id
				ORDER BY t.key
				""");
				ResultSet result = statement.executeQuery()) {
			while (result.next()) {
				final long count = result.getLong(6);
				final TagStats stats;
				if (result.wasNull()) {
					stats = null;
				} else {
					final double m2 = result.getDouble(8);
					final double variance = count == 0L ? 0.0 : Math.max(0.0, m2 / count);
					stats = TagStats.builder()
							.count(count)
							.mean(result.getDouble(7))
							.minValue(result.getDouble(9))
							.maxValue(result.getDouble(10))
							.minStep(result.getLong(11))
							.maxStep(result.getLong(12))
							.lastValue(result.getDouble(13))
							.variance(variance)
							.stdDev(Math.sqrt(variance))
							.build();
				}
				final String errorCode = result.getString(4);
				tags.add(TagInfo.builder()
						.key(result.getString(1))
						.type(result.getString(2))
						.status(result.getString(3))
						.stats(stats)
						.error(errorCode == null
								? null
								: new ApiError(errorCode, result.getString(5)))
						.build());
			}
		}
		return tags;
	}

	private static int calculatePercentage(CacheMetadata metadata) {
		final IngestState state = metadata.stateOrThrow();
		if (state == IngestState.PENDING) return 0;
		if (state == IngestState.READY) return 100;
		if (metadata.sourceSize() <= 0L) return state == IngestState.ERROR ? 0 : 100;
		final long percentage = metadata.committedOffset() * 100L / metadata.sourceSize();
		return (int) Math.max(0L, Math.min(99L, percentage));
	}

	private static RunInfo pendingRun(String runId) {
		return RunInfo.builder()
				.id(runId)
				.generation(null)
				.stats(new RunStats())
				.ingest(new IngestInfo(IngestState.PENDING.externalName(), 0, null))
				.tags(List.of())
				.build();
	}

	private static final class RunQueryContext implements AutoCloseable {
		private final ConnectionHandle handle;
		private final Connection connection;
		private final CacheMetadata metadata;

		private RunQueryContext(ConnectionHandle handle, CacheMetadata metadata) {
			this.handle = handle;
			this.connection = handle.connection();
			this.metadata = metadata;
		}

		@Override
		public void close() {
			try {
				handle.close();
			} catch (Exception e) {
				log.warn("Failed to close Metrics query snapshot: {}", e.getMessage());
			}
		}
	}

}
