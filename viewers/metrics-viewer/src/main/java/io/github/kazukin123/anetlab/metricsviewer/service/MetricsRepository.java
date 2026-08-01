package io.github.kazukin123.anetlab.metricsviewer.service;

import java.nio.file.Path;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Component;

import io.github.kazukin123.anetlab.metricsviewer.config.MetricsViewerSettings;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsCacheDatabase;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsCacheDatabase.CacheMetadata;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsCacheDatabase.ConnectionHandle;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsCacheDatabase.IngestState;
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

@Component
public class MetricsRepository {

	private static final Logger log = LoggerFactory.getLogger(MetricsRepository.class);

	private final RunScanner runScanner;
	private final MetricsCacheDatabase database;
	private final MetricsViewerSettings settings;
	private final MetricsRangeProjector rangeProjector;

	public MetricsRepository(
			RunScanner runScanner,
			MetricsCacheDatabase database,
			MetricsViewerSettings settings,
			MetricsRangeProjector rangeProjector) {
		this.runScanner = runScanner;
		this.database = database;
		this.settings = settings;
		this.rangeProjector = rangeProjector;
	}

	public RunInfo findRunInfo(String runId) {
		final Path runDir = runScanner.resolveRunDir(runId);
		try (ConnectionHandle handle = database.openRead(runDir)) {
			if (handle == null) return pendingRun(runId);

			final Map<String, String> meta = readMeta(handle.connection());
			final CacheMetadata metadata = new CacheMetadata(
					meta.get("generation"),
					IngestState.fromDb(meta.getOrDefault(
							"state", IngestState.PENDING.externalName())),
					parseLong(meta.get("committed_offset"), 0L),
					parseLong(meta.get("source_size"), 0L),
					meta.get("error_code"),
					meta.get("error_message"));
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
							metadata.state().externalName(),
							calculatePercentage(metadata),
							error))
					.tags(tags)
					.build();
		} catch (Exception e) {
			log.warn("Failed to read Run metadata: run={} message={}", runId, e.getMessage());
			return pendingRun(runId);
		}
	}

	public List<MetricsSeriesResult> query(List<MetricsSeriesRequest> requests) {
		final Set<String> existingRuns = new HashSet<>(runScanner.listRunId());
		final Map<String, RunQueryContext> contexts = new LinkedHashMap<>();
		final SeriesPlan[] plans = new SeriesPlan[requests.size()];

		try {
			// request内ではRunごとに1本だけread connectionを開き、同じsnapshotを使う。
			for (MetricsSeriesRequest request : requests) {
				if (!existingRuns.contains(request.getRunId())
						|| contexts.containsKey(request.getRunId())) {
					continue;
				}
				contexts.put(request.getRunId(), openQueryContext(request.getRunId()));
			}

			for (int i = 0; i < requests.size(); i++) {
				final MetricsSeriesRequest request = requests.get(i);
				plans[i] = inspectSeries(
						i,
						request,
						existingRuns.contains(request.getRunId()),
						contexts.get(request.getRunId()));
			}

			allocatePointBudgets(plans);

			final MetricsSeriesResult[] results = new MetricsSeriesResult[plans.length];
			for (SeriesPlan plan : plans) results[plan.index] = buildResult(plan);
			return Arrays.asList(results);
		} finally {
			for (RunQueryContext context : contexts.values()) {
				if (context != null) context.close();
			}
		}
	}

	private RunQueryContext openQueryContext(String runId) {
		ConnectionHandle handle = null;
		try {
			handle = database.openRead(runScanner.resolveRunDir(runId));
			if (handle == null) return null;
			handle.connection().setAutoCommit(false);
			return new RunQueryContext(handle, readMeta(handle.connection()));
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

	private SeriesPlan inspectSeries(
			int index,
			MetricsSeriesRequest request,
			boolean runExists,
			RunQueryContext context) {
		final SeriesPlan plan = new SeriesPlan(index, request);
		if (!runExists) {
			plan.availability = SeriesAvailability.NOT_FOUND;
			return plan;
		}
		if (context == null) {
			plan.availability = SeriesAvailability.PENDING;
			return plan;
		}

		plan.context = context;
		plan.generation = parseUuid(context.meta.get("generation"));
		addRunIssue(plan, context.meta);

		try {
			final IngestState ingestState = IngestState.fromDb(context.meta.getOrDefault(
					"state", IngestState.PENDING.externalName()));
			try (PreparedStatement statement = context.connection.prepareStatement("""
				SELECT t.id, t.status, t.error_code, t.error_message, s.count
				FROM tags t
				LEFT JOIN tag_stats s ON s.tag_id=t.id
				WHERE t.key=?
					""")) {
				statement.setString(1, request.getTagKey());
				try (ResultSet result = statement.executeQuery()) {
					if (!result.next()) {
						plan.availability = ingestState.isStillIngesting()
								? SeriesAvailability.PENDING
								: SeriesAvailability.NOT_FOUND;
						return plan;
					}
					plan.tagId = result.getLong(1);
					if ("error".equals(result.getString(2))) {
						plan.issues.add(new Issue(
								"tag",
								result.getString(3),
								result.getString(4)));
					}
					final long totalCount = result.getLong(5);
					if (result.wasNull() || totalCount == 0L) {
						plan.availability = ingestState.isStillIngesting()
								? SeriesAvailability.PENDING
								: SeriesAvailability.EMPTY;
						return plan;
					}

					plan.ordinalFrom = lowerBound(
							context.connection, plan.tagId, totalCount, request.getFromStep(), false);
					plan.ordinalTo = lowerBound(
							context.connection, plan.tagId, totalCount, request.getToStep(), true);
					plan.availableVertices = Math.max(0L, plan.ordinalTo - plan.ordinalFrom);
					plan.availability = plan.availableVertices > 0L
							? SeriesAvailability.OK
							: (ingestState.isStillIngesting()
									? SeriesAvailability.PENDING
									: SeriesAvailability.EMPTY);
					return plan;
				}
			}
		} catch (Exception e) {
			plan.availability = SeriesAvailability.PENDING;
			plan.issues.add(new Issue("run", "query_error", e.getMessage()));
			return plan;
		}
	}

	private void allocatePointBudgets(SeriesPlan[] plans) {
		long requiredMinimum = 0L;
		for (SeriesPlan plan : plans) {
			if (plan.availability != SeriesAvailability.OK) continue;
			final int requested = effectiveMaxPoints(plan.request);
			plan.cap = (int) Math.min((long) requested, plan.availableVertices);
			plan.pointBudget = Math.min(50, plan.cap);
			requiredMinimum += plan.pointBudget;
		}

		if (requiredMinimum > settings.getMaxPointsPerRequest()) {
			final Map<String, Object> body = new LinkedHashMap<>();
			body.put("seriesCount", plans.length);
			body.put("requiredMinimumPoints", requiredMinimum);
			body.put("maxPointsPerRequest", settings.getMaxPointsPerRequest());
			throw new MetricsApiException(HttpStatus.UNPROCESSABLE_ENTITY, body);
		}

		int remaining = settings.getMaxPointsPerRequest() - (int) requiredMinimum;
		while (remaining > 0) {
			final List<SeriesPlan> active = Arrays.stream(plans)
					.filter(plan -> plan.pointBudget < plan.cap)
					.toList();
			if (active.isEmpty()) break;

			final int fairShare = Math.max(1, remaining / active.size());
			boolean allocated = false;
			for (SeriesPlan plan : active) {
				final int addition = Math.min(plan.cap - plan.pointBudget, Math.min(fairShare, remaining));
				if (addition <= 0) continue;
				plan.pointBudget += addition;
				remaining -= addition;
				allocated = true;
				if (remaining == 0) break;
			}
			if (!allocated) break;
		}
	}

	private MetricsSeriesResult buildResult(SeriesPlan plan) {
		if (plan.availability != SeriesAvailability.OK) {
			return baseResult(plan)
					.level(null)
					.bucketWidth(null)
					.projection(null)
					.build();
		}

		try {
			final MetricsRangeProjector.Projection projection = rangeProjector.project(
					plan.context.connection,
					plan.context.meta.get("generation"),
					plan.request.getRunId(),
					plan.tagId,
					plan.ordinalFrom,
					plan.ordinalTo,
					plan.pointBudget);
			return baseResult(plan)
					.level(projection.level())
					.bucketWidth(projection.bucketWidth())
					.projection(projection.body())
					.build();
		} catch (Exception e) {
			plan.issues.add(new Issue("run", "query_error", e.getMessage()));
			return baseResult(plan)
					.availability(SeriesAvailability.PENDING.externalName())
					.level(null)
					.bucketWidth(null)
					.projection(null)
					.build();
		}
	}

	private static MetricsSeriesResult.MetricsSeriesResultBuilder baseResult(SeriesPlan plan) {
		return MetricsSeriesResult.builder()
				.runId(plan.request.getRunId())
				.tagKey(plan.request.getTagKey())
				.generation(plan.generation)
				.fromStep(plan.request.getFromStep())
				.toStep(plan.request.getToStep())
				.availability(plan.availability.externalName())
				.pointBudget(plan.pointBudget)
				.issues(List.copyOf(plan.issues));
	}

	private int effectiveMaxPoints(MetricsSeriesRequest request) {
		return request.getMaxPoints() == null
				? settings.getTargetPointsPerSeries()
				: request.getMaxPoints();
	}

	private static long lowerBound(
			Connection connection,
			long tagId,
			long count,
			long target,
			boolean upperBound) throws Exception {
		long low = 0L;
		long high = count;
		try (PreparedStatement statement = connection.prepareStatement(
				"SELECT step FROM scalars WHERE tag_id=? AND ordinal=?")) {
			while (low < high) {
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

	private static void addRunIssue(SeriesPlan plan, Map<String, String> meta) {
		final String code = meta.get("error_code");
		if (code != null) {
			plan.issues.add(new Issue("run", code, meta.get("error_message")));
		}
	}

	private static UUID parseUuid(String value) {
		try {
			return value == null ? null : UUID.fromString(value);
		} catch (IllegalArgumentException e) {
			return null;
		}
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

	private static Map<String, String> readMeta(Connection connection) throws Exception {
		final Map<String, String> meta = new HashMap<>();
		try (PreparedStatement statement = connection.prepareStatement("SELECT k, v FROM source_meta");
				ResultSet result = statement.executeQuery()) {
			while (result.next()) meta.put(result.getString(1), result.getString(2));
		}
		return meta;
	}

	private static int calculatePercentage(CacheMetadata metadata) {
		if (metadata.state() == IngestState.PENDING) return 0;
		if (metadata.state() == IngestState.READY) return 100;
		if (metadata.sourceSize() <= 0L) return metadata.state() == IngestState.ERROR ? 0 : 100;
		final long percentage = metadata.committedOffset() * 100L / metadata.sourceSize();
		return (int) Math.max(0L, Math.min(99L, percentage));
	}

	private static long parseLong(String value, long fallback) {
		if (value == null) return fallback;
		try {
			return Long.parseLong(value);
		} catch (NumberFormatException e) {
			return fallback;
		}
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

	private static final class RunQueryContext {
		private final ConnectionHandle handle;
		private final Connection connection;
		private final Map<String, String> meta;

		private RunQueryContext(ConnectionHandle handle, Map<String, String> meta) {
			this.handle = handle;
			this.connection = handle.connection();
			this.meta = meta;
		}

		private void close() {
			try {
				handle.close();
			} catch (Exception e) {
				log.warn("Failed to close Metrics query snapshot: {}", e.getMessage());
			}
		}
	}

	private static final class SeriesPlan {
		private final int index;
		private final MetricsSeriesRequest request;
		private final List<Issue> issues = new ArrayList<>();
		private RunQueryContext context;
		private UUID generation;
		private long tagId;
		private long ordinalFrom;
		private long ordinalTo;
		private long availableVertices;
		private SeriesAvailability availability;
		private int cap;
		private int pointBudget;

		private SeriesPlan(int index, MetricsSeriesRequest request) {
			this.index = index;
			this.request = request;
		}
	}

}

enum SeriesAvailability {
	OK("ok"),
	PENDING("pending"),
	NOT_FOUND("not_found"),
	EMPTY("empty");

	private final String externalName;

	SeriesAvailability(String externalName) {
		this.externalName = externalName;
	}

	String externalName() {
		return externalName;
	}
}
