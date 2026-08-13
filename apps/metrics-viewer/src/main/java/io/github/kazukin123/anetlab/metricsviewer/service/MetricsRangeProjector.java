package io.github.kazukin123.anetlab.metricsviewer.service;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

import io.github.kazukin123.anetlab.metricsviewer.util.MetricTraceEncoder;
import io.github.kazukin123.anetlab.metricsviewer.view.model.LodProjection;
import io.github.kazukin123.anetlab.metricsviewer.view.model.LodSummary;
import io.github.kazukin123.anetlab.metricsviewer.view.model.MinMaxProjection;
import io.github.kazukin123.anetlab.metricsviewer.view.model.RawProjection;

public class MetricsRangeProjector {

	private final LodPageCache pageCache;

	public MetricsRangeProjector(LodPageCache pageCache) {
		this.pageCache = pageCache;
	}

	public Projection project(
			Connection connection,
			String generation,
			String runId,
			long tagId,
			long ordinalFrom,
			long ordinalTo,
			int pointBudget) throws Exception {
		final long rawCount = ordinalTo - ordinalFrom;
		if (rawCount <= pointBudget) {
			return raw(connection, tagId, ordinalFrom, ordinalTo);
		}

		final LevelSelection selection = selectLevel(ordinalFrom, ordinalTo, pointBudget);
		final List<LodBucket> buckets = loadBuckets(
				connection,
				generation,
				runId,
				tagId,
				ordinalFrom,
				ordinalTo,
				selection);
		return lod(selection, buckets);
	}

	private static Projection raw(
			Connection connection,
			long tagId,
			long ordinalFrom,
			long ordinalTo) throws Exception {
		final int size = Math.toIntExact(ordinalTo - ordinalFrom);
		final double[] steps = new double[size];
		final float[] values = new float[size];
		try (PreparedStatement statement = connection.prepareStatement("""
				SELECT step, value
				FROM scalars
				WHERE tag_id=? AND ordinal>=? AND ordinal<?
				ORDER BY ordinal
				""")) {
			statement.setLong(1, tagId);
			statement.setLong(2, ordinalFrom);
			statement.setLong(3, ordinalTo);
			try (ResultSet result = statement.executeQuery()) {
				int i = 0;
				while (result.next()) {
					steps[i] = result.getLong(1);
					values[i] = (float) result.getDouble(2);
					i++;
				}
			}
		}
		return new Projection(
				0,
				1L,
				new RawProjection(
						MetricTraceEncoder.encodeDoubleArray(steps),
						MetricTraceEncoder.encodeFloatArray(values)));
	}

	private static Projection lod(LevelSelection selection, List<LodBucket> buckets) {
		final List<Candidate> minMaxCandidates = new ArrayList<>();
		final double[] summarySteps = new double[buckets.size()];
		final float[] mins = new float[buckets.size()];
		final float[] maxs = new float[buckets.size()];
		final float[] means = new float[buckets.size()];
		final double[] minSteps = new double[buckets.size()];
		final double[] maxSteps = new double[buckets.size()];

		for (int i = 0; i < buckets.size(); i++) {
			final LodBucket bucket = buckets.get(i);
			final List<Candidate> candidates = new ArrayList<>(List.of(
					new Candidate(bucket.minOrdinal(), bucket.minStep(), bucket.minValue()),
					new Candidate(bucket.maxOrdinal(), bucket.maxStep(), bucket.maxValue()),
					new Candidate(bucket.ordinalTo() - 1L, bucket.stepLast(), bucket.lastValue())));
			if (candidates.size() != LodBucket.POINTS_PER_LOD_BUCKET) {
				throw new IllegalStateException("LOD bucket requires exactly "
						+ LodBucket.POINTS_PER_LOD_BUCKET + " candidates");
			}
			candidates.sort(Comparator.comparingLong(Candidate::ordinal));
			long previousOrdinal = Long.MIN_VALUE;
			for (Candidate candidate : candidates) {
				if (candidate.ordinal() == previousOrdinal) continue;
				minMaxCandidates.add(candidate);
				previousOrdinal = candidate.ordinal();
			}

			summarySteps[i] = bucket.stepFirst();
			mins[i] = (float) bucket.minValue();
			maxs[i] = (float) bucket.maxValue();
			means[i] = (float) bucket.mean();
			minSteps[i] = bucket.minStep();
			maxSteps[i] = bucket.maxStep();
		}

		final double[] minMaxSteps = new double[minMaxCandidates.size()];
		final float[] minMaxValues = new float[minMaxCandidates.size()];
		for (int i = 0; i < minMaxCandidates.size(); i++) {
			minMaxSteps[i] = minMaxCandidates.get(i).step();
			minMaxValues[i] = (float) minMaxCandidates.get(i).value();
		}

		return new Projection(
				selection.level(),
				selection.width(),
				new LodProjection(
						new MinMaxProjection(
								MetricTraceEncoder.encodeDoubleArray(minMaxSteps),
								MetricTraceEncoder.encodeFloatArray(minMaxValues)),
						new LodSummary(
								MetricTraceEncoder.encodeDoubleArray(summarySteps),
								MetricTraceEncoder.encodeFloatArray(mins),
								MetricTraceEncoder.encodeFloatArray(maxs),
								MetricTraceEncoder.encodeFloatArray(means),
								MetricTraceEncoder.encodeDoubleArray(minSteps),
								MetricTraceEncoder.encodeDoubleArray(maxSteps))));
	}

	private List<LodBucket> loadBuckets(
			Connection connection,
			String generation,
			String runId,
			long tagId,
			long ordinalFrom,
			long ordinalTo,
			LevelSelection selection) throws Exception {
		final long firstBucket = ordinalFrom / selection.width();
		final long lastBucket = (ordinalTo - 1L) / selection.width();
		final List<LodBucket> buckets = new ArrayList<>();
		for (long bucket = firstBucket; bucket <= lastBucket; bucket++) {
			final long logicalFrom = Math.multiplyExact(bucket, selection.width());
			final long logicalTo = Math.addExact(logicalFrom, selection.width());
			final long exactFrom = Math.max(ordinalFrom, logicalFrom);
			final long exactTo = Math.min(ordinalTo, logicalTo);
			final LodBucket persisted = pageCache.find(
					connection, generation, runId, tagId, selection.level(), bucket);
			if (persisted != null && exactFrom == logicalFrom && exactTo == logicalTo) {
				buckets.add(persisted);
			} else {
				buckets.add(aggregateRange(
						connection,
						generation,
						runId,
						tagId,
						exactFrom,
						exactTo,
						selection.level() - 1));
			}
		}
		return buckets;
	}

	private LodBucket aggregateRange(
			Connection connection,
			String generation,
			String runId,
			long tagId,
			long ordinalFrom,
			long ordinalTo,
			int level) throws Exception {
		if (level <= 0) return aggregateL0(connection, tagId, ordinalFrom, ordinalTo);

		final long width = LodBucket.widthForLevel(level);
		final long firstBucket = ordinalFrom / width;
		final long lastBucket = (ordinalTo - 1L) / width;
		final AggregateBuilder aggregate = new AggregateBuilder();
		for (long bucket = firstBucket; bucket <= lastBucket; bucket++) {
			final long logicalFrom = Math.multiplyExact(bucket, width);
			final long logicalTo = Math.addExact(logicalFrom, width);
			final long exactFrom = Math.max(ordinalFrom, logicalFrom);
			final long exactTo = Math.min(ordinalTo, logicalTo);
			final LodBucket persisted = pageCache.find(
					connection, generation, runId, tagId, level, bucket);
			if (persisted != null && exactFrom == logicalFrom && exactTo == logicalTo) {
				aggregate.add(persisted);
			} else {
				aggregate.add(aggregateRange(
						connection,
						generation,
						runId,
						tagId,
						exactFrom,
						exactTo,
						level - 1));
			}
		}
		return aggregate.finish(ordinalFrom, ordinalTo);
	}

	private static LodBucket aggregateL0(
			Connection connection,
			long tagId,
			long ordinalFrom,
			long ordinalTo) throws Exception {
		final AggregateBuilder aggregate = new AggregateBuilder();
		try (PreparedStatement statement = connection.prepareStatement("""
				SELECT ordinal, step, value
				FROM scalars
				WHERE tag_id=? AND ordinal>=? AND ordinal<?
				ORDER BY ordinal
				""")) {
			statement.setLong(1, tagId);
			statement.setLong(2, ordinalFrom);
			statement.setLong(3, ordinalTo);
			try (ResultSet result = statement.executeQuery()) {
				while (result.next()) {
					aggregate.add(LodBucket.point(
							result.getLong(1),
							result.getLong(2),
							result.getDouble(3)));
				}
			}
		}
		return aggregate.finish(ordinalFrom, ordinalTo);
	}

	private static LevelSelection selectLevel(long ordinalFrom, long ordinalTo, int pointBudget) {
		int level = 1;
		long width = LodBucket.widthForLevel(level);
		while (bucketCount(ordinalFrom, ordinalTo, width)
				* LodBucket.POINTS_PER_LOD_BUCKET > pointBudget) {
			level++;
			width = LodBucket.widthForLevel(level);
		}
		return new LevelSelection(level, width);
	}

	private static long bucketCount(long ordinalFrom, long ordinalTo, long width) {
		return (ordinalTo - 1L) / width - ordinalFrom / width + 1L;
	}

	public record Projection(int level, long bucketWidth, Object body) {
	}

	private record LevelSelection(int level, long width) {
	}

	private record Candidate(long ordinal, long step, double value) {
	}

	private static final class AggregateBuilder {
		private long count;
		private long ordinalFrom;
		private long ordinalTo;
		private long stepFirst;
		private long stepLast;
		private long minOrdinal;
		private long minStep;
		private double minValue;
		private long maxOrdinal;
		private long maxStep;
		private double maxValue;
		private double mean;
		private double lastValue;

		private void add(LodBucket bucket) {
			if (count == 0L) {
				ordinalFrom = bucket.ordinalFrom();
				stepFirst = bucket.stepFirst();
				minOrdinal = bucket.minOrdinal();
				minStep = bucket.minStep();
				minValue = bucket.minValue();
				maxOrdinal = bucket.maxOrdinal();
				maxStep = bucket.maxStep();
				maxValue = bucket.maxValue();
				mean = bucket.mean();
			} else {
				if (bucket.minValue() < minValue
						|| (bucket.minValue() == minValue && bucket.minOrdinal() < minOrdinal)) {
					minOrdinal = bucket.minOrdinal();
					minStep = bucket.minStep();
					minValue = bucket.minValue();
				}
				if (bucket.maxValue() > maxValue
						|| (bucket.maxValue() == maxValue && bucket.maxOrdinal() < maxOrdinal)) {
					maxOrdinal = bucket.maxOrdinal();
					maxStep = bucket.maxStep();
					maxValue = bucket.maxValue();
				}
				final long combinedCount = count + bucket.count();
				mean += (bucket.mean() - mean) * bucket.count() / combinedCount;
			}
			count += bucket.count();
			ordinalTo = bucket.ordinalTo();
			stepLast = bucket.stepLast();
			lastValue = bucket.lastValue();
		}

		private LodBucket finish(long expectedFrom, long expectedTo) {
			if (count == 0L) throw new IllegalStateException("Cannot aggregate an empty ordinal range");
			if (ordinalFrom != expectedFrom || ordinalTo != expectedTo || count != expectedTo - expectedFrom) {
				throw new IllegalStateException("LOD aggregation produced a gap or overlap");
			}
			return new LodBucket(
					ordinalFrom,
					ordinalTo,
					count,
					stepFirst,
					stepLast,
					minOrdinal,
					minStep,
					minValue,
					maxOrdinal,
					maxStep,
					maxValue,
					mean,
					lastValue);
		}
	}
}
