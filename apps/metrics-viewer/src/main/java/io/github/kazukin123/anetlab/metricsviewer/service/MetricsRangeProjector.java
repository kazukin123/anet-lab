package io.github.kazukin123.anetlab.metricsviewer.service;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

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

	/** 同一snapshotの完成保存bucketを使い、共通Step区間へ正確に射影する。 */
	public Projection project(
			Connection connection, String generation, String runId, long tagId,
			long tagCount, long ordinalFrom, long ordinalTo, int pointBudget,
			long stepBucketWidth, QueryExecution query) throws Exception {
		query.checkpoint();
		if (stepBucketWidth == 1L && ordinalTo - ordinalFrom <= pointBudget) {
			return raw(connection, tagId, ordinalFrom, ordinalTo, query);
		}

		// 序数範囲を、収まる最大の完成bucketへ分解する。表示区間幅で保存levelを選ばない。
		final Map<Long, AggregateBuilder> bins = new LinkedHashMap<>();
		try (PreparedStatement l0 = connection.prepareStatement("""
				SELECT ordinal, step, value FROM scalars
				WHERE tag_id=? AND ordinal>=? AND ordinal<? ORDER BY ordinal
				"""); StatementRegistration ignored = query.registerStatement(l0)) {
			l0.setLong(1, tagId);
			long position = ordinalFrom;
			while (position < ordinalTo) {
				query.checkpoint();
				int level = 0;
				long width = 1L;
				while (width <= (ordinalTo - position) / LodBucket.LOD_FACTOR
						&& position % (width * LodBucket.LOD_FACTOR) == 0L) {
					width *= LodBucket.LOD_FACTOR;
					level++;
				}
				if (level == 0) {
					// 次のL1境界まで、または未完成末尾をまとめて読む。
					final long end = Math.min(ordinalTo,
							position + LodBucket.LOD_FACTOR - position % LodBucket.LOD_FACTOR);
					readPoints(l0, position, end, stepBucketWidth, bins, query);
					position = end;
				} else {
					projectBucket(connection, generation, runId, tagId, tagCount,
							level, position / width, stepBucketWidth, bins, l0, query);
					position += width;
				}
			}
		}
		final List<LodBucket> buckets = bins.values().stream()
				.map(bin -> bin.finish(bin.ordinalFrom, bin.ordinalTo)).toList();
		return lod(stepBucketWidth, buckets, query);
	}

	private void projectBucket(
			Connection connection, String generation, String runId, long tagId, long tagCount,
			int level, long bucketIndex, long stepBucketWidth, Map<Long, AggregateBuilder> bins,
			PreparedStatement l0, QueryExecution query) throws Exception {
		query.checkpoint();
		final LodBucket bucket = pageCache.find(
				connection, generation, runId, tagId, tagCount, level, bucketIndex, query);
		if (bucket == null) throw new IllegalStateException(
				"Missing completed LOD bucket: tag=" + tagId + " level=" + level + " bucket=" + bucketIndex);
		final long bin = Math.floorDiv(bucket.stepFirst(), stepBucketWidth);
		if (bin == Math.floorDiv(bucket.stepLast(), stepBucketWidth)) {
			// 全点が同一区間にある完成bucketだけを再利用する。平均は点数で加重する。
			bins.computeIfAbsent(bin, ignored -> new AggregateBuilder()).add(bucket);
		} else if (level == 1) {
			readPoints(l0, bucket.ordinalFrom(), bucket.ordinalTo(), stepBucketWidth, bins, query);
		} else {
			// 境界を跨ぐbucketだけ下位へ降り、極値を別区間へ混ぜない。
			for (int child = 0; child < LodBucket.LOD_FACTOR; child++) {
				projectBucket(connection, generation, runId, tagId, tagCount, level - 1,
						bucketIndex * LodBucket.LOD_FACTOR + child, stepBucketWidth, bins, l0, query);
			}
		}
	}

	private static void readPoints(
			PreparedStatement l0, long from, long to, long width,
			Map<Long, AggregateBuilder> bins, QueryExecution query) throws Exception {
		l0.setLong(2, from);
		l0.setLong(3, to);
		try (ResultSet result = l0.executeQuery()) {
			long expected = from;
			while (result.next()) {
				query.checkpoint();
				final long ordinal = result.getLong(1);
				final long step = result.getLong(2);
				if (ordinal != expected++) throw new IllegalStateException("Missing L0 ordinal " + (expected - 1));
				bins.computeIfAbsent(Math.floorDiv(step, width), ignored -> new AggregateBuilder())
						.add(LodBucket.point(ordinal, step, result.getDouble(3)));
			}
			if (expected != to) throw new IllegalStateException("Missing L0 ordinal " + expected);
		}
	}

	private static Projection raw(
			Connection connection,
			long tagId,
			long ordinalFrom,
			long ordinalTo,
			QueryExecution query) throws Exception {
		final int size = Math.toIntExact(ordinalTo - ordinalFrom);
		final double[] steps = new double[size];
		final float[] values = new float[size];
		try (PreparedStatement statement = connection.prepareStatement("""
				SELECT step, value
				FROM scalars
				WHERE tag_id=? AND ordinal>=? AND ordinal<?
				ORDER BY ordinal
				""");
				StatementRegistration ignored = query.registerStatement(statement)) {
			statement.setLong(1, tagId);
			statement.setLong(2, ordinalFrom);
			statement.setLong(3, ordinalTo);
			try (ResultSet result = statement.executeQuery()) {
				int i = 0;
				while (result.next()) {
					query.checkpoint();
					if (i >= size) throw new IllegalStateException("Unexpected L0 rows: tag=" + tagId);
					steps[i] = result.getLong(1);
					values[i] = (float) result.getDouble(2);
					i++;
				}
				if (i != size) throw new IllegalStateException("Missing L0 rows: tag=" + tagId);
			}
		}
		return new Projection(
				1L,
				new RawProjection(
						MetricTraceEncoder.encodeDoubleArray(steps),
						MetricTraceEncoder.encodeFloatArray(values)));
	}

	private static Projection lod(
			long stepBucketWidth,
			List<LodBucket> buckets,
			QueryExecution query) {
		final List<Candidate> minMaxCandidates = new ArrayList<>();
		final double[] summarySteps = new double[buckets.size()];
		final float[] mins = new float[buckets.size()];
		final float[] maxs = new float[buckets.size()];
		final float[] means = new float[buckets.size()];
		final double[] minSteps = new double[buckets.size()];
		final double[] maxSteps = new double[buckets.size()];

		for (int i = 0; i < buckets.size(); i++) {
			query.checkpoint();
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

			summarySteps[i] = Math.floorDiv(bucket.stepFirst(), stepBucketWidth) * stepBucketWidth;
			mins[i] = (float) bucket.minValue();
			maxs[i] = (float) bucket.maxValue();
			means[i] = (float) bucket.mean();
			minSteps[i] = bucket.minStep();
			maxSteps[i] = bucket.maxStep();
		}

		final double[] minMaxSteps = new double[minMaxCandidates.size()];
		final float[] minMaxValues = new float[minMaxCandidates.size()];
		for (int i = 0; i < minMaxCandidates.size(); i++) {
			query.checkpoint();
			minMaxSteps[i] = minMaxCandidates.get(i).step();
			minMaxValues[i] = (float) minMaxCandidates.get(i).value();
		}

		return new Projection(
				stepBucketWidth,
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

	public record Projection(long stepBucketWidth, Object body) {
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
