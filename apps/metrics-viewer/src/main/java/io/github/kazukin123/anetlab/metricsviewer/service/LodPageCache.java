package io.github.kazukin123.anetlab.metricsviewer.service;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import io.github.kazukin123.anetlab.metricsviewer.config.MetricsViewerSettings;

public class LodPageCache {

	static final int LOD_PAGE_BUCKETS = 1024;
	private static final int LONG_FIELDS = 8;
	private static final int DOUBLE_FIELDS = 4;

	private final long capacityBytes;
	private final Map<PageKey, Page> pages = new LinkedHashMap<>(16, 0.75f, true);
	private long usedBytes;

	public LodPageCache(MetricsViewerSettings settings) {
		this.capacityBytes = settings.getCacheMemoryBytes();
	}

	/**
	 * snapshotで完成済みのbucketを返す。未完成のbucketならnullを返し、呼び出し側が下位levelから再集約する。
	 * tagCountはconnectionと同じsnapshotで読んだtagの全点数とする。
	 */
	LodBucket find(
			Connection connection,
			String generation,
			String runId,
			long tagId,
			long tagCount,
			int level,
			long bucket,
			QueryExecution query) throws SQLException {
		query.checkpoint();
		// 子16件がそろった瞬間に親を書くので、完成bucketは各levelの先頭から「全点数/幅」個だけある。
		// それより後ろは未完成なので、DBを引かずに再集約へ回す。
		final long width = LodBucket.widthForLevel(level);
		final long completeBuckets = tagCount / width;
		if (bucket >= completeBuckets) return null;

		final LodBucket found;
		if (capacityBytes == 0L) {
			found = loadBucket(connection, tagId, level, bucket, query);
		} else {
			final long pageIndex = Math.floorDiv(bucket, LOD_PAGE_BUCKETS);
			found = pageFor(
					connection, generation, runId, tagId, level, pageIndex, completeBuckets, query)
					.find(bucket, width, query);
		}
		if (found == null) {
			throw new IllegalStateException("Complete LOD bucket is missing: run=" + runId
					+ " tagId=" + tagId + " level=" + level + " bucket=" + bucket
					+ " completeBuckets=" + completeBuckets);
		}
		return found;
	}

	private Page pageFor(
			Connection connection,
			String generation,
			String runId,
			long tagId,
			int level,
			long pageIndex,
			long completeBuckets,
			QueryExecution query) throws SQLException {
		// このsnapshotでpageに入っているべきbucket数。末尾pageだけが1024未満になる。
		final long firstBucket = Math.multiplyExact(pageIndex, LOD_PAGE_BUCKETS);
		final int expectedBuckets = (int) Math.min(LOD_PAGE_BUCKETS, completeBuckets - firstBucket);
		final PageKey key = new PageKey(generation, runId, tagId, level, pageIndex);
		Page page;
		synchronized (this) {
			page = pages.get(key);
		}
		// 完成bucketは書き換わらず末尾へ増えるだけなので、保持中のpageが足りていればそのまま使える。
		// 足りないのは、保持後に末尾pageで新しいbucketが完成した場合だけである。
		if (page != null && page.size() >= expectedBuckets) return page;

		final Page loaded = loadPage(connection, tagId, level, pageIndex, query);
		if (loaded.size() < expectedBuckets) {
			throw new IllegalStateException("LOD page is missing complete buckets: run=" + runId
					+ " tagId=" + tagId + " level=" + level + " page=" + pageIndex
					+ " expected=" + expectedBuckets + " actual=" + loaded.size());
		}
		return retain(key, loaded);
	}

	public synchronized void retainRuns(Set<String> runIds) {
		final Iterator<Map.Entry<PageKey, Page>> iterator = pages.entrySet().iterator();
		while (iterator.hasNext()) {
			final Map.Entry<PageKey, Page> entry = iterator.next();
			if (!runIds.contains(entry.getKey().runId())) {
				usedBytes -= entry.getValue().byteSize();
				iterator.remove();
			}
		}
	}

	public synchronized void invalidateGeneration(String runId, String generation) {
		final Iterator<Map.Entry<PageKey, Page>> iterator = pages.entrySet().iterator();
		while (iterator.hasNext()) {
			final Map.Entry<PageKey, Page> entry = iterator.next();
			if (entry.getKey().runId().equals(runId)
					&& !entry.getKey().generation().equals(generation)) {
				usedBytes -= entry.getValue().byteSize();
				iterator.remove();
			}
		}
	}

	synchronized int pageCount() {
		return pages.size();
	}

	synchronized long usedBytes() {
		return usedBytes;
	}

	private static LodBucket loadBucket(
			Connection connection,
			long tagId,
			int level,
			long bucket,
			QueryExecution query) throws SQLException {
		try (PreparedStatement statement = connection.prepareStatement("""
				SELECT bucket, cnt, step_first, step_last,
				       min_ordinal, min_step, vmin,
				       max_ordinal, max_step, vmax, vmean, vlast
				FROM scalars_lod
				WHERE tag_id=? AND level=? AND bucket=?
				""");
				StatementRegistration ignored = query.registerStatement(statement)) {
			statement.setLong(1, tagId);
			statement.setInt(2, level);
			statement.setLong(3, bucket);
			try (ResultSet result = statement.executeQuery()) {
				query.checkpoint();
				return result.next() ? LodBucket.fromLodRow(result, level) : null;
			}
		}
	}

	private Page retain(PageKey key, Page loaded) {
		synchronized (this) {
			// 同じpageはbucketが末尾へ増えるだけなので、bucket数の多い方が新しい内容を含む。
			// 古いsnapshotで読んだpageが、並行queryの保持した新しいpageを縮めないようにする。
			final Page existing = pages.get(key);
			if (existing != null && existing.size() >= loaded.size()) return existing;
			if (existing != null) {
				pages.remove(key);
				usedBytes -= existing.byteSize();
			}
			if (loaded.byteSize() == 0L) return loaded;
			if (loaded.byteSize() > capacityBytes) return loaded;

			while (!pages.isEmpty() && usedBytes + loaded.byteSize() > capacityBytes) {
				final Iterator<Map.Entry<PageKey, Page>> iterator = pages.entrySet().iterator();
				final Map.Entry<PageKey, Page> eldest = iterator.next();
				usedBytes -= eldest.getValue().byteSize();
				iterator.remove();
			}
			pages.put(key, loaded);
			usedBytes += loaded.byteSize();
			return loaded;
		}
	}

	private static Page loadPage(
			Connection connection,
			long tagId,
			int level,
			long pageIndex,
			QueryExecution query) throws SQLException {
		final long firstBucket = Math.multiplyExact(pageIndex, LOD_PAGE_BUCKETS);
		final long lastBucketExclusive = Math.addExact(firstBucket, LOD_PAGE_BUCKETS);
		final List<LodBucket> rows = new ArrayList<>();
		try (PreparedStatement statement = connection.prepareStatement("""
				SELECT bucket, cnt, step_first, step_last,
				       min_ordinal, min_step, vmin,
				       max_ordinal, max_step, vmax, vmean, vlast
				FROM scalars_lod
				WHERE tag_id=? AND level=? AND bucket>=? AND bucket<?
				ORDER BY bucket
				""");
				StatementRegistration ignored = query.registerStatement(statement)) {
			statement.setLong(1, tagId);
			statement.setInt(2, level);
			statement.setLong(3, firstBucket);
			statement.setLong(4, lastBucketExclusive);
			try (ResultSet result = statement.executeQuery()) {
				while (result.next()) {
					query.checkpoint();
					rows.add(LodBucket.fromLodRow(result, level));
				}
			}
		}
		return Page.from(rows, LodBucket.widthForLevel(level), query);
	}

	private record PageKey(
			String generation,
			String runId,
			long tagId,
			int level,
			long pageIndex) {
	}

	private static final class Page {
		private final long[] buckets;
		private final long[] counts;
		private final long[] stepFirsts;
		private final long[] stepLasts;
		private final long[] minOrdinals;
		private final long[] minSteps;
		private final long[] maxOrdinals;
		private final long[] maxSteps;
		private final double[] minValues;
		private final double[] maxValues;
		private final double[] means;
		private final double[] lastValues;

		private Page(int size) {
			buckets = new long[size];
			counts = new long[size];
			stepFirsts = new long[size];
			stepLasts = new long[size];
			minOrdinals = new long[size];
			minSteps = new long[size];
			maxOrdinals = new long[size];
			maxSteps = new long[size];
			minValues = new double[size];
			maxValues = new double[size];
			means = new double[size];
			lastValues = new double[size];
		}

		private static Page from(List<LodBucket> rows, long width, QueryExecution query) {
			final Page page = new Page(rows.size());
			for (int i = 0; i < rows.size(); i++) {
				query.checkpoint();
				final LodBucket row = rows.get(i);
				page.buckets[i] = row.ordinalFrom() / width;
				page.counts[i] = row.count();
				page.stepFirsts[i] = row.stepFirst();
				page.stepLasts[i] = row.stepLast();
				page.minOrdinals[i] = row.minOrdinal();
				page.minSteps[i] = row.minStep();
				page.minValues[i] = row.minValue();
				page.maxOrdinals[i] = row.maxOrdinal();
				page.maxSteps[i] = row.maxStep();
				page.maxValues[i] = row.maxValue();
				page.means[i] = row.mean();
				page.lastValues[i] = row.lastValue();
			}
			return page;
		}

		private LodBucket find(long bucket, long width, QueryExecution query) {
			int low = 0;
			int high = buckets.length - 1;
			while (low <= high) {
				query.checkpoint();
				final int middle = (low + high) >>> 1;
				if (buckets[middle] < bucket) low = middle + 1;
				else if (buckets[middle] > bucket) {
					high = middle - 1;
				} else {
					final long ordinalFrom = Math.multiplyExact(bucket, width);
					return new LodBucket(
							ordinalFrom,
							ordinalFrom + counts[middle],
							counts[middle],
							stepFirsts[middle],
							stepLasts[middle],
							minOrdinals[middle],
							minSteps[middle],
							minValues[middle],
							maxOrdinals[middle],
							maxSteps[middle],
							maxValues[middle],
							means[middle],
							lastValues[middle]);
				}
			}
			return null;
		}

		private int size() {
			return buckets.length;
		}

		private long byteSize() {
			return (long) buckets.length
					* (LONG_FIELDS * Long.BYTES + DOUBLE_FIELDS * Double.BYTES);
		}
	}
}
