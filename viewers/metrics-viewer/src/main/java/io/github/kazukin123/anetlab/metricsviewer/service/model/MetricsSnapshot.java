package io.github.kazukin123.anetlab.metricsviewer.service.model;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

import com.fasterxml.jackson.annotation.JsonIgnore;

import io.github.kazukin123.anetlab.metricsviewer.infra.model.MetricsFileBlock;
import io.github.kazukin123.anetlab.metricsviewer.infra.model.MetricsFileLine;
import io.github.kazukin123.anetlab.metricsviewer.view.model.RunStats;
import io.github.kazukin123.anetlab.metricsviewer.view.model.TagInfo;
import io.github.kazukin123.anetlab.metricsviewer.view.model.TagStats;

/**
 * Snapshot of metrics data for a single run.
 * Thread-safe structure used by MetricsRepository.
 */
public class MetricsSnapshot {

	private static final String TAG_TYPE_SCALER = "scalar";

	private final Map<TagInfo, List<Point>> tagValueMap = new ConcurrentHashMap<>(); // tagId → [step, value]
	private final Map<TagInfo, TagStats> tagStatsMap = new ConcurrentHashMap<>(); // tagId → TagStats
	private transient final Map<String, Object> tagLocks = new ConcurrentHashMap<>();
	private volatile long lastReadPosition = 0L;
	private RunStats runStats = new RunStats();

	/**
	 * MetricsFileBlockを統合。
	 * タグ単位にまとめてロックを取り、スレッド安全にPointと統計情報を更新。
	 */
	public void merge(MetricsFileBlock block) {
		if (block == null || block.getLines() == null) return;

		// scalarのみ抽出してタグ単位にまとめる
		final Map<String, List<MetricsFileLine>> grouped = block.getLines().stream()
				.filter(line -> line != null && line.getTag() != null)
				.filter(line -> TAG_TYPE_SCALER.equals(line.getType()))
				.collect(Collectors.groupingBy(MetricsFileLine::getTag));

		// タグ単位にロックしてまとめて追加
		for (Map.Entry<String, List<MetricsFileLine>> entry : grouped.entrySet()) {
			final String tagKey = entry.getKey();
			final List<MetricsFileLine> lines = entry.getValue();
			final Object lock = tagLocks.computeIfAbsent(tagKey, k -> new Object());

			synchronized (lock) {
				final TagInfo tagInfo = TagInfo.builder().key(tagKey).type(TAG_TYPE_SCALER).build();
				final List<Point> list = tagValueMap.computeIfAbsent(tagInfo, k -> new ArrayList<>());
				final TagStats stats = tagStatsMap.computeIfAbsent(tagInfo, k -> new TagStats());

				for (MetricsFileLine line : lines) {
					final int step = line.getStep();
					final float value = line.getValue();

					list.add(Point.builder().step(step).value(value).build());
					stats.record(step, value);

					if (step > runStats.getMaxStep()) {
						runStats.setMaxStep(step);
					}
				}
			}
		}

		this.lastReadPosition = block.getEndOffset();
	}

	/**
	 * 指定タグの指定step以降のデータを取得する。
	 * @param tagKey タグキー
	 * @param fromStep 取得開始ステップ（これより大きいstepのみ返す）
	 * @return 指定範囲のPointリスト（存在しない場合は空リスト）
	 */
	public List<Point> getPointsSince(String tagKey, int fromStep) {
	    synchronized (tagLocks.computeIfAbsent(tagKey, k -> new Object())) {
			if (tagKey == null) {
				return Collections.emptyList();
			}
	
			// tagKeyでPointを引く
			final TagInfo tag = new TagInfo(tagKey); // type固定で問題ない場合
			final List<Point> list = tagValueMap.get(tag);
			if (list == null || list.isEmpty()) {
				return Collections.emptyList();
			}
	
			// fromStep=0以下なら全件返す
			if (fromStep <= 0) {
				return list;
			}
	
			// 差分抽出（stream or subList）
			return list.stream()
					.filter(p -> p.getStep() > fromStep)
					.collect(Collectors.toList());
	    }
	}

	/**
	 * 指定タグの全体統計情報を返す（差分ではなく全体基準）。
	 */
	public TagStats getTagStats(String tagKey) {
	    if (tagKey == null) return null;
	    final Object lock = tagLocks.computeIfAbsent(tagKey, k -> new Object());
	    synchronized (lock) {
	        return tagStatsMap.get(new TagInfo(tagKey));
	    }
	}

	@JsonIgnore
	public List<TagInfo> getTags() {
		return new ArrayList<>(tagValueMap.keySet());
	}
	
	@JsonIgnore
	public List<String> listTagKeys() {
		return tagValueMap.keySet().stream().map(TagInfo::getKey).toList();
	}

	@JsonIgnore
	public RunStats getRunStats() {
		return runStats;
	}

	@JsonIgnore
	public long getLastReadPosition() {
		return lastReadPosition;
	}

	@JsonIgnore
	public void setLastReadPosition(long pos) {
		this.lastReadPosition = pos;
	}

	@JsonIgnore
	public long getTotalPoints() {
		return tagValueMap.values().stream().mapToLong(List::size).sum();
	}
}
