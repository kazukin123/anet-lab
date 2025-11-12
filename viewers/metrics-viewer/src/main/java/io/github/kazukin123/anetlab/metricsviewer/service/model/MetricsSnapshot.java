package io.github.kazukin123.anetlab.metricsviewer.service.model;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import com.fasterxml.jackson.annotation.JsonIgnore;

import io.github.kazukin123.anetlab.metricsviewer.infra.model.MetricsFileBlock;
import io.github.kazukin123.anetlab.metricsviewer.infra.model.MetricsFileLine;
import io.github.kazukin123.anetlab.metricsviewer.view.model.RunStats;
import io.github.kazukin123.anetlab.metricsviewer.view.model.TagInfo;
import io.github.kazukin123.anetlab.metricsviewer.view.model.TagStats;
import io.github.kazukin123.anetlab.metricsviewer.view.model.TagTrace;

/**
 * Snapshot of metrics data for a single run.
 * Thread-safe structure used by MetricsRepository.
 */
public class MetricsSnapshot {

	private static final String TAG_TYPE_SCALER = "scalar";

	private final Map<TagInfo, List<Point>> tagValueMap = new ConcurrentHashMap<>(); // tagId → [step, value]
	private final Map<TagInfo, TagStats> tagStatsMap = new ConcurrentHashMap<>(); // tagId → TagStats
	private volatile long lastReadPosition = 0L;
	private RunStats runStats = new RunStats();

	public void merge(MetricsFileBlock block) {
		if (block == null || block.getLines() == null) return;

		for (MetricsFileLine line : block.getLines()) {
			if (line == null || line.getTag() == null) continue;

			// scaler以外は現状非サポート
			if (!TAG_TYPE_SCALER.equals(line.getType())) continue;

			final int step = line.getStep();
			final float value = line.getValue();

			// Run単位の統計計算
			if (step > runStats.getMaxStep()) {
				runStats.setMaxStep(step);
			}

			// Scalerタグ内容を登録
			final TagInfo tagInfo = TagInfo.builder().key(line.getTag()).type(line.getType()).build();
			final Point point = Point.builder()
					.step(line.getStep())
					.value(line.getValue()) // MetricLine.values は double 型
					.build();

			// スレッド安全なリスト操作
			tagValueMap.computeIfAbsent(tagInfo, k -> new ArrayList<>()).add(point);
			tagStatsMap.computeIfAbsent(tagInfo, k -> new TagStats()).record(step, value);
		}

		this.lastReadPosition = block.getEndOffset();
	}

	public List<TagTrace> findTagTrace(List<String> tagKeys) {
		final List<TagTrace> tagTraceList = new ArrayList<>();

		tagValueMap.forEach((tagInfo, points) -> {
			if (tagKeys == null || tagKeys.isEmpty() || tagKeys.contains(tagInfo.getKey())) {
				// step, value の順序を保持して詰め直し
				final int steps[] = new int[points.size()];
				final float values[] = new float[points.size()];
				final int size = points.size();
				for (int i = 0; i < size; i++) {
					final Point p = points.get(i);
					steps[i] = p.getStep();
					values[i] = p.getValue();
				}

				// stats取得
				final TagStats tagStats = tagStatsMap.get(tagInfo);

				// Tag単位の結果「trace」としてまとめ
				final TagTrace tagData = TagTrace.builder()
						.runId(null) // Repository 側でRunIdを付与する想定
						.tagKey(tagInfo.getKey())
						.type(tagInfo.getType())
						.stats(tagStats)
						.steps(steps)
						.values(values)
						.build();

				tagTraceList.add(tagData);
			}
		});

		return tagTraceList;
	}

	@JsonIgnore
	public List<TagInfo> getTags() {
		return new ArrayList<>(tagValueMap.keySet());
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
