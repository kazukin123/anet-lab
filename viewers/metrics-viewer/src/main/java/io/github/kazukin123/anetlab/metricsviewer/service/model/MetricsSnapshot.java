package io.github.kazukin123.anetlab.metricsviewer.service.model;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import com.fasterxml.jackson.annotation.JsonIgnore;

import io.github.kazukin123.anetlab.metricsviewer.infra.model.MetricFileBlock;
import io.github.kazukin123.anetlab.metricsviewer.infra.model.MetricLine;
import io.github.kazukin123.anetlab.metricsviewer.view.model.MetricTrace;
import io.github.kazukin123.anetlab.metricsviewer.view.model.RunStats;
import io.github.kazukin123.anetlab.metricsviewer.view.model.Tag;
import io.github.kazukin123.anetlab.metricsviewer.view.model.TagStats;

/**
 * Snapshot of metrics data for a single run.
 * Thread-safe structure used by MetricsRepository.
 */
public class MetricsSnapshot {
	
	private static final String TAG_TYPE_SCALER = "scalar";

    private final Map<Tag, List<Point>> tagValues = new ConcurrentHashMap<>();    // tag → [step, value]
    private final Map<Tag, TagStats> tagStats = new ConcurrentHashMap<>();

    /** Last read byte offset from metrics.jsonl */
    private volatile long lastReadPosition = 0L;
    private RunStats runStats = new RunStats();

    /** Merges a parsed MetricFileBlock into this snapshot. */
    public void merge(MetricFileBlock block) {
        if (block == null || block.getLines() == null) return;

        for (MetricLine line : block.getLines()) {
            if (line == null || line.getTag() == null) continue;

            // scaler以外は現状非サポート
            if (!TAG_TYPE_SCALER.equals(line.getType())) continue;


            long step = line.getStep();
            double value = line.getValue();

            // Run単位の統計計算
            if (step > runStats.getMaxStep()) runStats.setMaxStep(step);
            
            // Scalerタグ内容を登録
            Tag tag = new Tag(line.getTag(), line.getType());
            Point point = Point.builder()
                    .step(line.getStep())
                    .value(line.getValue()) // MetricLine.values は double 型
                    .build();

            // スレッド安全なリスト操作
            tagValues.computeIfAbsent(tag, k -> new ArrayList<>()).add(point);
            tagStats.computeIfAbsent(tag, k -> new TagStats()).record(step, value);
        }

        this.lastReadPosition = block.getEndOffset();
    }

    /** Converts stored points into traces for the specified tag keys. */
    public List<MetricTrace> getMetricsTrace(List<String> tagKeys) {
        List<MetricTrace> traces = new ArrayList<>();

        tagValues.forEach((tag, points) -> {
            if (tagKeys == null || tagKeys.isEmpty() || tagKeys.contains(tag.getKey())) {
                // step, value の順序を保持
                List<Integer> steps = new ArrayList<>(points.size());
                List<Double> values = new ArrayList<>(points.size());
                for (Point p : points) {
                    steps.add((int) p.getStep());
                    values.add(p.getValue());
                }

                TagStats stats = tagStats != null ? tagStats.get(tag) : null;

                MetricTrace trace = MetricTrace.builder()
                        .runId(null) // Repository 側でRunIdを付与する想定
                        .tagKey(tag.getKey())
                        .type(tag.getType())
                        .stats(stats)
                        .steps(steps)
                        .values(values)
                        .build();

                traces.add(trace);
            }
        });

        return traces;
    }

    /** Returns current tag list for this snapshot. */
    public List<Tag> getTags() {
        return new ArrayList<>(tagValues.keySet());
    }
    
    public RunStats getStats() {
    	return runStats;
    }

    public long getLastReadPosition() {
        return lastReadPosition;
    }

    public void setLastReadPosition(long pos) {
        this.lastReadPosition = pos;
    }

    @JsonIgnore
    public long getTotalPoints() {
	    return tagValues.values().stream().mapToLong(List::size).sum();    
	}
}
