package io.github.kazukin123.anetlab.metricsviewer.service.model;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import com.fasterxml.jackson.annotation.JsonIgnore;

import io.github.kazukin123.anetlab.metricsviewer.infra.model.MetricFileBlock;
import io.github.kazukin123.anetlab.metricsviewer.infra.model.MetricLine;
import io.github.kazukin123.anetlab.metricsviewer.view.model.MetricTrace;
import io.github.kazukin123.anetlab.metricsviewer.view.model.Tag;

/**
 * Snapshot of metrics data for a single run.
 * Thread-safe structure used by MetricsRepository.
 */
public class MetricsSnapshot implements Serializable {

    private static final long serialVersionUID = 2L;

    /** tag → [step, value] */
    private final Map<Tag, List<MetricPoint>> tagValues = new ConcurrentHashMap<>();

    /** Last read byte offset from metrics.jsonl */
    private volatile long lastReadPosition = 0L;

    /** Merges a parsed MetricFileBlock into this snapshot. */
    public void merge(MetricFileBlock block) {
        if (block == null || block.getLines() == null) return;

        for (MetricLine line : block.getLines()) {
            if (line == null || line.getTag() == null) continue;

            Tag tag = new Tag(line.getTag(), line.getType());
            MetricPoint point = MetricPoint.builder()
                    .step(line.getStep())
                    .value(line.getValue()) // MetricLine.values は double 型
                    .build();

            // スレッド安全なリスト操作
            tagValues.computeIfAbsent(tag, k -> new ArrayList<>()).add(point);
//            tagValues.computeIfAbsent(tag, k ->
//                    Collections.synchronizedList(new ArrayList<>())
//            ).add(point);
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
                for (MetricPoint p : points) {
                    steps.add((int) p.getStep());
                    values.add(p.getValue());
                }

                MetricTrace trace = MetricTrace.builder()
                        .runId(null) // Repository 側でRunIdを付与する想定
                        .tagKey(tag.getKey())
                        .type(tag.getType())
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
