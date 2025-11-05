package io.github.kazukin123.anetlab.metricsviewer.model;

import java.io.Serializable;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Snapshot of all metrics within a run.
 * Thread-safe for concurrent read/write access from async loader and API threads.
 */
public class MetricsSnapshot implements Serializable {
    private static final long serialVersionUID = 1L;

    /** tag → series */
    private final Map<String, MetricSeries> series = new ConcurrentHashMap<>();

    /** last read position in metrics.jsonl */
    private volatile long lastReadPosition = 0L;

    public Map<String, MetricSeries> getSeries() {
        return series;
    }

    public long getLastReadPosition() {
        return lastReadPosition;
    }

    public void setLastReadPosition(long pos) {
        this.lastReadPosition = pos;
    }

    /**
     * Merge new entries into the existing snapshot.
     * This method is synchronized to ensure visibility between threads.
     */
    public synchronized void merge(List<MetricEntry> entries) {
        for (MetricEntry e : entries) {
            MetricSeries s = series.computeIfAbsent(e.getTag(), MetricSeries::new);
            s.addPoint(new MetricPoint(e.getStep(), e.getValue(), e.getTimestamp()));
        }
    }
}
