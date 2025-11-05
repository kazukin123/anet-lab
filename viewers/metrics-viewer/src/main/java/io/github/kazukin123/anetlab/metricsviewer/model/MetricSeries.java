package io.github.kazukin123.anetlab.metricsviewer.model;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * One metric tag and its time series points.
 * Thread-safe: points list is synchronized.
 */
public class MetricSeries implements Serializable {
    private static final long serialVersionUID = 1L;

    private final String tag;
    private final List<MetricPoint> points = Collections.synchronizedList(new ArrayList<>());

    public MetricSeries(String tag) {
        this.tag = tag;
    }

    public String getTag() {
        return tag;
    }

    public void addPoint(MetricPoint p) {
        points.add(p);
    }

    /**
     * Returns a copy to avoid ConcurrentModificationException.
     */
    public List<MetricPoint> getPoints() {
        synchronized (points) {
            return new ArrayList<>(points);
        }
    }

    public int size() {
        synchronized (points) {
            return points.size();
        }
    }
}
