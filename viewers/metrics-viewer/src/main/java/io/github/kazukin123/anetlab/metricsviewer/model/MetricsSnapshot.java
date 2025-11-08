package io.github.kazukin123.anetlab.metricsviewer.model;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class MetricsSnapshot implements Serializable {
	
    private static final long serialVersionUID = 2L;

    /** tag → [step, value] */
    private final Map<Tag, List<MetricPoint>> tagValues = new ConcurrentHashMap<>();

    private volatile long lastReadPosition = 0L;

    
    public void merge(MetricFileBlock block) {
    }

    public List<MetricTrace> getMetricsTrace(List<String> tagKeys) {
        return null;
    }


    public long getLastReadPosition() {
        return lastReadPosition;
    }

    public void setLastReadPosition(long pos) {
        this.lastReadPosition = pos;
    }

	public List<Tag> getTags() {
		return new ArrayList<>(tagValues.keySet());
	}
}
