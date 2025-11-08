package io.github.kazukin123.anetlab.metricsviewer.service;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import io.github.kazukin123.anetlab.metricsviewer.model.MetricFileBlock;
import io.github.kazukin123.anetlab.metricsviewer.model.MetricTrace;
import io.github.kazukin123.anetlab.metricsviewer.model.MetricsSnapshot;
import io.github.kazukin123.anetlab.metricsviewer.model.Tag;

/**
 * Handles Serializable cache load/save per run.
 */
@Component
public class MetricsRepository {

    private static final Logger log = LoggerFactory.getLogger(MetricsRepository.class);
    private static final String SNAPSHOT_FILENAME = "metrics_snapshot.ser";
    
    
    // runId → snapshot(tag → (step,value))
    private final Map<String, MetricsSnapshot> snapshots = new ConcurrentHashMap<>();

    
    public List<MetricTrace> getTraces(List<String> runIds, List<String> tagKeys) {
		// TODO 自動生成されたメソッド・スタブ
    	return null;
	}
    
    public List<MetricTrace> getTracesDiff(List<String> runIds, List<String> tagKeys, List<Long> fromSteps) {
		// TODO 自動生成されたメソッド・スタブ
    	return null;
	}

    public void mergeMetrics(MetricFileBlock metricFileBlock) {
		// TODO 自動生成されたメソッド・スタブ
    }
    
    public void saveCache(String runId) {
		// TODO 自動生成されたメソッド・スタブ
    }

    public List<Tag> getTagsForRun(String runId) {
        MetricsSnapshot snapshot = snapshots.get(runId);
        if (snapshot == null) return Collections.emptyList();
        return snapshot.getTags(); // MetricsSnapshot側で集約済み
    }   
}
