package io.github.kazukin123.anetlab.metricsviewer.service;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import io.github.kazukin123.anetlab.metricsviewer.dto.GetMetricsRequest;
import io.github.kazukin123.anetlab.metricsviewer.dto.GetMetricsResponse;
import io.github.kazukin123.anetlab.metricsviewer.dto.GetRunsResponse;
import io.github.kazukin123.anetlab.metricsviewer.model.MetricTrace;
import io.github.kazukin123.anetlab.metricsviewer.model.Run;
import io.github.kazukin123.anetlab.metricsviewer.model.Tag;

/**
 * Main service layer that connects Controller with Repository & LoadingThread.
 */
@Service
public class MetricsService {

    private static final Logger log = LoggerFactory.getLogger(MetricsService.class);

    private final RunRepository runRepository;
    private final MetricsRepository metricsRepository;
    private final LoadingThread loadingThread;

    @Autowired
    public MetricsService(RunRepository runRepository,
                          MetricsRepository metricsRepository,
                          LoadingThread loadingThread) {
        this.runRepository = runRepository;
        this.metricsRepository = metricsRepository;
        this.loadingThread = loadingThread;
    }

    @PostConstruct
    private void initialize() {
        log.info("MetricsService initialized. Starting LoadingThread.");
        loadingThread.start();
    }

    @PreDestroy
    private void shutdown() {
        log.info("Stopping LoadingThread.");
        loadingThread.terminate();
    }

    /**
     * Returns run list with tags (used by /api/runs).
     */
    public GetRunsResponse getRuns() {
        List<Run> runs = runRepository.getRuns();

        // 各Runにタグを補完
        for (Run run : runs) {
            List<Tag> tags = metricsRepository.getTagsForRun(run.getId());
            run.setTags(tags);
        }

        GetRunsResponse res = new GetRunsResponse();
        res.setRuns(runs);

        log.debug("getRuns: {} runs", runs.size());
        return res;
    }

    /**
     * Returns metric traces (used by /api/metrics).
     */
    public GetMetricsResponse getMetrics(GetMetricsRequest req) {
        if (req == null) return new GetMetricsResponse();

        List<String> runIds = req.getRunIds();
        List<String> tagKeys = req.getTagKeys();

        // --- ① 空指定時は全件扱いに補完 ---
        if (runIds == null || runIds.isEmpty()) {
            runIds = new ArrayList<>();
            for (Run r : runRepository.getRuns()) {
                runIds.add(r.getId());
            }
        }

        if (tagKeys == null || tagKeys.isEmpty()) {
            // 全RunのTagを走査して一意にまとめる
            Set<String> allTags = new LinkedHashSet<>();
            for (String runId : runIds) {
                List<Tag> tags = metricsRepository.getTagsForRun(runId);
                for (Tag tag : tags) {
                    allTags.add(tag.getKey());
                }
            }
            tagKeys = new ArrayList<>(allTags);
        }

        // --- ② ロードリクエスト発行（非同期） ---
        loadingThread.request(runIds, tagKeys, 1000);

        // --- ③ 現在のキャッシュを返す ---
        List<MetricTrace> traces = metricsRepository.getTraces(runIds, tagKeys);

        GetMetricsResponse res = new GetMetricsResponse();
        res.setData(traces); // ← フロント仕様に合わせた命名
        log.debug("getMetrics: runs={} tags={} traces={}", runIds, tagKeys, traces.size());
        return res;
    }
}
