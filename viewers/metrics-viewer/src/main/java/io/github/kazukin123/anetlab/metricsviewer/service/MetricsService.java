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

import io.github.kazukin123.anetlab.metricsviewer.infra.RunScanner;
import io.github.kazukin123.anetlab.metricsviewer.view.model.GetMetricsRequest;
import io.github.kazukin123.anetlab.metricsviewer.view.model.GetMetricsResponse;
import io.github.kazukin123.anetlab.metricsviewer.view.model.GetRunsResponse;
import io.github.kazukin123.anetlab.metricsviewer.view.model.MetricTrace;
import io.github.kazukin123.anetlab.metricsviewer.view.model.Run;
import io.github.kazukin123.anetlab.metricsviewer.view.model.RunStats;
import io.github.kazukin123.anetlab.metricsviewer.view.model.Tag;

/**
 * Main service layer that connects Controller with Repository & LoadingThread.
 */
@Service
public class MetricsService {

    private static final Logger log = LoggerFactory.getLogger(MetricsService.class);

    private final RunScanner runScanner;
    private final MetricsRepository metricsRepository;
    private final LoadingThread loadingThread;

    @Autowired
    public MetricsService(RunScanner runScanner,
                          MetricsRepository metricsRepository,
                          LoadingThread loadingThread) {
        this.runScanner = runScanner;
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
        List<String> runIds = runScanner.getRunIds();

        // Run情報を生成
        List<Run> runs = new ArrayList<>();
        for (String runId : runIds) {
            RunStats stats = metricsRepository.getStats(runId);
            List<Tag> tags = metricsRepository.getTagsForRun(runId);
            Run run = Run.builder().id(runId).stats(stats).tags(tags).build();
            runs.add(run);
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
            runIds = runScanner.getRunIds();
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
        res.setData(traces);

        log.debug("getMetrics: runs={} tags={} traces={}", runIds, tagKeys, traces.size());
        return res;
    }
}
