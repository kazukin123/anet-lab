package io.github.kazukin123.anetlab.metricsviewer.service;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsFileReader;
import io.github.kazukin123.anetlab.metricsviewer.model.MetricSeries;
import io.github.kazukin123.anetlab.metricsviewer.model.MetricsSnapshot;
import io.github.kazukin123.anetlab.metricsviewer.model.RunInfo;

/**
 * Core service coordinating async metric loading and API access.
 * Thread-safe snapshot access.
 */
@Service
public class MetricsService {

    private static final Logger log = LoggerFactory.getLogger(MetricsService.class);

    private final RunRepository runRepository;
    private final CacheManager cacheManager;
    private final MetricsFileReader fileReader;
    private final AsyncMetricsLoader asyncLoader;

    private final Map<String, MetricsSnapshot> snapshots = new ConcurrentHashMap<>();

    public MetricsService(
            RunRepository runRepository,
            CacheManager cacheManager,
            MetricsFileReader fileReader,
            AsyncMetricsLoader asyncLoader) {

        this.runRepository = runRepository;
        this.cacheManager = cacheManager;
        this.fileReader = fileReader;
        this.asyncLoader = asyncLoader;
        initialize();
    }

    private void initialize() {
        log.info("Initializing MetricsService (thread-safe async mode)...");
        runRepository.scanRuns();
        for (RunInfo run : runRepository.getAll()) {
            Path jsonl = run.getJsonlPath();
            MetricsSnapshot cached = cacheManager.load(run);
            boolean needFull = (cached == null);
            boolean outdated = false;
            try {
                Path serPath = cacheManager.getCachePath(run);
                long jsonlTime = Files.getLastModifiedTime(jsonl).toMillis();
                long serTime = Files.exists(serPath)
                        ? Files.getLastModifiedTime(serPath).toMillis()
                        : 0L;
                outdated = jsonlTime > serTime;
            } catch (IOException ignore) {}

            if (needFull) {
                log.info("No cache for {}. Submitting async full load.", run.getRunId());
                asyncLoader.submitFullLoad(run, fileReader, cacheManager, s -> snapshots.put(run.getRunId(), s));
            } else {
                snapshots.put(run.getRunId(), cached);
                if (outdated) {
                    log.info("Submitting async diff load for outdated run {}", run.getRunId());
                    asyncLoader.submitDiffLoad(run, cached, fileReader, cacheManager);
                } else {
                    log.info("Cache up-to-date for run {}", run.getRunId());
                }
            }
        }
        log.info("MetricsService async initialization complete ({} runs)", runRepository.getAll().size());
    }

    public List<RunInfo> listRuns() {
        return runRepository.getAll();
    }

    /**
     * Fetches current series safely. Async loader may update in parallel.
     */
    public List<MetricSeries> fetchSeries(String runId, Optional<String> tagOpt) {
        RunInfo run = runRepository.find(runId)
                .orElseThrow(() -> new IllegalArgumentException("Unknown runId: " + runId));
        MetricsSnapshot snapshot = snapshots.computeIfAbsent(runId, id -> new MetricsSnapshot());

        asyncLoader.submitDiffLoad(run, snapshot, fileReader, cacheManager);

        // --- 同期ブロックで整合性を確保 ---
        synchronized (snapshot) {
            if (tagOpt.isPresent()) {
                String tag = tagOpt.get();
                MetricSeries s = snapshot.getSeries().get(tag);
                if (s == null) return Collections.emptyList();
                return Collections.singletonList(s);
            }
            return new ArrayList<>(snapshot.getSeries().values());
        }
    }
}
