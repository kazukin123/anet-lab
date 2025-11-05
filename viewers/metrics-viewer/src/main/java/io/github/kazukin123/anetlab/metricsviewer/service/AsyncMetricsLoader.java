package io.github.kazukin123.anetlab.metricsviewer.service;

import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import jakarta.annotation.PreDestroy;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsFileReader;
import io.github.kazukin123.anetlab.metricsviewer.model.MetricEntry;
import io.github.kazukin123.anetlab.metricsviewer.model.MetricsSnapshot;
import io.github.kazukin123.anetlab.metricsviewer.model.RunInfo;

@Component
public class AsyncMetricsLoader implements AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(AsyncMetricsLoader.class);
    private final ExecutorService executor = Executors.newSingleThreadExecutor();

    public void submitDiffLoad(
            final RunInfo run,
            final MetricsSnapshot snapshot,
            final MetricsFileReader reader,
            final CacheManager cacheManager) {

        executor.submit(() -> {
            try {
                List<MetricEntry> newEntries = reader.readNewEntries(run, run.getJsonlPath());
                if (!newEntries.isEmpty()) {
                    synchronized (snapshot) {
                        snapshot.merge(newEntries);
                        snapshot.setLastReadPosition(run.getLastReadPosition());
                    }
                    cacheManager.save(run, snapshot);
                    log.info("Async diff load: {} new entries for run {}", newEntries.size(), run.getRunId());
                } else {
                    log.debug("Async diff load: no new entries for {}", run.getRunId());
                }
            } catch (Exception e) {
                log.warn("Async diff load failed for {}: {}", run.getRunId(), e.getMessage());
            }
        });
    }

    public void submitFullLoad(
            final RunInfo run,
            final MetricsFileReader reader,
            final CacheManager cacheManager,
            final java.util.function.Consumer<MetricsSnapshot> consumer) {

        executor.submit(() -> {
            try {
                MetricsSnapshot snap = reader.parseFull(run);
                cacheManager.save(run, snap);
                consumer.accept(snap);
                log.info("Async full load completed for run {}", run.getRunId());
            } catch (Exception e) {
                log.warn("Async full load failed for {}: {}", run.getRunId(), e.getMessage());
            }
        });
    }

    @PreDestroy
    @Override
    public void close() {
        log.info("Shutting down AsyncMetricsLoader...");
        executor.shutdown();
    }
}
