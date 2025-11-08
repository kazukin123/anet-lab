package io.github.kazukin123.anetlab.metricsviewer.service;

import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import io.github.kazukin123.anetlab.metricsviewer.dto.GetMetricsRequest;
import io.github.kazukin123.anetlab.metricsviewer.dto.GetMetricsResponse;
import io.github.kazukin123.anetlab.metricsviewer.dto.GetRunsResponse;

/**
 * Core service coordinating async metric loading and API access.
 * Thread-safe snapshot access.
 */
@Service
public class MetricsService {

    private static final Logger log = LoggerFactory.getLogger(MetricsService.class);

    private final RunRepository runRepository;
    private final MetricsRepository metricsRepository;
    private final LoadingThread loadingThread;

    @Autowired
    public MetricsService(
            RunRepository runRepository,
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
        loadingThread.interrupt(); // または loadingThread.terminate();
    }

    public GetRunsResponse getRuns() {
		// TODO 自動生成されたメソッド・スタブ
		return null;
	}

	public GetMetricsResponse getMetrics(GetMetricsRequest req) {
		// TODO 自動生成されたメソッド・スタブ
		return null;
	}
}
