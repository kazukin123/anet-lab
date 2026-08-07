package io.github.kazukin123.anetlab.metricsviewer.service;

import java.util.List;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;

import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;

import io.github.kazukin123.anetlab.metricsviewer.config.MetricsViewerSettings;
import io.github.kazukin123.anetlab.metricsviewer.infra.RunScanner;
import io.github.kazukin123.anetlab.metricsviewer.view.model.GetMetricsRequest;
import io.github.kazukin123.anetlab.metricsviewer.view.model.GetMetricsResponse;
import io.github.kazukin123.anetlab.metricsviewer.view.model.GetRunsResponse;
import io.github.kazukin123.anetlab.metricsviewer.view.model.RunInfo;
import io.github.kazukin123.anetlab.metricsviewer.view.model.MetricsSeriesRequest;
import io.github.kazukin123.anetlab.metricsviewer.view.model.PrioritizeRunsRequest;

@Service
public class MetricsService {

	private static final long SHUTDOWN_WAIT_TIMEOUT_MS = 30_000L;
	private static final Logger log = LoggerFactory.getLogger(MetricsService.class);

	private final RunScanner runScanner;
	private final MetricsRepository metricsRepository;
	private final LoadingThread loadingThread;
	private final IngestScheduler ingestScheduler;
	private final LodPageCache lodPageCache;
	private final MetricsViewerSettings settings;
	private final Semaphore querySemaphore;

	public MetricsService(
			RunScanner runScanner,
			MetricsRepository metricsRepository,
			LoadingThread loadingThread,
			IngestScheduler ingestScheduler,
			LodPageCache lodPageCache,
			MetricsViewerSettings settings) {
		this.runScanner = runScanner;
		this.metricsRepository = metricsRepository;
		this.loadingThread = loadingThread;
		this.ingestScheduler = ingestScheduler;
		this.lodPageCache = lodPageCache;
		this.settings = settings;
		this.querySemaphore = new Semaphore(settings.getMaxConcurrentQueries(), true);
	}

	@PostConstruct
	private void initialize() {
		log.info("MetricsService initialized. Starting LoadingThread.");
		loadingThread.start();
	}

	@PreDestroy
	private void shutdown() {
		loadingThread.terminateAndWait(SHUTDOWN_WAIT_TIMEOUT_MS);
	}

	public GetRunsResponse getRuns() {
		final List<String> runIds = runScanner.listRunId();
		final Set<String> existingRuns = Set.copyOf(runIds);
		lodPageCache.retainRuns(existingRuns);
		final List<RunInfo> runs = runIds.stream()
				.map(metricsRepository::findRunInfo)
				.toList();
		for (RunInfo run : runs) {
			if (run.getGeneration() != null) {
				lodPageCache.invalidateGeneration(run.getId(), run.getGeneration().toString());
			}
		}
		final GetRunsResponse response = new GetRunsResponse();
		response.setRuns(runs);
		return response;
	}

	public void prioritizeRuns(PrioritizeRunsRequest request) {
		if (request == null
				|| !request.getUnknownFields().isEmpty()
				|| request.getRunIds() == null) {
			throw invalidRequest("Request must contain only a runIds array");
		}
		final Set<String> existing = Set.copyOf(runScanner.listRunId());
		final Set<String> requested = new LinkedHashSet<>();
		for (String runId : request.getRunIds()) {
			if (runId == null || runId.isBlank() || !existing.contains(runId)) {
				throw invalidRequest("runIds must contain only existing non-empty Run ids");
			}
			requested.add(runId);
		}
		ingestScheduler.replacePriority(requested);
	}

	public GetMetricsResponse getMetrics(GetMetricsRequest request) {
		validateRequest(request);

		boolean acquired = false;
		try {
			acquired = querySemaphore.tryAcquire(5L, TimeUnit.SECONDS);
			if (!acquired) throw queryBusy(request, "timeout");
			return new GetMetricsResponse(metricsRepository.query(request.getSeries()));
		} catch (InterruptedException e) {
			Thread.currentThread().interrupt();
			throw queryBusy(request, "interrupted");
		} finally {
			if (acquired) querySemaphore.release();
		}
	}

	private void validateRequest(GetMetricsRequest request) {
		if (request == null || !request.getUnknownFields().isEmpty() || request.getSeries() == null) {
			throw invalidRequest("Request must contain only a series array");
		}
		for (int i = 0; i < request.getSeries().size(); i++) {
			final MetricsSeriesRequest series = request.getSeries().get(i);
			if (series == null || !series.getUnknownFields().isEmpty()) {
				throw invalidRequest("Invalid series shape at index " + i);
			}
			if (series.getRunId() == null || series.getRunId().isBlank()
					|| series.getTagKey() == null || series.getTagKey().isBlank()) {
				throw invalidRequest("runId and tagKey must be non-empty at index " + i);
			}
			if (series.getFromStep() == null || series.getToStep() == null) {
				throw invalidRequest("fromStep and toStep are required at index " + i);
			}
			if (!isSafeInteger(series.getFromStep()) || !isSafeInteger(series.getToStep())) {
				throw invalidRequest("fromStep and toStep must be JavaScript safe integers at index " + i);
			}
			if (series.getFromStep() > series.getToStep()) {
				throw invalidRequest("fromStep must not exceed toStep at index " + i);
			}
			final int maxPoints = series.getMaxPoints() == null
					? settings.getTargetPointsPerSeries()
					: series.getMaxPoints();
			if (maxPoints < 3 || maxPoints > settings.getMaxPointsPerRequest()) {
				throw invalidRequest("maxPoints must be 3.."
						+ settings.getMaxPointsPerRequest() + " at index " + i);
			}
		}
	}

	private static boolean isSafeInteger(long value) {
		return -9_007_199_254_740_991L <= value && value <= 9_007_199_254_740_991L;
	}

	private static MetricsApiException invalidRequest(String message) {
		return new MetricsApiException(
				HttpStatus.BAD_REQUEST,
				Map.of("code", "invalid_request", "message", message));
	}

	private MetricsApiException queryBusy(GetMetricsRequest request, String reason) {
		log.warn(
				"Metrics query rejected: code=query_busy reason={} series={} active={} queued={}",
				reason,
				request.getSeries().size(),
				settings.getMaxConcurrentQueries() - querySemaphore.availablePermits(),
				querySemaphore.getQueueLength());
		final HttpHeaders headers = new HttpHeaders();
		headers.set("Retry-After", "2");
		return new MetricsApiException(
				HttpStatus.SERVICE_UNAVAILABLE,
				Map.of("code", "query_busy", "message", "Metrics query capacity is busy"),
				headers);
	}
}
