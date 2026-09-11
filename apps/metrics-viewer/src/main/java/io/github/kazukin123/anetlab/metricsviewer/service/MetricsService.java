package io.github.kazukin123.anetlab.metricsviewer.service;

import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;

import io.github.kazukin123.anetlab.metricsviewer.config.MetricsViewerSettings;
import io.github.kazukin123.anetlab.metricsviewer.view.model.GetMetricsRequest;
import io.github.kazukin123.anetlab.metricsviewer.view.model.GetMetricsResponse;
import io.github.kazukin123.anetlab.metricsviewer.view.model.GetRunsResponse;
import io.github.kazukin123.anetlab.metricsviewer.view.model.GetWorkspacesResponse;
import io.github.kazukin123.anetlab.metricsviewer.view.model.RunInfo;
import io.github.kazukin123.anetlab.metricsviewer.view.model.MetricsSeriesRequest;
import io.github.kazukin123.anetlab.metricsviewer.view.model.PrioritizeRunsRequest;
import io.github.kazukin123.anetlab.metricsviewer.view.model.SwitchWorkspaceRequest;

@Service
public class MetricsService {

	private static final long SHUTDOWN_WAIT_TIMEOUT_MS = 30_000L;
	private static final Logger log = LoggerFactory.getLogger(MetricsService.class);

	private final WorkspaceManager workspaceManager;
	private final LoadingThread loadingThread;
	private final MetricsViewerSettings settings;
	private final MetricsQueryCoordinator queryCoordinator;

	public MetricsService(
			WorkspaceManager workspaceManager,
			LoadingThread loadingThread,
			MetricsViewerSettings settings,
			MetricsQueryCoordinator queryCoordinator) {
		this.workspaceManager = workspaceManager;
		this.loadingThread = loadingThread;
		this.settings = settings;
		this.queryCoordinator = queryCoordinator;
	}

	@PostConstruct
	private void initialize() {
		log.info("MetricsService initialized. Starting LoadingThread.");
		loadingThread.start();
	}

	@PreDestroy
	private void shutdown() {
		queryCoordinator.cancelAll();
		loadingThread.terminateAndWait(SHUTDOWN_WAIT_TIMEOUT_MS);
		workspaceManager.shutdown();
	}

	public GetRunsResponse getRuns() {
		try (WorkspaceManager.Lease lease = workspaceManager.acquireLease()) {
			final List<String> runIds = lease.runScanner().listRunId();
			final Set<String> existingRuns = Set.copyOf(runIds);
			lease.pageCache().retainRuns(existingRuns);
			final List<RunInfo> runs = runIds.stream()
					.map(lease.repository()::findRunInfo)
					.toList();
			for (RunInfo run : runs) {
				if (run.getGeneration() != null) {
					lease.pageCache().invalidateGeneration(
							run.getId(), run.getGeneration().toString());
				}
			}
			final GetRunsResponse response = new GetRunsResponse();
			response.setRuns(runs);
			return response;
		}
	}

	public void prioritizeRuns(PrioritizeRunsRequest request) {
		try (WorkspaceManager.Lease lease = workspaceManager.acquireLease()) {
			if (request == null
					|| !request.getUnknownFields().isEmpty()
					|| request.getRunIds() == null) {
				throw invalidRequest("Request must contain only a runIds array");
			}
			final Set<String> existing = Set.copyOf(lease.runScanner().listRunId());
			final Set<String> requested = new LinkedHashSet<>();
			for (String runId : request.getRunIds()) {
				if (runId == null || runId.isBlank() || !existing.contains(runId)) {
					throw invalidRequest("runIds must contain only existing non-empty Run ids");
				}
				requested.add(runId);
			}
			lease.ingestScheduler().replacePriority(requested);
		}
	}

	public GetMetricsResponse getMetrics(
			GetMetricsRequest request,
			String queryChannelHeader,
			String querySequenceHeader) {
		validateRequest(request);
		final QueryChannel queryChannel = validateQueryChannel(queryChannelHeader);
		final long querySequence = validateQuerySequence(querySequenceHeader);

		try {
			return queryCoordinator.run(queryChannel, querySequence, query -> {
				final WorkspaceManager.Lease lease;
				try {
					lease = workspaceManager.acquireLease();
				} catch (IllegalStateException e) {
					throw new QueryCapacityException("shutdown", 0, 0);
				}
				try (lease) {
					query.bindWorkspace(lease.epoch());
					return new GetMetricsResponse(lease.repository().query(request.getSeries(), query));
				}
			});
		} catch (QueryCancelledException e) {
			throw superseded();
		} catch (QueryCapacityException e) {
			throw queryBusy(request, e);
		}
	}

	public GetWorkspacesResponse getWorkspaces() {
		final GetWorkspacesResponse response = new GetWorkspacesResponse();
		response.setCurrent(workspaceManager.currentName());
		response.setWorkspaces(workspaceManager.listWorkspaceNames());
		return response;
	}

	public void switchWorkspace(SwitchWorkspaceRequest request) {
		if (request == null || !request.getUnknownFields().isEmpty()
				|| request.getName() == null || request.getName().isBlank()) {
			throw invalidRequest("Request must contain only a non-empty name string");
		}
		final WorkspaceManager.SwitchResult result = workspaceManager.switchWorkspace(request.getName());
		if (result == WorkspaceManager.SwitchResult.UNKNOWN) {
			throw new MetricsApiException(
					HttpStatus.NOT_FOUND,
					Map.of("code", "unknown_workspace", "message",
							"Unknown workspace: " + request.getName()));
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

	private static QueryChannel validateQueryChannel(String value) {
		if (value == null || value.isBlank() || value.length() > 128) {
			throw invalidRequest("X-Query-Channel must be a non-blank string of 1..128 characters");
		}
		return new QueryChannel(value);
	}

	private static long validateQuerySequence(String value) {
		if (value == null) {
			throw invalidRequest("X-Query-Sequence must be a JavaScript safe non-negative integer");
		}
		try {
			final long sequence = Long.parseLong(value);
			if (sequence < 0L || !isSafeInteger(sequence)) {
				throw invalidRequest(
						"X-Query-Sequence must be a JavaScript safe non-negative integer");
			}
			return sequence;
		} catch (NumberFormatException e) {
			throw invalidRequest("X-Query-Sequence must be a JavaScript safe non-negative integer");
		}
	}

	private static MetricsApiException invalidRequest(String message) {
		return new MetricsApiException(
				HttpStatus.BAD_REQUEST,
				Map.of("code", "invalid_request", "message", message));
	}

	private static MetricsApiException superseded() {
		return new MetricsApiException(
				HttpStatus.CONFLICT,
				Map.of("code", "superseded", "message", "Metrics query was superseded"));
	}

	private MetricsApiException queryBusy(GetMetricsRequest request, QueryCapacityException capacity) {
		log.warn(
				"Metrics query rejected: code=query_busy reason={} series={} active={} queued={}",
				capacity.reason(),
				request.getSeries().size(),
				capacity.active(),
				capacity.queued());
		final HttpHeaders headers = new HttpHeaders();
		headers.set("Retry-After", "2");
		return new MetricsApiException(
				HttpStatus.SERVICE_UNAVAILABLE,
				Map.of("code", "query_busy", "message", "Metrics query capacity is busy"),
				headers);
	}
}
