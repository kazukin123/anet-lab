package io.github.kazukin123.anetlab.metricsviewer.service;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import io.github.kazukin123.anetlab.metricsviewer.infra.RunScanner;
import io.github.kazukin123.anetlab.metricsviewer.view.model.GetMetricsRequest;
import io.github.kazukin123.anetlab.metricsviewer.view.model.GetMetricsResponse;
import io.github.kazukin123.anetlab.metricsviewer.view.model.GetRunsResponse;
import io.github.kazukin123.anetlab.metricsviewer.view.model.RunInfo;
import io.github.kazukin123.anetlab.metricsviewer.view.model.RunStats;
import io.github.kazukin123.anetlab.metricsviewer.view.model.TagInfo;
import io.github.kazukin123.anetlab.metricsviewer.view.model.TagTrace;

/**
 * Main service layer that connects Controller with Repository & LoadingThread.
 */
@Service
public class MetricsService {

	private static final Logger log = LoggerFactory.getLogger(MetricsService.class);

	private final RunScanner runScanner;
	private final MetricsRepository metricsRepository;
	private final LoadingThread loadingThread;

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
		final List<String> runIds = runScanner.listRunId();

		// Run情報を生成
		final List<RunInfo> runs = new ArrayList<>();
		for (String runId : runIds) {
			final RunStats stats = metricsRepository.getRunStats(runId);
			final List<TagInfo> tags = metricsRepository.findTagInfo(runId);
			final RunInfo runInfo = RunInfo.builder().id(runId).stats(stats).tags(tags).build();
			runs.add(runInfo);
		}

		final GetRunsResponse resp = new GetRunsResponse();
		resp.setRuns(runs);

		log.debug("getRuns: runs.size={}", runs.size());
		return resp;
	}

    public GetMetricsResponse getMetrics(GetMetricsRequest request) {
        final GetMetricsResponse response = new GetMetricsResponse();

        try {
            // --- 差分ロード（runTagMapあり） ---
            if (request != null && request.getRunTagMap() != null && !request.getRunTagMap().isEmpty()) {
                log.debug("getMetrics: diff mode start. runs={}", request.getRunTagMap().keySet());

                final List<TagTrace> traces = metricsRepository.findTagTraceDiff(request.getRunTagMap());
                response.setData(traces);

                log.debug("getMetrics: diff mode complete. traces={}", traces.size());
                return response;
            }

            // --- フルロード（全Run・全Tag・全Step） ---
            log.debug("getMetrics: full mode start (no runTagMap)");

            // すべてのRunIDを取得
            final List<String> allRunIds = metricsRepository.listAllRunIds();
            final List<TagTrace> allTraces = new ArrayList<>();

            // 各Runについて全タグを取得して全件ロード
            for (String runId : allRunIds) {
                final List<String> tagKeys = metricsRepository.listTagKeys(runId);
                if (tagKeys == null || tagKeys.isEmpty()) continue;

                final Map<String, Integer> tagMap = new LinkedHashMap<>();
                for (String tag : tagKeys) tagMap.put(tag, 0); // fromStep=0で全件

                final Map<String, Map<String, Integer>> runTagMap = Map.of(runId, tagMap);
                final List<TagTrace> traces = metricsRepository.findTagTraceDiff(runTagMap);

                allTraces.addAll(traces);
            }

            response.setData(allTraces);
            log.debug("getMetrics: full mode complete. runs={} traces={}",
                      allRunIds.size(), allTraces.size());

        } catch (Exception ex) {
            log.error("getMetrics: failed to process request", ex);
        }

        return response;
    }

}
