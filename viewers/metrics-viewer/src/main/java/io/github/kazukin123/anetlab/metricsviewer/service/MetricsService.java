package io.github.kazukin123.anetlab.metricsviewer.service;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

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

	/**
	 * Returns metric traces (used by /api/metrics).
	 */
	public GetMetricsResponse getMetrics(GetMetricsRequest req) {
		if (req == null) return new GetMetricsResponse();

		List<String> runIds = req.getRunIds();
		List<String> tagKeys = req.getTagKeys();

		// --- ① 空指定時は全件扱いに補完 ---
		if (runIds == null || runIds.isEmpty()) {
			runIds = runScanner.listRunId();
		}

		if (tagKeys == null || tagKeys.isEmpty()) {
			// 全RunのTagを走査して一意にまとめる
			final Set<String> allTags = new LinkedHashSet<>();
			for (String runId : runIds) {
				final List<TagInfo> tags = metricsRepository.findTagInfo(runId);
				for (TagInfo tag : tags) {
					allTags.add(tag.getKey());
				}
			}
			tagKeys = new ArrayList<>(allTags);
		}

		// --- ② ロードリクエスト発行（非同期） ---
		loadingThread.request(runIds, tagKeys, 1000);

		// --- ③ 現在のキャッシュを返す ---
		final List<TagTrace> tagTraceList = metricsRepository.findTagTrace(runIds, tagKeys);

		final GetMetricsResponse res = new GetMetricsResponse();
		res.setData(tagTraceList);

		log.debug("getMetrics: runs={} tags={} traces={}", runIds, tagKeys, tagTraceList.size());
		return res;
	}
}
