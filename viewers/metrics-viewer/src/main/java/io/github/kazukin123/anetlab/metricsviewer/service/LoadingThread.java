package io.github.kazukin123.anetlab.metricsviewer.service;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicReference;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsFileReader;
import io.github.kazukin123.anetlab.metricsviewer.infra.RunScanner;
import io.github.kazukin123.anetlab.metricsviewer.infra.model.MetricsFileBlock;

/**
 * Background thread that periodically scans runs directory and updates metrics cache.
 */
@Component
public class LoadingThread extends Thread {

	private static final int SLEEP_MS = 10000;
	private static final int MAX_LINES = 1000000;
	private static final int SAVE_INTERVAL_BLOCKS = 100;

	private static final Logger log = LoggerFactory.getLogger(LoadingThread.class);

	private final RunScanner runScanner;
	private final MetricsRepository metricsRepository;
	private final MetricsFileReader fileReader;

	private volatile boolean running = true;
	private final AtomicReference<Request> requestRef = new AtomicReference<>();
	private final Map<String, Integer> saveCounter = new ConcurrentHashMap<>();

	public LoadingThread(RunScanner runScanner,
			MetricsRepository metricsRepository,
			MetricsFileReader fileReader) {
		this.runScanner = runScanner;
		this.metricsRepository = metricsRepository;
		this.fileReader = fileReader;
		setName("Metrics-LoadingThread");
		setDaemon(true);
	}

	/** Request structure for on-demand loading. */
	private static class Request {
		List<String> runIds;
		List<String> tagKeys;
		int maxMsec;

		Request(List<String> runIds, List<String> tagKeys, int maxMsec) {
			this.runIds = runIds;
			this.tagKeys = tagKeys;
			this.maxMsec = maxMsec;
		}
	}

	/** Called by service layer to request prioritized load. */
	public void request(List<String> runIds, List<String> tagKeys, int maxMsec) {
		requestRef.set(new Request(runIds, tagKeys, maxMsec));
	}

	/** Graceful termination signal. */
	public void terminate() {
		running = false;
		this.interrupt();
	}

	public void terminateAndWait(long timeoutMs) {
		terminate();
		if (Thread.currentThread() == this) {
			return;
		}
		try {
			join(timeoutMs);
			if (isAlive()) {
				log.warn("LoadingThread did not stop within {}ms.", timeoutMs);
			}
		} catch (InterruptedException e) {
			Thread.currentThread().interrupt();
			log.warn("Interrupted while waiting for LoadingThread to stop.");
		}
	}

	@Override
	public void run() {
		try {
			log.info("LoadingThread started. Loading cache.");
			metricsRepository.loadCache(runScanner.getRunsDir());
			log.info("Cache loading completed.");

			while (running && !isInterrupted()) {
				try {
					final Path runsDir = this.runScanner.getRunsDir();
					metricsRepository.loadCacheForRun(runsDir);

					final Request req = requestRef.getAndSet(null);
					if (req != null) {
						log.debug("Processing request: runs={} tags={}", req.runIds, req.tagKeys);
						processRuns(req.runIds);
					} else {
						final List<String> runIds = runScanner.listRunId();
						processRuns(runIds);
					}
					Thread.sleep(SLEEP_MS);
				} catch (InterruptedException e) {
					break;
				} catch (Exception e) {
					log.warn("LoadingThread error: {}", e.getMessage());
				}
			}
		} finally {
			saveAllLoadedCaches();
			log.info("LoadingThread stopped.");
		}
	}

	/** Process each run sequentially and merge new metrics. */
	private void processRuns(List<String> runIds) {
		if (runIds == null || runIds.isEmpty()) return;

		for (String runId : runIds) {
			if (!running || isInterrupted()) return;
			try {
				// 対象Runのディレクトリ・ファイルを決定
				final Path runDir = runScanner.resolveRunDir(runId);
				final Path metricsFile = runDir.resolve("metrics.jsonl");
				if (!Files.exists(metricsFile)) continue;

				// 最後の位置からブロック読み込み
				final long lastPos = metricsRepository.getLastReadPosition(runId);
				final MetricsFileBlock block = fileReader.parseDiff(metricsFile, lastPos, MAX_LINES);
				if (block.getLines().isEmpty()) continue;

				// メモリ上でマージ
				metricsRepository.mergeMetrics(runId, block);

				// 未セーブが一定量溜まったらファイル書き出し
				final int dirtyCount = saveCounter.merge(runId, 1, Integer::sum);
				if (dirtyCount >= SAVE_INTERVAL_BLOCKS) {
					log.info("Saving cache. runId={} dirtyCount={}", runId, dirtyCount);
					metricsRepository.saveCache(runDir, runId);
					saveCounter.put(runId, 0);
				}

				log.debug("Loaded {} lines for run={} endOffset={} eof={}",
						block.getLines().size(), runId, block.getEndOffset(), block.isEOF());
			} catch (Exception e) {
				log.warn("Failed to load metrics for run {}: {}", runId, e.getMessage());
			}
		}
	}

	private void saveAllLoadedCaches() {
		final List<String> runIds = metricsRepository.listAllRunIds();
		if (runIds.isEmpty()) {
			log.info("No loaded cache to save.");
			return;
		}

		log.info("Saving all loaded caches. runs={}", runIds.size());
		for (String runId : runIds) {
			try {
				metricsRepository.saveCache(runScanner.resolveRunDir(runId), runId);
				saveCounter.put(runId, 0);
			} catch (Exception e) {
				log.warn("Failed to save cache on shutdown. runId={} message={}", runId, e.getMessage());
			}
		}
	}
}
