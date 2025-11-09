package io.github.kazukin123.anetlab.metricsviewer.service;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicReference;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import io.github.kazukin123.anetlab.metricsviewer.io.MetricsFileReader;
import io.github.kazukin123.anetlab.metricsviewer.model.MetricFileBlock;
import io.github.kazukin123.anetlab.metricsviewer.model.MetricsSnapshot;
import io.github.kazukin123.anetlab.metricsviewer.model.Run;

/**
 * Background thread that periodically scans runs directory
 * and updates metrics cache.
 */
@Component
public class LoadingThread extends Thread {

    private static final int SLEEP_MS = 100;
    private static final int MAX_LINES = 5000;
    private static final int SAVE_INTERVAL_BLOCKS = 20;

    private static final Logger log = LoggerFactory.getLogger(LoadingThread.class);

    private final RunRepository runRepository;
    private final MetricsRepository metricsRepository;
    private final MetricsFileReader fileReader;

    private volatile boolean running = true;
    private final AtomicReference<Request> requestRef = new AtomicReference<>();
    private final Map<String, Integer> saveCounter = new ConcurrentHashMap<>();

    public LoadingThread(RunRepository runRepository,
                         MetricsRepository metricsRepository,
                         MetricsFileReader fileReader) {
        this.runRepository = runRepository;
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

    @Override
    public void run() {
    	// 最初に読めるだけ全部のキャッシュを読む
        log.info("LoadingThread started. Loading cache.");
        metricsRepository.loadCacheAll(runRepository.getRunsDir());
        log.info("Cache loading completed.");

        // スレッドメインループ
        while (running && !isInterrupted()) {
        	try {
        		// 新しいRunが見つかってるかもなのでキャッシュ読込みを試みる
        		Path runsDir = this.runRepository.getRunsDir();
                metricsRepository.loadCacheForRun(runsDir);

                // 優先リクエストがあれば先に処理
                Request req = requestRef.getAndSet(null);
                if (req != null) {
                    log.debug("Processing request: runs={} tags={}", req.runIds, req.tagKeys);
                    processRuns(req.runIds);
                } else {
                    // 定期スキャン
                    List<Run> runs = runRepository.getRuns();
                    List<String> runIds = new ArrayList<>();
                    for (Run r : runs) runIds.add(r.getId());
                    processRuns(runIds);
                }
                // リラックス
                Thread.sleep(SLEEP_MS);
            } catch (InterruptedException e) {
                break;
            } catch (Exception e) {
                log.warn("LoadingThread error: {}", e.getMessage());
            }
        }	
        log.info("LoadingThread stopped.");
    }

    /** Process each run sequentially and merge new metrics. */
    private void processRuns(List<String> runIds) {
        if (runIds == null || runIds.isEmpty()) return;

        for (String runId : runIds) {
            try {
            	// 対象Runのディレクトリ・ファイルを決定
                Path runDir = runRepository.resolveRunDir(runId);
                Path metricsFile = Path.of("runs", runId, "metrics.jsonl");
                if (!Files.exists(metricsFile)) continue;

                // 最後の位置からブロック読み込み
                long lastPos = Optional.ofNullable(metricsRepository.getSnapshots().get(runId))
                        .map(MetricsSnapshot::getLastReadPosition)
                        .orElse(0L);
                MetricFileBlock block = fileReader.parseDiff(metricsFile, lastPos, MAX_LINES);
                if (block.getLines().isEmpty()) continue;

                // メモリ上でマージ
                metricsRepository.mergeMetrics(runId, block);
                
                // 未セーブが一定量溜まったらファイル書き出し
                int dirtyCount = saveCounter.merge(runId, 1, Integer::sum);
                if (dirtyCount >= SAVE_INTERVAL_BLOCKS || block.isEOF()) {
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
}
