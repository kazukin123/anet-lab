package io.github.kazukin123.anetlab.metricsviewer.infra;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import io.github.kazukin123.anetlab.metricsviewer.model.MetricEntry;
import io.github.kazukin123.anetlab.metricsviewer.model.MetricsSnapshot;
import io.github.kazukin123.anetlab.metricsviewer.model.RunInfo;

/**
 * JSONL reader supporting full and differential parsing using file offset.
 * Thread-safe when used with MetricsSnapshot.merge().
 */
@Component
public class MetricsFileReader {

    private static final Logger log = LoggerFactory.getLogger(MetricsFileReader.class);
    private final ObjectMapper mapper = new ObjectMapper();

    /** 行数ベース進行ログ間隔 */
    private static final int PROGRESS_INTERVAL = 100000;
    /** 時間ベース進行ログ間隔(ms) */
    private static final long PROGRESS_INTERVAL_MS = 2000;
    /** mergeバッチサイズ */
    private static final int BATCH_SIZE = 1000;

    /**
     * Perform full parse of metrics.jsonl from start to end.
     */
    public MetricsSnapshot parseFull(RunInfo run) throws IOException {
        log.info("Parsing full JSONL file for run {}", run.getRunId());
        MetricsSnapshot snap = new MetricsSnapshot();
        long offset = 0L;
        int count = 0;
        long t0 = System.nanoTime();
        long lastProgressCount = 0;
        long lastLogTime = System.currentTimeMillis();
        List<MetricEntry> buffer = new ArrayList<>(BATCH_SIZE);

        try (RandomAccessFile raf = new RandomAccessFile(run.getJsonlPath().toFile(), "r")) {
            String line;
            while ((line = raf.readLine()) != null) {
                MetricEntry entry = parseLineSafe(line);
                if (entry != null) {
                    buffer.add(entry);
                    count++;
                }

                if (buffer.size() >= BATCH_SIZE) {
                    snap.merge(buffer);
                    buffer.clear();
                }

                long now = System.currentTimeMillis();
                if ((count - lastProgressCount >= PROGRESS_INTERVAL) ||
                    (now - lastLogTime >= PROGRESS_INTERVAL_MS)) {
                    log.info("Parsing run {}... {} lines processed", run.getRunId(), count);
                    lastProgressCount = count;
                    lastLogTime = now;
                }
            }

            if (!buffer.isEmpty()) {
                snap.merge(buffer);
                buffer.clear();
            }

            offset = raf.getFilePointer();
        }

        // 最終進行ログ
        if (count > lastProgressCount) {
            log.info("parseFull() Parsing run {}... {} lines processed (final checkpoint)", run.getRunId(), count);
        }

        snap.setLastReadPosition(offset);
        run.setLastReadPosition(offset);

        long elapsedMs = (System.nanoTime() - t0) / 1_000_000;
        log.info("Full parse complete: {} entries read for run {} ({} ms)",
                count, run.getRunId(), elapsedMs);
        return snap;
    }

    /**
     * Reads newly appended metric entries based on the last position in the snapshot.
     * Synchronizes the final offset back to the snapshot after reading.
     */
    public int readNewEntries(RunInfo run, MetricsSnapshot snapshot) throws IOException {
        Path jsonl = Path.of(run.getRunPath(), "metrics.jsonl");
        if (!Files.exists(jsonl)) {
            log.warn("metrics.jsonl not found for {}", run.getRunId());
            return 0;
        }

        long fileSize = Files.size(jsonl);
        long offset = snapshot.getLastReadPosition(); // ★ snapshot起点に変更

        log.info("Starting diff read for run {} (offset {} / fileSize {})",
                run.getRunId(), offset, fileSize);

        if (fileSize <= offset) {
            log.debug("No new data for run {} (fileSize={} <= offset={})",
                    run.getRunId(), fileSize, offset);
            return 0;
        }

        // privateメソッドで差分読み込み
        List<MetricEntry> entries = parseFromOffset(run, jsonl, offset);
        if (entries.isEmpty()) {
            log.debug("No new entries found for {}", run.getRunId());
            return 0;
        }

        // merge entries into snapshot
        snapshot.merge(entries);

        // ★ RunInfoは一時カーソルとして使用、終了位置をSnapshotに反映
        snapshot.setLastReadPosition(run.getLastReadPosition());

        log.info("readNewEntries() completed for {} (new offset={})",
                run.getRunId(), snapshot.getLastReadPosition());

        return entries.size();
    }

    /**
     * Parse metrics.jsonl from specific offset (for differential update).
     */
    private List<MetricEntry> parseFromOffset(RunInfo run, Path jsonl, long offset) throws IOException {
        List<MetricEntry> list = new ArrayList<>();
        long t0 = System.nanoTime();
        int count = 0;
        long lastProgressCount = 0;
        long lastLogTime = System.currentTimeMillis();
        List<MetricEntry> buffer = new ArrayList<>(BATCH_SIZE);

        try (RandomAccessFile raf = new RandomAccessFile(jsonl.toFile(), "r")) {
            raf.seek(offset);
            String line;
            while ((line = raf.readLine()) != null) {
                MetricEntry entry = parseLineSafe(line);
                if (entry != null) {
                    buffer.add(entry);
                    count++;
                }

                if (buffer.size() >= BATCH_SIZE) {
                    list.addAll(buffer);
                    buffer.clear();
                }

                long now = System.currentTimeMillis();
                if ((count - lastProgressCount >= PROGRESS_INTERVAL) ||
                    (now - lastLogTime >= PROGRESS_INTERVAL_MS)) {
                    log.info("parseFromOffset() Reading diff for {}. {} lines processed (offset {}→{})", run.getRunId(), count, offset, raf.getFilePointer());
                    lastProgressCount = count;
                    lastLogTime = now;
                }
            }

            if (!buffer.isEmpty()) {
                list.addAll(buffer);
                buffer.clear();
            }
            run.setLastReadPosition(raf.getFilePointer());
        }

        if (count > 0) {
            long elapsed = (System.nanoTime() - t0) / 1_000_000;
            log.info("Read {} new entries for run {} (offset {}→{}, {} ms)",
                    count, run.getRunId(), offset, run.getLastReadPosition(), elapsed);
        } else {
            log.debug("No new entries found for run {}", run.getRunId());
        }

        return list;
    }

    /**
     * Safely parse a JSONL line. Returns null if malformed.
     */
    private MetricEntry parseLineSafe(String raw) {
        try {
            String line = new String(raw.getBytes(StandardCharsets.ISO_8859_1), StandardCharsets.UTF_8);
            JsonNode n = mapper.readTree(line);

            String type = n.has("type") ? n.get("type").asText() : "scalar";
            if (!"scalar".equals(type)) return null;

            String tag = n.has("tag") ? n.get("tag").asText() : null;
            if (tag == null || tag.isEmpty()) return null;

            long step = n.has("step") ? n.get("step").asLong() : -1L;
            if (step < 0) return null;

            double value = n.has("value") ? n.get("value").asDouble() : Double.NaN;
            long ts = n.has("timestamp") ? n.get("timestamp").asLong() : System.currentTimeMillis();

            return new MetricEntry(tag, step, value, ts);
        } catch (Exception e) {
            log.debug("Skipping invalid JSONL line: {}", e.getMessage());
            return null;
        }
    }
}
