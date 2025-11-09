package io.github.kazukin123.anetlab.metricsviewer.infra;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.RandomAccessFile;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;

import io.github.kazukin123.anetlab.metricsviewer.infra.model.MetricFileBlock;
import io.github.kazukin123.anetlab.metricsviewer.infra.model.MetricLine;

@Component
public class MetricsFileReader {

    private static final Logger log = LoggerFactory.getLogger(MetricsFileReader.class);
    private static final int BUFFER_SIZE = 1 << 16; // 64KB
    private static final int PROGRESS_INTERVAL = 100_000;
    private static final long PROGRESS_INTERVAL_MS = 2000;
    private static final ObjectMapper METRIC_LINE_READER = new ObjectMapper();

    /**
     * ファイル全体をパースする。
     */
    public MetricFileBlock parseFull(Path jsonlFile, int maxLines) throws IOException {
        return readInternal(jsonlFile, 0, maxLines);
    }

    /**
     * 指定オフセットからパースする（増分読み込み）。
     */
    public MetricFileBlock parseDiff(Path jsonlFile, long startOffset, int maxLines) throws IOException {
        return readInternal(jsonlFile, startOffset, maxLines);
    }

    /**
     * 実際の読み込み処理。
     */
    private MetricFileBlock readInternal(Path jsonlFile, long startOffset, int maxLines) throws IOException {
        if (!Files.exists(jsonlFile)) {
            log.warn("Metrics file not found: {}", jsonlFile);
            return new MetricFileBlock(0, 0, List.of(), 0L, true);
        }

        // --- 行単位にオフセットを補正する ---
        if (startOffset > 0) {
            try (RandomAccessFile raf = new RandomAccessFile(jsonlFile.toFile(), "r")) {
                raf.seek(startOffset);
                int b;
                boolean skipped = false;
                while ((b = raf.read()) != -1) {
                    if (b == '\n') {
                        skipped = true;
                        break;
                    }
                }
                if (skipped) {
                    long newPos = raf.getFilePointer();
                    if (newPos > startOffset) {
                        log.debug("Offset adjusted from {} → {}", startOffset, newPos);
                        startOffset = newPos;
                    }
                }
            }
        }

        long fileLastModified = Files.getLastModifiedTime(jsonlFile).toMillis();
        List<MetricLine> lines = new ArrayList<>(maxLines > 0 ? maxLines : 4096);

        long bytesRead = startOffset;
        long lastLogTime = System.currentTimeMillis();
        long tStart = System.currentTimeMillis();

        log.debug("parse start file={} offset={} maxLines={}", jsonlFile, startOffset, maxLines);

        try (InputStream in = Files.newInputStream(jsonlFile);
             BufferedReader br = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8), BUFFER_SIZE)) {

            // 指定位置までスキップ（RandomAccessFileで補正済み）
            long skipped = in.skip(startOffset);
            if (skipped < startOffset) {
                log.warn("Skipped {} < requested {} (file may have truncated)", skipped, startOffset);
            }

            String line;
            int lineCount = 0;

            while ((line = br.readLine()) != null) {
                bytesRead += line.getBytes(StandardCharsets.UTF_8).length + 1;
                lineCount++;

                try {
                    MetricLine obj = METRIC_LINE_READER.readValue(line, MetricLine.class);
                    lines.add(obj);
                } catch (JsonProcessingException e) {
                    boolean looksTruncated =
                            line.length() < 10 ||
                            !line.trim().startsWith("{") ||
                            !line.trim().endsWith("}");

                    if (looksTruncated) {
                        // ファイル途中 or 書きかけJSONは想定内
                        if (log.isDebugEnabled()) {
                            log.debug("Skip incomplete JSON fragment near offset {} (len={}): {}",
                                    bytesRead, line.length(),
                                    line.substring(0, Math.min(80, line.length())));
                        }
                        continue;
                    }

                    // 構造が完結しているのに壊れている → エラー扱い
                    String excerpt = line.length() > 160 ? line.substring(0, 160) + "..." : line;
                    log.error("JSON parse error (corrupted?) near offset {}: {} line={}",
                            bytesRead, e.getOriginalMessage(), excerpt);
                }

                if (lineCount % PROGRESS_INTERVAL == 0 ||
                        System.currentTimeMillis() - lastLogTime >= PROGRESS_INTERVAL_MS) {
                    log.debug("reading {} lines={} offset={}MB", jsonlFile.getFileName(),
                            lineCount, bytesRead / 1_000_000);
                    lastLogTime = System.currentTimeMillis();
                }

                if (maxLines > 0 && lineCount >= maxLines) break;
            }

            boolean eof = (br.readLine() == null);
            long tEnd = System.currentTimeMillis();

            log.debug("parse done file={} lines={} readBytes={}MB duration={}ms eof={}",
                    jsonlFile.getFileName(),
                    lineCount, bytesRead / 1_000_000,
                    (tEnd - tStart), eof);

            return new MetricFileBlock((int) startOffset, (int) bytesRead, lines, fileLastModified, eof);

        } catch (IOException e) {
            log.error("Error reading metrics file {}: {}", jsonlFile, e.getMessage());
            throw e;
        }
    }
}
