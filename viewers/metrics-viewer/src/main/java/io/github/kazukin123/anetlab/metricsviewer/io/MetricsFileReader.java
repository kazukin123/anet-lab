package io.github.kazukin123.anetlab.metricsviewer.io;

import java.io.ByteArrayOutputStream;
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

import com.fasterxml.jackson.databind.ObjectMapper;

import io.github.kazukin123.anetlab.metricsviewer.model.MetricFileBlock;
import io.github.kazukin123.anetlab.metricsviewer.model.MetricLine;

/**
 * Reads metrics.jsonl files and returns MetricFileBlock.
 * Supports full or incremental (diff) parsing via file byte offset.
 */
@Component
public class MetricsFileReader {

    private static final Logger log = LoggerFactory.getLogger(MetricsFileReader.class);
    private static final int MAX_LINE_LENGTH = 1024 * 32; // 32KB safeguard
    private final ObjectMapper mapper = new ObjectMapper();

    /**
     * Reads the entire file from beginning.
     */
    public MetricFileBlock parseFull(Path jsonlFile, int maxLines) throws IOException {
        return parseInternal(jsonlFile, 0L, maxLines);
    }

    /**
     * Reads from given byte offset until maxLines or EOF.
     */
    public MetricFileBlock parseDiff(Path jsonlFile, long startOffset, int maxLines) throws IOException {
        return parseInternal(jsonlFile, startOffset, maxLines);
    }

    private MetricFileBlock parseInternal(Path jsonlFile, long startOffset, int maxLines) throws IOException {
        List<MetricLine> lines = new ArrayList<>();
        String runId = jsonlFile.getParent().getFileName().toString();
        long fileLength = Files.size(jsonlFile);
        long lastModified = Files.getLastModifiedTime(jsonlFile).toMillis();

        try (RandomAccessFile raf = new RandomAccessFile(jsonlFile.toFile(), "r")) {
            raf.seek(startOffset);
            String line;
            long lastGoodOffset = startOffset;
            int count = 0;

            while ((line = safeReadLine(raf)) != null) {
                long currentPos = raf.getFilePointer();
                if (line.isEmpty()) {
                    lastGoodOffset = currentPos;
                    continue;
                }

                try {
                    MetricLine ml = mapper.readValue(line, MetricLine.class);
                    lines.add(ml);
                    count++;
                    lastGoodOffset = currentPos;
                } catch (Exception e) {
                    // likely incomplete JSON at EOF → stop here safely
                    log.warn("JSON parse error near offset {}: {}", lastGoodOffset, e.getMessage());
                    break;
                }

                if (maxLines > 0 && count >= maxLines) break;
            }

            boolean eof = (lastGoodOffset >= fileLength);
            return MetricFileBlock.builder()
                    .runId(runId)
                    .startOffset((int) startOffset)
                    .endOffset((int) lastGoodOffset)
                    .lines(lines)
                    .lastModified(lastModified)
                    .isEOF(eof)
                    .build();
        } catch (IOException e) {
            throw new MetricsParseException("Failed to read file: " + jsonlFile, startOffset, null, e);
        }
    }

    /**
     * Reads a UTF-8 line from RandomAccessFile.
     * Returns null on EOF.
     */
    private String safeReadLine(RandomAccessFile raf) throws IOException {
        ByteArrayOutputStream buffer = new ByteArrayOutputStream();
        int b;
        int count = 0;
        while ((b = raf.read()) != -1) {
            if (b == '\n') break;
            if (b != '\r') buffer.write(b);
            count++;
            if (count > MAX_LINE_LENGTH) {
                throw new IOException("Line exceeds max length at offset " + (raf.getFilePointer() - count));
            }
        }
        if (b == -1 && buffer.size() == 0) return null;
        return buffer.toString(StandardCharsets.UTF_8);
    }
}
