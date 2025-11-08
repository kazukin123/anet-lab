package io.github.kazukin123.anetlab.metricsviewer.io;

import java.io.IOException;
import java.nio.file.Path;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import io.github.kazukin123.anetlab.metricsviewer.model.MetricFileBlock;

/**
 * JSONL reader supporting full and differential parsing using file offset.
 * Thread-safe when used with MetricsSnapshot.merge().
 */
@Component
public class MetricsFileReader {

    private static final Logger log = LoggerFactory.getLogger(MetricsFileReader.class);

    /** 行数ベース進行ログ間隔 */
    private static final int PROGRESS_INTERVAL = 100000;
    /** 時間ベース進行ログ間隔(ms) */
    private static final long PROGRESS_INTERVAL_MS = 2000;
    /** mergeバッチサイズ */
    private static final int BATCH_SIZE = 1000;

    public MetricFileBlock parseFull(Path jsonlFile, int maxLines) throws IOException {
    	return null;
    }

    public MetricFileBlock parseDiff(Path jsonlFile, int startOffset, int maxLines) throws IOException {
    	return null;
    }

}
