package io.github.kazukin123.anetlab.metricsviewer.model;

import lombok.AllArgsConstructor;
import lombok.Data;

/**
 * Parsed single JSONL entry. Converts to MetricPoint during merge.
 */
@Data
@AllArgsConstructor
public class MetricEntry {
    private String tag;
    private long step;
    private double value;
    private long timestamp;
}
