package io.github.kazukin123.anetlab.metricsviewer.model;

import java.io.Serializable;

import lombok.AllArgsConstructor;
import lombok.Data;

/**
 * Single data point in a time series.
 */
@Data
@AllArgsConstructor
public class MetricPoint implements Serializable {
    private static final long serialVersionUID = 1L;
    private long step;
    private double value;
    private long timestamp;
}
