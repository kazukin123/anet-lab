package io.github.kazukin123.anetlab.metricsviewer.view.model;

import java.util.List;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;

@Data
@Builder
@AllArgsConstructor
public class MetricTrace {
    private final String runId;
    private final String tagKey;
    private final String type;
    private final List<Integer> steps;
    private final List<Double> values;
}
