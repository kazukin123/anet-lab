package io.github.kazukin123.anetlab.metricsviewer.model;

import java.util.List;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;

@Data
@Builder
@AllArgsConstructor
public class MetricTrace {
    private String runId;
    private String tagKey;
    private String type;
    private List<Integer> steps;
    private List<Double> values;
}
