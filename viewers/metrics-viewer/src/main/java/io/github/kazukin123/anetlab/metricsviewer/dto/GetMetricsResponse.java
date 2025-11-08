package io.github.kazukin123.anetlab.metricsviewer.dto;

import java.util.List;

import io.github.kazukin123.anetlab.metricsviewer.model.MetricTrace;
import lombok.Data;

@Data
public class GetMetricsResponse {
    private List<MetricTrace> data;
}
