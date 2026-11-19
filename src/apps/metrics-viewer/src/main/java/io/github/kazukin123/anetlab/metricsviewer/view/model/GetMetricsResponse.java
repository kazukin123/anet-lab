package io.github.kazukin123.anetlab.metricsviewer.view.model;

import java.util.List;

public record GetMetricsResponse(List<MetricsSeriesResult> data) {
}
