package io.github.kazukin123.anetlab.metricsviewer.view.model;

public record RangeStatsRequest(String runId, String tagKey, long fromStep, long toStep) {
}
