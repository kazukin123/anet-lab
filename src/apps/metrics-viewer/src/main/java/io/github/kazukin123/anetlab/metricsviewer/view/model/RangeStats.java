package io.github.kazukin123.anetlab.metricsviewer.view.model;

import java.util.UUID;

import com.fasterxml.jackson.annotation.JsonUnwrapped;

public record RangeStats(String runId, String tagKey, long fromStep, long toStep,
		UUID generation, String availability, @JsonUnwrapped TagStats stats) {
}
