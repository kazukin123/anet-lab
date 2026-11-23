package io.github.kazukin123.anetlab.metricsviewer.view.model;

import java.util.List;

import com.fasterxml.jackson.annotation.JsonInclude;

public record RunConfigResponse(List<RunConfig> runs) {

	@JsonInclude(JsonInclude.Include.NON_NULL)
	public record RunConfig(String runId, String status, String message, List<Entry> entries) {
	}

	public record Entry(String key, String value) {
	}
}
