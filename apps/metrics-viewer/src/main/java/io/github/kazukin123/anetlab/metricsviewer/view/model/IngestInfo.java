package io.github.kazukin123.anetlab.metricsviewer.view.model;

import com.fasterxml.jackson.annotation.JsonInclude;

@JsonInclude(JsonInclude.Include.NON_NULL)
public record IngestInfo(String state, int percentage, ApiError error) {
}
