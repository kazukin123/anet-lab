package io.github.kazukin123.anetlab.metricsviewer.view.model;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import com.fasterxml.jackson.annotation.JsonAnySetter;

import lombok.Data;

@Data
public class GetMetricsRequest {
	private List<MetricsSeriesRequest> series;
	private final Map<String, Object> unknownFields = new LinkedHashMap<>();

	@JsonAnySetter
	public void captureUnknownField(String name, Object value) {
		unknownFields.put(name, value);
	}
}
