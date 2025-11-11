package io.github.kazukin123.anetlab.metricsviewer.infra.model;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;

@Data
@Builder
@AllArgsConstructor
@JsonIgnoreProperties(ignoreUnknown = true)
public class MetricLine {
	private final String type;
	private final String tag;
	private final int step;
	private final float value;

	public MetricLine() {
		type = null;
		tag = null;
		step = 0;
		value = 0.0f;
	}
}