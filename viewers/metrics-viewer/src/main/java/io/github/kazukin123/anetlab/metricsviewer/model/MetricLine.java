package io.github.kazukin123.anetlab.metricsviewer.model;

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
	private final long step;
	private final double value;

	public MetricLine() {
		type = null;
		tag = null;
		step = 0L;
		value = 0.0;
	}
}