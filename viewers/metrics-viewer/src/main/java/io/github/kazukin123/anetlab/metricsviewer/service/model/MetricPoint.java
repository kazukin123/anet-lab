package io.github.kazukin123.anetlab.metricsviewer.service.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;

@Data
@AllArgsConstructor
@Builder
public class MetricPoint {
	private final long step;
	private final double value;
	
	public MetricPoint() {
		step = 0;
		value = 0.0;
	}
}
