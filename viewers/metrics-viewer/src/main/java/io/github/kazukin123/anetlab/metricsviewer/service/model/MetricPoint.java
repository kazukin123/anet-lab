package io.github.kazukin123.anetlab.metricsviewer.service.model;

import java.io.Serializable;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;

@Data
@AllArgsConstructor
@Builder
public class MetricPoint implements Serializable {
	private final long step;
	private final double value;
	
	public MetricPoint() {
		step = 0;
		value = 0.0;
	}
}
