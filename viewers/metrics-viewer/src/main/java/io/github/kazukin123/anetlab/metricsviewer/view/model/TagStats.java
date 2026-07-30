package io.github.kazukin123.anetlab.metricsviewer.view.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;

@Data
@Builder
@AllArgsConstructor
public class TagStats {
	private long minStep;
	private long maxStep;
	private long count;
	private double lastValue;
	private double minValue;
	private double maxValue;
	private double mean;
	private double variance;
	private double stdDev;
}
