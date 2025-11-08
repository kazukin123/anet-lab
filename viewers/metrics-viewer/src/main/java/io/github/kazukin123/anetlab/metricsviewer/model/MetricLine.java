package io.github.kazukin123.anetlab.metricsviewer.model;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class MetricLine {
	private final String type;
	private final String tagKey;
	private final long step;
	private final double values;
}