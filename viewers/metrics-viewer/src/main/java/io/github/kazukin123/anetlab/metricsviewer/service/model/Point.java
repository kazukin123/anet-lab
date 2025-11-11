package io.github.kazukin123.anetlab.metricsviewer.service.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;

@Data
@AllArgsConstructor
@Builder
public class Point {
	private final int step;
	private final float value;
	
	public Point() {
		step = 0;
		value = 0.0f;
	}
}
