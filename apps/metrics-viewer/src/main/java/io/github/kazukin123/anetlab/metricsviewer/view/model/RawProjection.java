package io.github.kazukin123.anetlab.metricsviewer.view.model;

import java.util.List;

public record RawProjection(String kind, List<String> steps, List<String> values) {

	public RawProjection(List<String> steps, List<String> values) {
		this("raw", steps, values);
	}
}
