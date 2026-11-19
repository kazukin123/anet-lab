package io.github.kazukin123.anetlab.metricsviewer.view.model;

import java.util.List;

public record LodSummary(
		List<String> steps,
		List<String> mins,
		List<String> maxs,
		List<String> means,
		List<String> minSteps,
		List<String> maxSteps) {
}
