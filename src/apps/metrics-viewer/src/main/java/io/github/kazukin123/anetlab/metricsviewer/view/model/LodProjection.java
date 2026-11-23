package io.github.kazukin123.anetlab.metricsviewer.view.model;

public record LodProjection(String kind, MinMaxProjection minMax, LodSummary summary) {

	public LodProjection(MinMaxProjection minMax, LodSummary summary) {
		this("lod", minMax, summary);
	}
}
