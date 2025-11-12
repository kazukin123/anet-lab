package io.github.kazukin123.anetlab.metricsviewer.view.model;

import java.util.List;

import lombok.Data;

@Data
public class GetMetricsResponse {
	private List<TagTrace> data;
}
