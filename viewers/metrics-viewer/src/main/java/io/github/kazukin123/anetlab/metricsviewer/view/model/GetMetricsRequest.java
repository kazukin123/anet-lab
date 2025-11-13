package io.github.kazukin123.anetlab.metricsviewer.view.model;

import java.util.Map;

import lombok.Data;

@Data
public class GetMetricsRequest {
	private Map<String, Map<String, Integer>> runTagMap;
}
