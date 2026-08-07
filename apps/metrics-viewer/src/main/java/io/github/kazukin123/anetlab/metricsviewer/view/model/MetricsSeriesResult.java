package io.github.kazukin123.anetlab.metricsviewer.view.model;

import java.util.List;
import java.util.UUID;

import com.fasterxml.jackson.annotation.JsonInclude;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;

@Data
@Builder
@AllArgsConstructor
public class MetricsSeriesResult {
	private String runId;
	private String tagKey;
	private UUID generation;
	private long fromStep;
	private long toStep;
	private String availability;
	private int pointBudget;
	private Integer level;
	private Long bucketWidth;
	@JsonInclude(JsonInclude.Include.NON_EMPTY)
	private List<Issue> issues;
	private Object projection;
}
