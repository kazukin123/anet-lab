package io.github.kazukin123.anetlab.metricsviewer.view.model;

import java.util.List;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;

@Data
@Builder
@AllArgsConstructor
public class RunInfo {
	private String id;
	private RunStats stats;
	private List<TagInfo> tags;
}
