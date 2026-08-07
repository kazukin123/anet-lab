package io.github.kazukin123.anetlab.metricsviewer.view.model;

import java.util.List;
import java.util.UUID;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;

@Data
@Builder
@AllArgsConstructor
public class RunInfo {
	private String id;
	private UUID generation;
	private RunStats stats;
	private IngestInfo ingest;
	private List<TagInfo> tags;
}
