package io.github.kazukin123.anetlab.metricsviewer.model;

import java.util.List;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;

@Data
@Builder
@AllArgsConstructor
public class MetricFileBlock {
	private final String runId;
	private final int startOffset;
	private final int endOffset;
	private final List<MetricLine> lines;
	private final long lastModified;
	private final boolean isEOF;
}
