package io.github.kazukin123.anetlab.metricsviewer.infra.model;

import java.util.List;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;

@Data
@Builder
@AllArgsConstructor
public class MetricsFileBlock {
	
	private final int startOffset;
	private final int endOffset;
	private final List<MetricsFileLine> lines;
	private final long lastModified;
	private final boolean isEOF;
}
