package io.github.kazukin123.anetlab.metricsviewer.view.model;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;

@Data
@Builder
@AllArgsConstructor
public class RunStats {
	private int maxStep;
	
	public RunStats() {
		this.maxStep = 0;
	}
}
