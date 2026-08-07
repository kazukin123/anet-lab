package io.github.kazukin123.anetlab.metricsviewer.service;

import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

import org.springframework.stereotype.Component;

/**
 * 同名Runが作業セットへ連続して存在する間、世代を跨いで同じWARNを抑止する。
 */
@Component
public class RunWarningRegistry {

	private static final String DUPLICATE_SOURCE = "duplicate_source";
	private static final String SCALAR_SKIP = "scalar_skip";

	private final Set<WarningKey> issuedWarnings = ConcurrentHashMap.newKeySet();

	public boolean firstDuplicateSource(String runId) {
		return issuedWarnings.add(new WarningKey(runId, DUPLICATE_SOURCE, "", ""));
	}

	public boolean firstScalarSkip(String runId, String tag, String reason) {
		return issuedWarnings.add(new WarningKey(runId, SCALAR_SKIP, tag, reason));
	}

	public void retainRuns(Set<String> runIds) {
		issuedWarnings.removeIf(warning -> !runIds.contains(warning.runId()));
	}

	private record WarningKey(String runId, String type, String tag, String reason) {
	}
}
