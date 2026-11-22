package io.github.kazukin123.anetlab.metricsviewer.service;

import java.io.IOException;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.util.List;
import java.util.Set;

import org.springframework.stereotype.Service;

import io.github.kazukin123.anetlab.metricsviewer.view.model.RunConfigResponse;
import io.github.kazukin123.anetlab.metricsviewer.view.model.RunConfigResponse.Entry;
import io.github.kazukin123.anetlab.metricsviewer.view.model.RunConfigResponse.RunConfig;

@Service
public class RunConfigService {

	private final WorkspaceManager workspaceManager;

	public RunConfigService(WorkspaceManager workspaceManager) {
		this.workspaceManager = workspaceManager;
	}

	public RunConfigResponse getConfig(List<String> runIds) {
		// 作業セットを要求の開始時に固定し、登録済みの Run だけを解決する。
		try (WorkspaceManager.Lease lease = workspaceManager.acquireLease()) {
			final Set<String> existingRuns = Set.copyOf(lease.runScanner().listRunId());
			return new RunConfigResponse(runIds.stream().map(runId -> {
				if (!existingRuns.contains(runId)) {
					return new RunConfig(runId, "not_found", "Unknown Run: " + runId, null);
				}
				final Path path = lease.runScanner().resolveRunDir(runId).resolve("config/config_data.txt");
				return readRun(runId, path);
			}).toList());
		}
	}

	private RunConfig readRun(String runId, Path path) {
		// ファイルの状態は Run ごとに扱い、残りの列の読み込みを続ける。
		try {
			final List<Entry> entries = ConfigTextReader.read(path).entrySet().stream()
					.map(entry -> new Entry(entry.getKey(), entry.getValue())).toList();
			return new RunConfig(runId, "ok", null, entries);
		} catch (NoSuchFileException e) {
			return new RunConfig(runId, "missing", path.toString(), null);
		} catch (ConfigTextReader.FormatException e) {
			return new RunConfig(runId, "error", e.getMessage(), null);
		} catch (IOException e) {
			return new RunConfig(runId, "error", path + ": " + e.getMessage(), null);
		}
	}
}
