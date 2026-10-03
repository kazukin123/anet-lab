package io.github.kazukin123.anetlab.metricsviewer.service;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.AtomicReference;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsSource;
import io.github.kazukin123.anetlab.metricsviewer.infra.RunScanner;

public class IngestScheduler {

	private static final Logger log = LoggerFactory.getLogger(IngestScheduler.class);

	private final RunScanner runScanner;
	private final MetricsIngestor ingestor;
	private final GzipInputSessions gzipSessions;
	private final RunWarningRegistry warningRegistry;
	private final AtomicReference<Set<String>> priorityRunIds =
			new AtomicReference<>(Set.of());
	private int priorityCursor;
	private int backgroundCursor;
	private int cycleSlot;
	private List<String> cyclePriorityRuns = List.of();
	private List<String> cycleBackgroundRuns = List.of();
	private final Set<String> cycleExhaustedRuns = new HashSet<>();

	public IngestScheduler(
			RunScanner runScanner,
			MetricsIngestor ingestor,
			GzipInputSessions gzipSessions,
			RunWarningRegistry warningRegistry) {
		this.runScanner = runScanner;
		this.ingestor = ingestor;
		this.gzipSessions = gzipSessions;
		this.warningRegistry = warningRegistry;
	}

	public void replacePriority(Set<String> runIds) {
		priorityRunIds.set(Set.copyOf(new LinkedHashSet<>(runIds)));
	}

	Set<String> priorityRunIds() {
		return priorityRunIds.get();
	}

	public boolean runNextBlock() {
		if (cycleSlot == 0) refreshWorkSet();

		// 4 slot周期の現在位置に従い、優先Runまたは背景Runへ1 blockを配分する。
		final boolean preferPriority = cycleSlot < 3;
		AttemptResult result = preferPriority
				? attemptNext(cyclePriorityRuns, true)
				: attemptNext(cycleBackgroundRuns, false);
		if (!result.consumedSlot()) {
			result = preferPriority
					? attemptNext(cycleBackgroundRuns, false)
					: attemptNext(cyclePriorityRuns, true);
		}
		cycleSlot = (cycleSlot + 1) % 4;
		return result.immediateRetry();
	}

	private void refreshWorkSet() {
		cycleExhaustedRuns.clear();
		final Set<String> priorityAtScanStart = priorityRunIds.get();
		final List<String> allRuns = runScanner.listRunId();
		final Set<String> existing = Set.copyOf(allRuns);
		priorityRunIds.updateAndGet(current -> {
			final Set<String> retained = new LinkedHashSet<>(current);
			// scan開始後に追加されたpriorityを、古いscan結果で削除しない。
			retained.removeIf(runId -> priorityAtScanStart.contains(runId)
					&& !existing.contains(runId));
			return Set.copyOf(retained);
		});
		final Set<Path> runDirs = allRuns.stream().map(runScanner::resolveRunDir).collect(
				java.util.stream.Collectors.toUnmodifiableSet());
		gzipSessions.retainRuns(runDirs);
		ingestor.retainRuns(runDirs);
		warningRegistry.retainRuns(existing);

		final Set<String> priority = priorityRunIds.get();
		final List<String> selected = new ArrayList<>();
		final List<String> background = new ArrayList<>();
		for (String runId : allRuns) {
			if (priority.contains(runId)) selected.add(runId);
			else background.add(runId);
		}
		cyclePriorityRuns = List.copyOf(selected);
		cycleBackgroundRuns = List.copyOf(background);
	}

	private AttemptResult attemptNext(List<String> runIds, boolean priority) {
		if (runIds.isEmpty()) return AttemptResult.NONE;
		int cursor = Math.floorMod(priority ? priorityCursor : backgroundCursor, runIds.size());
		for (int attempt = 0; attempt < runIds.size(); attempt++) {
			final String runId = runIds.get(cursor);
			cursor = (cursor + 1) % runIds.size();
			if (priority) priorityCursor = cursor;
			else backgroundCursor = cursor;
			if (cycleExhaustedRuns.contains(runId)) continue;
			final AttemptResult result = ingestOne(runId);
			if (result.exhausted()) cycleExhaustedRuns.add(runId);
			if (result.consumedSlot()) return result;
		}
		return AttemptResult.NONE;
	}

	private AttemptResult ingestOne(String runId) {
		final Path runDir = runScanner.resolveRunDir(runId);
		try {
			warnForDuplicateSources(runId, runDir);
			final MetricsSource source = MetricsSource.select(runDir).orElse(null);
			if (source == null) {
				ingestor.forgetRun(runDir);
				return AttemptResult.EXHAUSTED;
			}
			final MetricsIngestor.IngestOutcome outcome =
					ingestor.ingestBlock(runId, runDir, source);
			return new AttemptResult(
					outcome.didWork() || outcome.immediateRetry(),
					outcome.immediateRetry(),
					!outcome.immediateRetry());
		} catch (Exception e) {
			log.warn("Failed to ingest Metrics source: run={} message={}", runId, e.getMessage());
			return AttemptResult.EXHAUSTED;
		}
	}

	private void warnForDuplicateSources(String runId, Path runDir) {
		if (Files.isRegularFile(runDir.resolve("metrics.jsonl"))
				&& Files.isRegularFile(runDir.resolve("metrics.jsonl.gz"))
				&& warningRegistry.firstDuplicateSource(runId)) {
			log.warn("Both metrics.jsonl and metrics.jsonl.gz exist; using metrics.jsonl: run={}", runId);
		}
	}

	private record AttemptResult(
			boolean consumedSlot,
			boolean immediateRetry,
			boolean exhausted) {
		private static final AttemptResult NONE = new AttemptResult(false, false, false);
		private static final AttemptResult EXHAUSTED = new AttemptResult(false, false, true);
	}
}
