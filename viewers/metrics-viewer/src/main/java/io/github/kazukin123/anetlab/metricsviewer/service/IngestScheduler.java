package io.github.kazukin123.anetlab.metricsviewer.service;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.AtomicReference;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsSource;
import io.github.kazukin123.anetlab.metricsviewer.infra.RunScanner;

@Component
public class IngestScheduler {

	private static final Logger log = LoggerFactory.getLogger(IngestScheduler.class);

	private final RunScanner runScanner;
	private final MetricsIngestor ingestor;
	private final GzipInputSessions gzipSessions;
	private final Set<String> duplicateSourceWarnings = java.util.concurrent.ConcurrentHashMap.newKeySet();
	private final AtomicReference<Set<String>> priorityRunIds =
			new AtomicReference<>(Set.of());
	private int priorityCursor;
	private int backgroundCursor;

	public IngestScheduler(
			RunScanner runScanner,
			MetricsIngestor ingestor,
			GzipInputSessions gzipSessions) {
		this.runScanner = runScanner;
		this.ingestor = ingestor;
		this.gzipSessions = gzipSessions;
	}

	public void replacePriority(Set<String> runIds) {
		priorityRunIds.set(Set.copyOf(new LinkedHashSet<>(runIds)));
	}

	Set<String> priorityRunIds() {
		return priorityRunIds.get();
	}

	public boolean runCycle() {
		final List<String> allRuns = runScanner.listRunId();
		final Set<String> existing = Set.copyOf(allRuns);
		priorityRunIds.updateAndGet(current -> {
			final Set<String> retained = new LinkedHashSet<>(current);
			retained.retainAll(existing);
			return Set.copyOf(retained);
		});
		gzipSessions.retainRuns(allRuns.stream().map(runScanner::resolveRunDir).collect(
				java.util.stream.Collectors.toUnmodifiableSet()));
		duplicateSourceWarnings.retainAll(existing);

		final Set<String> priority = priorityRunIds.get();
		final List<String> selected = new ArrayList<>();
		final List<String> background = new ArrayList<>();
		for (String runId : allRuns) {
			if (priority.contains(runId)) selected.add(runId);
			else background.add(runId);
		}

		boolean didWork = false;
		for (int slot = 0; slot < 4; slot++) {
			final boolean preferPriority = slot < 3;
			boolean slotWorked = preferPriority
					? attemptNext(selected, true)
					: attemptNext(background, false);
			if (!slotWorked) {
				slotWorked = preferPriority
						? attemptNext(background, false)
						: attemptNext(selected, true);
			}
			didWork |= slotWorked;
		}
		return didWork;
	}

	private boolean attemptNext(List<String> runIds, boolean priority) {
		if (runIds.isEmpty()) return false;
		int cursor = Math.floorMod(priority ? priorityCursor : backgroundCursor, runIds.size());
		for (int attempt = 0; attempt < runIds.size(); attempt++) {
			final String runId = runIds.get(cursor);
			cursor = (cursor + 1) % runIds.size();
			if (priority) priorityCursor = cursor;
			else backgroundCursor = cursor;
			if (ingestOne(runId)) return true;
		}
		return false;
	}

	private boolean ingestOne(String runId) {
		final Path runDir = runScanner.resolveRunDir(runId);
		try {
			warnForDuplicateSources(runId, runDir);
			final MetricsSource source = MetricsSource.select(runDir).orElse(null);
			if (source == null) return false;
			return ingestor.ingestBlock(runId, runDir, source).didWork();
		} catch (Exception e) {
			log.warn("Failed to ingest Metrics source: run={} message={}", runId, e.getMessage());
			return false;
		}
	}

	private void warnForDuplicateSources(String runId, Path runDir) {
		if (Files.isRegularFile(runDir.resolve("metrics.jsonl"))
				&& Files.isRegularFile(runDir.resolve("metrics.jsonl.gz"))
				&& duplicateSourceWarnings.add(runId)) {
			log.warn("Both metrics.jsonl and metrics.jsonl.gz exist; using metrics.jsonl: run={}", runId);
		}
	}
}
