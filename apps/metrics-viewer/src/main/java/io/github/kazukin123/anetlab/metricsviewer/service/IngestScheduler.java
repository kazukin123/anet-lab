package io.github.kazukin123.anetlab.metricsviewer.service;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.nio.file.attribute.BasicFileAttributes;
import java.nio.file.attribute.FileTime;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.LongSupplier;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsCacheDatabase;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsCacheDatabase.CacheInspection;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsCacheDatabase.IngestState;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsSource;

/** 単一writerが全workspaceの検査・配分・停止状態と入力resourceを所有する。 */
public class IngestScheduler {

	private static final Logger log = LoggerFactory.getLogger(IngestScheduler.class);
	private static final int INSPECTION_RUN_LIMIT = 128;
	private static final long INSPECTION_NANOS = 50_000_000L;
	private static final long SCAN_INTERVAL_NANOS = 10_000_000_000L;

	private final Supplier<List<Path>> discoverRuns;
	private final MetricsIngestor ingestor;
	private final GzipInputSessions gzipSessions;
	private final RunWarningRegistry warningRegistry;
	private final LongSupplier nanoTime;
	private final long scanIntervalNanos;
	private final AtomicReference<Set<Path>> priorityRunDirs = new AtomicReference<>(Set.of());
	private final AtomicBoolean rescanRequested = new AtomicBoolean();
	private final Map<Path, RunWork> work = new LinkedHashMap<>();
	private final Map<Path, Integer> workspaceRunCursors = new HashMap<>();
	private final Map<Path, InspectionSummary> inspectionSummaries = new HashMap<>();
	private final Set<Path> cycleExhaustedRuns = new HashSet<>();
	private Iterator<Path> scan;
	private Set<Path> seen;
	private Set<Path> priorityAtScanStart = Set.of();
	private long nextScan;
	private boolean scanStarted;
	private boolean scanFailed;
	private boolean closed;
	private int workspaceCursor;
	private int priorityCursor;
	private int priorityProbeCursor;
	private int cycleSlot;

	public IngestScheduler(Supplier<List<Path>> discoverRuns, MetricsIngestor ingestor, GzipInputSessions gzipSessions,
			RunWarningRegistry warningRegistry) {
		this(discoverRuns, ingestor, gzipSessions, warningRegistry, System::nanoTime, SCAN_INTERVAL_NANOS);
	}

	IngestScheduler(Supplier<List<Path>> discoverRuns, MetricsIngestor ingestor, GzipInputSessions gzipSessions,
			RunWarningRegistry warningRegistry, LongSupplier nanoTime, long scanIntervalNanos) {
		if (scanIntervalNanos < 0)
			throw new IllegalArgumentException("scanIntervalNanos must be non-negative");
		this.discoverRuns = discoverRuns;
		this.ingestor = ingestor;
		this.gzipSessions = gzipSessions;
		this.warningRegistry = warningRegistry;
		this.nanoTime = nanoTime;
		this.scanIntervalNanos = scanIntervalNanos;
	}

	public void replacePriorityPaths(Set<Path> runDirs) {
		final Set<Path> normalized = runDirs.stream().map(p -> p.toAbsolutePath().normalize())
				.collect(Collectors.toUnmodifiableSet());
		priorityRunDirs.set(normalized);
		gzipSessions.replacePriority(normalized);
		rescanRequested.set(true);
	}

	Set<Path> priorityRunPaths() {
		return priorityRunDirs.get();
	}

	public boolean runNextBlock() {
		if (closed)
			throw new IllegalStateException("Metrics ingest scheduler is closed");
		if (cycleSlot == 0)
			cycleExhaustedRuns.clear();
		// 背景block中の追記も次の優先slotで拾い、cycle末尾まで待たせない。
		probePriorityAttributes();
		scanSegment();

		// terminal・停止Runは属性検査で変化を見つけるまでblock配分へ戻さない。
		final Set<Path> priority = priorityRunDirs.get();
		final List<Path> selected = new ArrayList<>();
		final List<Path> background = new ArrayList<>();
		for (Map.Entry<Path, RunWork> entry : work.entrySet()) {
			if (!entry.getValue().runnable)
				continue;
			if (priority.contains(entry.getKey()))
				selected.add(entry.getKey());
			else
				background.add(entry.getKey());
		}
		final boolean preferPriority = cycleSlot < 3;
		AttemptResult result = preferPriority ? attemptPriority(selected) : attemptBackground(background);
		if (!result.consumedSlot()) {
			result = preferPriority ? attemptBackground(background) : attemptPriority(selected);
		}
		cycleSlot = (cycleSlot + 1) % 4;
		return result.immediateRetry() || scan != null || hasActionableWork();
	}

	private boolean hasActionableWork() {
		// 現blockの完了と待ち行列の枯渇を分け、枠待ちだけならbusy loopを避ける。
		for (Map.Entry<Path, RunWork> entry : work.entrySet()) {
			final RunWork job = entry.getValue();
			if (!job.runnable)
				continue;
			if (job.waitingReason == null || job.observation == null || job.observation.source() == null
					|| !"jsonl.gz".equals(job.observation.source().kind())
					|| gzipSessions.admissionReason(entry.getKey()) == null)
				return true;
		}
		return false;
	}

	private void probePriorityAttributes() {
		final List<Path> priority = priorityRunDirs.get().stream().sorted().toList();
		if (priority.isEmpty())
			return;
		final long begin = nanoTime.getAsLong();
		// 選択Runはblock境界で属性だけを見る。無変化ならDB・本文へ進まない。
		for (int index = 0; index < Math.min(priority.size(), INSPECTION_RUN_LIMIT); index++) {
			if (index > 0 && nanoTime.getAsLong() - begin >= INSPECTION_NANOS)
				break;
			priorityProbeCursor = Math.floorMod(priorityProbeCursor, priority.size());
			final Path run = priority.get(priorityProbeCursor);
			priorityProbeCursor = (priorityProbeCursor + 1) % priority.size();
			final RunWork job = work.get(run);
			if (job == null)
				continue;
			final Observation current = Observation.capture(run);
			if (current.equals(job.observation))
				continue;
			job.observation = current;
			job.inspection = null;
			job.failureReason = null;
			job.runnable = true;
			cycleExhaustedRuns.remove(run);
			ingestor.invalidateObservation(run);
		}
	}

	private void scanSegment() {
		final boolean requested = rescanRequested.getAndSet(false);
		if (requested && scan != null) {
			final Iterator<Path> pending = scan;
			// 全走査の残りを保持したまま、新しい優先対象を次区間の先頭へ挿入する。
			scan = Stream.concat(
					priorityRunDirs.get().stream().sorted()
							.filter(p -> Files.isRegularFile(p.resolve("metrics.jsonl"))
									|| Files.isRegularFile(p.resolve("metrics.jsonl.gz"))),
					java.util.stream.StreamSupport.stream(java.util.Spliterators.spliteratorUnknownSize(pending, 0),
							false))
					.distinct().iterator();
		}
		if (scan == null && (requested || ((!scanStarted || nanoTime.getAsLong() - nextScan >= 0) && cycleSlot == 0))) {
			try {
				final long listingStart = nanoTime.getAsLong();
				priorityAtScanStart = priorityRunDirs.get();
				final List<Path> discovered = discoverRuns.get();
				final Set<Path> existing = discovered.stream().map(p -> p.toAbsolutePath().normalize())
						.collect(Collectors.toSet());
				// 優先要求が来たRunは一覧の末尾にあっても先に検査する。
				scan = Stream.concat(priorityAtScanStart.stream().sorted().filter(existing::contains),
						discovered.stream().map(p -> p.toAbsolutePath().normalize())).distinct().iterator();
				seen = new HashSet<>();
				scanStarted = true;
				scanFailed = false;
				log.debug("Metrics workspace listing: runs={} elapsedMs={}", discovered.size(),
						(nanoTime.getAsLong() - listingStart) / 1_000_000);
			}
			catch (UncheckedIOException e) {
				// 列挙失敗を空のworkspaceと解釈せず、開始済み作業とstreamを残す。
				if (!scanFailed)
					log.error("Metrics workspace inspection failed: message={}", e.getMessage());
				scanFailed = true;
				scanStarted = true;
				nextScan = nanoTime.getAsLong() + scanIntervalNanos;
				return;
			}
		}
		if (scan == null)
			return;
		final long begin = nanoTime.getAsLong();
		int count = 0;
		while (scan.hasNext() && count < INSPECTION_RUN_LIMIT
				&& (count == 0 || nanoTime.getAsLong() - begin < INSPECTION_NANOS)) {
			final Path run = scan.next();
			seen.add(run);
			inspectRun(run, work.computeIfAbsent(run, ignored -> new RunWork()));
			count++;
		}
		log.debug("Metrics workspace inspection segment: runs={} elapsedMs={}", count,
				(nanoTime.getAsLong() - begin) / 1_000_000);
		if (scan.hasNext())
			return;

		// 完了した列挙だけを登録解除の根拠にし、走査中の部分集合でresourceを捨てない。
		work.keySet().retainAll(seen);
		workspaceRunCursors.keySet().removeIf(ws -> seen.stream().noneMatch(p -> p.getParent().getParent().equals(ws)));
		priorityRunDirs.updateAndGet(current -> {
			final Set<Path> retained = new LinkedHashSet<>(current);
			retained.removeIf(p -> priorityAtScanStart.contains(p) && !seen.contains(p));
			return Set.copyOf(retained);
		});
		gzipSessions.replacePriority(priorityRunDirs.get());
		gzipSessions.retainRuns(seen);
		ingestor.retainRuns(seen);
		warningRegistry.retainRuns(seen.stream().map(Path::toString).collect(Collectors.toSet()));
		logInspectionSummaries();
		scan = null;
		nextScan = nanoTime.getAsLong() + scanIntervalNanos;
	}

	private void logInspectionSummaries() {
		final Map<Path, List<RunWork>> grouped = new LinkedHashMap<>();
		for (Map.Entry<Path, RunWork> entry : work.entrySet()) {
			grouped.computeIfAbsent(entry.getKey().getParent().getParent(), ignored -> new ArrayList<>())
					.add(entry.getValue());
		}
		// 毎回の検査ではなく、確定したworkspace集計の初回・変化だけを公開する。
		for (Map.Entry<Path, List<RunWork>> entry : grouped.entrySet()) {
			final Map<String, Long> reasons = new java.util.TreeMap<>();
			long pending = 0;
			long failed = 0;
			for (RunWork job : entry.getValue()) {
				if (job.failureReason != null || job.state == IngestState.ERROR)
					failed++;
				if (!job.runnable)
					continue;
				pending++;
				final String reason = job.inspection == null || job.inspection.reason() == null
						? "validation_required"
						: job.inspection.reason();
				reasons.merge(reason, 1L, Long::sum);
			}
			final InspectionSummary summary = new InspectionSummary(entry.getValue().size(), pending, failed,
					Map.copyOf(reasons));
			if (!summary.equals(inspectionSummaries.put(entry.getKey(), summary))) {
				log.info("Metrics workspace inspected: workspace={} runs={} pending={} failed={} reasons={}",
						entry.getKey().getFileName(), summary.runs(), summary.pending(), summary.failed(), reasons);
			}
		}
		inspectionSummaries.keySet().retainAll(grouped.keySet());
	}

	private record InspectionSummary(long runs, long pending, long failed, Map<String, Long> reasons) {
	}

	private void inspectRun(Path run, RunWork job) {
		final Observation observation = Observation.capture(run);
		if (observation.equals(job.observation))
			return;
		try {
			if (observation.error() != null)
				throw new IOException(observation.error());
			if (observation.source() == null)
				return;
			if (job.observation != null)
				ingestor.invalidateObservation(run);
			final CacheInspection inspection = ingestor.inspectCache(run, observation.source());
			job.runnable = true;
			job.failureReason = null;
			job.observation = observation;
			job.inspection = inspection;
			job.state = inspection == null ? null : inspection.state();
			if (inspection != null && inspection.reason() == null)
				job.lastDetection = null;
			if (inspection != null && inspection.reason() != null) {
				final String detection = inspection.reason() + ":" + inspection.generation() + ":"
						+ inspection.schemaVersion();
				if (!detection.equals(job.lastDetection)) {
					log.info(
							"Metrics cache detected: workspace={} run={} generation={} reason={}"
									+ " oldVersion={} targetVersion={}",
							workspace(run), run.getFileName(),
							inspection.generation() == null ? "unknown" : inspection.generation(), inspection.reason(),
							inspection.schemaVersion(), MetricsCacheDatabase.SCHEMA_VERSION);
				}
				job.lastDetection = detection;
			}
		}
		catch (IOException | SQLException e) {
			stopRun(run, job, "inspection", e);
		}
	}

	private AttemptResult attemptPriority(List<Path> runs) {
		if (runs.isEmpty())
			return AttemptResult.NONE;
		int cursor = Math.floorMod(priorityCursor, runs.size());
		for (int attempt = 0; attempt < runs.size(); attempt++) {
			final Path run = runs.get(cursor);
			cursor = (cursor + 1) % runs.size();
			priorityCursor = cursor;
			if (cycleExhaustedRuns.contains(run))
				continue;
			final AttemptResult result = ingestOne(run);
			if (result.exhausted())
				cycleExhaustedRuns.add(run);
			if (result.consumedSlot()) return result;
		}
		return AttemptResult.NONE;
	}

	private AttemptResult attemptBackground(List<Path> runs) {
		if (runs.isEmpty())
			return AttemptResult.NONE;
		final Map<Path, List<Path>> grouped = new LinkedHashMap<>();
		for (Path run : runs) {
			grouped.computeIfAbsent(run.getParent().getParent(), ignored -> new ArrayList<>()).add(run);
		}
		final List<Path> workspaces = List.copyOf(grouped.keySet());

		// workspaceを先に巡回し、それぞれのRun数に他workspaceの待ち時間を比例させない。
		for (int attempt = 0; attempt < workspaces.size(); attempt++) {
			workspaceCursor = Math.floorMod(workspaceCursor, workspaces.size());
			final Path ws = workspaces.get(workspaceCursor);
			workspaceCursor = (workspaceCursor + 1) % workspaces.size();
			final List<Path> candidates = grouped.get(ws);
			int cursor = Math.floorMod(workspaceRunCursors.getOrDefault(ws, 0), candidates.size());
			for (int index = 0; index < candidates.size(); index++) {
				final Path run = candidates.get(cursor);
				cursor = (cursor + 1) % candidates.size();
				workspaceRunCursors.put(ws, cursor);
				if (cycleExhaustedRuns.contains(run))
					continue;
				final AttemptResult result = ingestOne(run);
				if (result.exhausted())
					cycleExhaustedRuns.add(run);
				if (result.consumedSlot())
					return result;
			}
		}
		return AttemptResult.NONE;
	}

	private AttemptResult ingestOne(Path run) {
		final RunWork job = work.get(run);
		try {
			final Observation before = Observation.capture(run);
			if (before.error() != null)
				throw new IOException(before.error());
			final MetricsSource source = before.source();
			if (source == null) {
				gzipSessions.close(run);
				ingestor.releaseRun(run);
				work.remove(run);
				return AttemptResult.EXHAUSTED;
			}
			final boolean sourceChanged = job.observation != null
					&& (!java.util.Objects.equals(before.source(), job.observation.source())
							|| !java.util.Objects.equals(before.sourceFile(), job.observation.sourceFile()));
			if (sourceChanged)
				ingestor.invalidateObservation(run);
			// rawへ切り替わったRunの旧decoderは継続できないため、取り込み前に枠を解放する。
			if (!"jsonl.gz".equals(source.kind()))
				gzipSessions.close(run);
			if (Files.isRegularFile(run.resolve("metrics.jsonl"))
					&& Files.isRegularFile(run.resolve("metrics.jsonl.gz"))
					&& warningRegistry.firstDuplicateSource(run.toString())) {
				log.warn("Both metrics.jsonl and metrics.jsonl.gz exist; using metrics.jsonl: run={}"
						+ " workspace={} path={}", run.getFileName(), workspace(run), run);
			}
			if ("jsonl.gz".equals(source.kind()) && !gzipSessions.hasSession(run, source)) {
				gzipSessions.close(run);
				if (sourceChanged || job.inspection == null || job.inspection.inputRequired()) {
					final String reason = gzipSessions.admissionReason(run);
					if (reason != null)
						return waitForResource(run, job, reason);
				}
			}
			if (job.waitingReason != null) {
				log.info("Metrics cache resumed: workspace={} run={} reason={} waitMs={}", workspace(run),
						run.getFileName(), job.waitingReason, (nanoTime.getAsLong() - job.waitingStarted) / 1_000_000);
				job.waitingReason = null;
			}
			final MetricsIngestor.IngestOutcome outcome =
					ingestor.ingestBlock(run.getFileName().toString(), run, source);
			final Observation after = Observation.capture(run);
			job.state = outcome.state();
			if (!outcome.immediateRetry())
				job.lastDetection = null;
			job.observation = after;
			job.runnable = outcome.immediateRetry() || !java.util.Objects.equals(before.source(), after.source())
					|| !java.util.Objects.equals(before.sourceFile(), after.sourceFile());
			return new AttemptResult(outcome.didWork() || job.runnable, job.runnable, !job.runnable);
		}
		catch (GzipInputSessions.ResourceUnavailableException e) {
			job.observation = Observation.capture(run);
			return waitForResource(run, job, e.reason());
		}
		catch (IOException | SQLException e) {
			stopRun(run, job, "ingest", e);
			return AttemptResult.EXHAUSTED;
		}
		catch (RuntimeException e) {
			throw e;
		} catch (Exception e) {
			throw new IllegalStateException("Unexpected Metrics ingest failure: path=" + run, e);
		}
	}

	private AttemptResult waitForResource(Path run, RunWork job, String reason) {
		if (!reason.equals(job.waitingReason)) {
			job.waitingReason = reason;
			job.waitingStarted = nanoTime.getAsLong();
			log.info("Metrics cache waiting: workspace={} run={} generation={} reason={}", workspace(run),
					run.getFileName(),
					job.inspection == null || job.inspection.generation() == null
							? "unknown"
							: job.inspection.generation(),
					reason);
		}
		return AttemptResult.EXHAUSTED;
	}

	private void stopRun(Path run, RunWork job, String phase, Exception failure) {
		// rollback・stream解放後の観測を保存。同じ属性では再接続も再試行もしない。
		gzipSessions.close(run);
		job.observation = Observation.capture(run);
		job.failureReason = failure.getMessage() == null ? failure.getClass().getName() : failure.getMessage();
		job.runnable = false;
		job.waitingReason = null;
		log.error("Metrics cache failed: workspace={} run={} path={} phase={} message={} {}", workspace(run),
				run.getFileName(), run, phase, job.failureReason, ingestor.progressDiagnostics(run));
		ingestor.forgetRun(run);
	}

	void close() {
		if (closed)
			return;
		closed = true;
		gzipSessions.closeAll();
		ingestor.close();
		warningRegistry.retainRuns(Set.of());
		work.clear();
		workspaceRunCursors.clear();
		inspectionSummaries.clear();
		scan = null;
	}

	private static String workspace(Path run) {
		return run.getParent().getParent().getFileName().toString();
	}

	private static final class RunWork {
		private Observation observation;
		private CacheInspection inspection;
		private IngestState state;
		private String failureReason;
		private String waitingReason;
		private String lastDetection;
		private long waitingStarted;
		private boolean runnable = true;
	}

	private record FileObservation(long size, FileTime modified, FileTime created, Object key) {
		private static FileObservation capture(Path file) throws IOException {
			try {
				final BasicFileAttributes attributes = Files.readAttributes(file, BasicFileAttributes.class);
				return new FileObservation(attributes.size(), attributes.lastModifiedTime(), attributes.creationTime(),
						attributes.fileKey());
			}
			catch (NoSuchFileException e) {
				return null;
			}
		}
	}

	private record Observation(MetricsSource source, FileObservation sourceFile, FileObservation cache,
			FileObservation wal, String error) {
		private static Observation capture(Path run) {
			try {
				final MetricsSource source = MetricsSource.select(run).orElse(null);
				final FileObservation wal = FileObservation.capture(run.resolve("metrics_cache.db-wal"));
				return new Observation(source, source == null ? null : FileObservation.capture(source.path()),
						FileObservation.capture(run.resolve("metrics_cache.db")),
						wal == null || wal.size() == 0 ? null : wal, null);
			}
			catch (IOException e) {
				return new Observation(null, null, null, null, e.getClass().getName() + ": " + e.getMessage());
			}
		}
	}

	private record AttemptResult(boolean consumedSlot, boolean immediateRetry, boolean exhausted) {
		private static final AttemptResult NONE = new AttemptResult(false, false, false);
		private static final AttemptResult EXHAUSTED = new AttemptResult(false, false, true);
	}
}
