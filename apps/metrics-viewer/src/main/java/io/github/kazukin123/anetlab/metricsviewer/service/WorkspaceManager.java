package io.github.kazukin123.anetlab.metricsviewer.service;

import java.io.UncheckedIOException;
import java.util.stream.Collectors;
import java.util.Set;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Supplier;
import java.util.stream.Stream;

import jakarta.annotation.PreDestroy;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import io.github.kazukin123.anetlab.metricsviewer.config.MetricsViewerSettings;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsCacheDatabase;
import io.github.kazukin123.anetlab.metricsviewer.infra.RunScanner;

/** current workspace と workspace 固有サービス群のライフサイクルを管理する。 */
@Component
public class WorkspaceManager {

	private static final Logger log = LoggerFactory.getLogger(WorkspaceManager.class);

	private final Path workspacesDir;
	private final MetricsCacheDatabase database;
	private final MetricsViewerSettings settings;
	private final MetricsQueryCoordinator queryCoordinator;
	private final IngestScheduler scheduler;
	private final Map<String, List<Path>> listedWorkspaceRuns = new HashMap<>();
	private final Map<String, String> workspaceListingFailures = new HashMap<>();
	private final AtomicReference<WorkspaceSnapshot> current;
	private final ReentrantLock ingestGate = new ReentrantLock(true);
	private final AtomicInteger pendingSwitchRequests = new AtomicInteger();
	private final AtomicInteger pendingPriorityRequests = new AtomicInteger();
	private volatile boolean ingestStopRequested;
	private volatile Runnable ingestWakeup = () -> {
	};
	private volatile boolean shutdown;

	@Autowired
	public WorkspaceManager(
			@Value("${metricsviewer.workspaces-dir:workspaces}") String workspacesDirPath,
			@Value("${metricsviewer.initial-workspace:_default}") String initialWorkspace,
			MetricsCacheDatabase database,
			MetricsViewerSettings settings,
			MetricsQueryCoordinator queryCoordinator) {
		this(workspacesDirPath, initialWorkspace, database, settings,
				queryCoordinator, GzipInputSessions::new);
	}

	WorkspaceManager(
			String workspacesDirPath,
			String initialWorkspace,
			MetricsCacheDatabase database,
			MetricsViewerSettings settings,
			MetricsQueryCoordinator queryCoordinator,
			Supplier<GzipInputSessions> gzipSessionsFactory) {
		this(workspacesDirPath, initialWorkspace, database, settings, queryCoordinator, gzipSessionsFactory,
				MetricsIngestor.MAX_BLOCK_LINES);
	}

	WorkspaceManager(String workspacesDirPath, String initialWorkspace, MetricsCacheDatabase database,
			MetricsViewerSettings settings, MetricsQueryCoordinator queryCoordinator,
			Supplier<GzipInputSessions> gzipSessionsFactory, int maxBlockLines) {
		this.workspacesDir = normalizeWorkspacesDir(workspacesDirPath);
		validateInitialWorkspace(initialWorkspace, this.workspacesDir);
		this.database = database;
		this.settings = settings;
		this.queryCoordinator = queryCoordinator;
		final GzipInputSessions gzipSessions = gzipSessionsFactory.get();
		final RunWarningRegistry warnings = new RunWarningRegistry();
		final MetricsIngestor ingestor = new MetricsIngestor(database, gzipSessions, new LodIngestWriter(), warnings,
				maxBlockLines,
				() -> pendingSwitchRequests.get() > 0 || pendingPriorityRequests.get() > 0 || ingestStopRequested);
		this.scheduler = new IngestScheduler(this::discoverRuns, ingestor, gzipSessions, warnings);
		this.current = new AtomicReference<>(createSnapshot(0L, initialWorkspace));
		if (!Files.isDirectory(this.workspacesDir.resolve(initialWorkspace))) {
			log.warn("Initial workspace directory not found: workspace={} path={}",
					initialWorkspace, this.workspacesDir.resolve(initialWorkspace));
		}
	}

	public Path getWorkspacesDir() {
		return workspacesDir;
	}

	public String currentName() {
		return current.get().name();
	}

	public Path currentRunsDir() {
		return current.get().root().resolve("runs");
	}

	public long currentEpoch() {
		return current.get().epoch();
	}

	public List<String> listWorkspaceNames() {
		if (!Files.isDirectory(workspacesDir)) return List.of();
		try (Stream<Path> children = Files.list(workspacesDir)) {
			return children
					.filter(Files::isDirectory)
					.filter(this::isWorkspace)
					.map(path -> path.getFileName().toString())
					.sorted()
					.toList();
		} catch (IOException e) {
			throw new IllegalStateException("Failed to scan workspaces directory: " + workspacesDir, e);
		}
	}

	public Lease acquireLease() {
		while (true) {
			if (shutdown) throw new IllegalStateException("WorkspaceManager is shut down");

			// current読取後に切替でretireされた場合だけ、後継snapshotを読み直して再試行する。
			final WorkspaceSnapshot snapshot = current.get();
			final Lease lease = snapshot.tryAcquire();
			if (lease != null) return lease;
		}
	}

	public SwitchResult switchWorkspace(String name) {
		final SwitchResult result;
		long cancelledEpoch = Long.MIN_VALUE;
		pendingSwitchRequests.incrementAndGet();
		ingestWakeup.run();
		ingestGate.lock();
		try {
			ensureRunning();
			final WorkspaceSnapshot previous = current.get();
			if (previous.name().equals(name)) {
				result = SwitchResult.NO_OP;
			} else if (!listWorkspaceNames().contains(name)) {
				result = SwitchResult.UNKNOWN;
			} else {
				// cycle と同じゲート内で current を再確認してから一度だけ交換する。
				final WorkspaceSnapshot next = createSnapshot(previous.epoch() + 1L, name);
				current.set(next);
				scheduler.replacePriorityPaths(Set.of());
				previous.retire();
				cancelledEpoch = previous.epoch();
				result = SwitchResult.SWITCHED;
			}
		} finally {
			pendingSwitchRequests.decrementAndGet();
			ingestGate.unlock();
		}
		if (result == SwitchResult.SWITCHED) {
			queryCoordinator.cancelWorkspace(cancelledEpoch);
			ingestWakeup.run();
		}
		return result;
	}

	public boolean runIngestCycle() {
		if (ingestStopRequested)
			return false;
		try (Lease lease = acquireLease()) {
			boolean immediateRetry = false;
			for (int slot = 0; slot < 4; slot++) {
				ingestGate.lock();
				try {
					// shutdown開始後は現在blockより先へ進まず、取得済みleaseの解放へ移る。
					if (shutdown || ingestStopRequested)
						return immediateRetry;
					// 切替済みなら旧snapshotへ残りblockを割り当てず、次cycleを即時実行させる。
					if (current.get().epoch() != lease.epoch()) return true;
					immediateRetry = scheduler.runNextBlock();
				} finally {
					ingestGate.unlock();
				}
			}
			return immediateRetry;
		}
	}

	@PreDestroy
	public void shutdown() {
		requestIngestStop();
		ingestGate.lock();
		try {
			if (shutdown) return;
			// terminal状態を先に公開し、新規leaseと切替を止めてからcurrentをretireする。
			shutdown = true;
			scheduler.close();
			listedWorkspaceRuns.clear();
			workspaceListingFailures.clear();
			current.get().retire();
		} finally {
			ingestGate.unlock();
		}
	}

	private void ensureRunning() {
		if (shutdown) throw new IllegalStateException("WorkspaceManager is shut down");
	}

	private WorkspaceSnapshot createSnapshot(long epoch, String name) {
		final Path root = workspacesDir.resolve(name).normalize();
		final RunScanner runScanner = new RunScanner(root.resolve("runs").toString());
		final LodPageCache pageCache = new LodPageCache(settings);
		final MetricsRangeProjector projector = new MetricsRangeProjector(pageCache);
		final MetricsRepository repository = new MetricsRepository(
				runScanner, database, settings, projector);
		return new WorkspaceSnapshot(
				epoch, name, root, runScanner, pageCache, repository);
	}

	Set<Path> priorityRunPaths() {
		return scheduler.priorityRunPaths();
	}

	private List<Path> discoverRuns() {
		final ArrayList<Path> runs = new ArrayList<>();
		final List<String> names;
		try {
			names = listWorkspaceNames();
		}
		catch (IllegalStateException e) {
			if (e.getCause() instanceof IOException io)
				throw new UncheckedIOException(io);
			throw e;
		}
		listedWorkspaceRuns.keySet().retainAll(names);
		workspaceListingFailures.keySet().retainAll(names);
		for (String name : names) {
			final Path dir = workspacesDir.resolve(name).resolve("runs");
			try {
				final List<Path> listed;
				if (!Files.exists(dir))
					listed = List.of();
				else
					try (Stream<Path> children = Files.list(dir)) {
						listed = children.filter(Files::isDirectory)
								.filter(p -> Files.isRegularFile(p.resolve("metrics.jsonl"))
										|| Files.isRegularFile(p.resolve("metrics.jsonl.gz")))
								.sorted().toList();
					}
				listedWorkspaceRuns.put(name, listed);
				workspaceListingFailures.remove(name);
				runs.addAll(listed);
			}
			catch (IOException e) {
				// 一部workspaceの列挙失敗を空集合と誤認せず、他workspaceの検査を継続する。
				final String message = e.getClass().getName() + ": " + e.getMessage();
				if (!message.equals(workspaceListingFailures.put(name, message)))
					log.error("Metrics workspace listing failed: workspace={} path={} message={}", name, dir, message);
				runs.addAll(listedWorkspaceRuns.getOrDefault(name, List.of()));
			}
		}
		return runs;
	}

	public void replacePriority(long epoch, Set<Path> paths) {
		final Set<Path> normalized = paths.stream().map(p -> p.toAbsolutePath().normalize())
				.collect(Collectors.toUnmodifiableSet());
		// 通常の同一選択の再要求は、稼働blockを中断せず検査周期も早めない。
		if (!shutdown && current.get().epoch() == epoch && scheduler.priorityRunPaths().equals(normalized))
			return;
		pendingPriorityRequests.incrementAndGet();
		ingestWakeup.run();
		ingestGate.lock();
		try {
			if (!shutdown && current.get().epoch() == epoch)
				scheduler.replacePriorityPaths(normalized);
		}
		finally {
			pendingPriorityRequests.decrementAndGet();
			ingestGate.unlock();
		}
		ingestWakeup.run();
	}

	void setIngestWakeup(Runnable wakeup) {
		ingestWakeup = wakeup;
	}

	void requestIngestStop() {
		ingestStopRequested = true;
		ingestWakeup.run();
	}

	void closeIngestResources() {
		ingestGate.lock();
		try {
			scheduler.close();
			listedWorkspaceRuns.clear();
			workspaceListingFailures.clear();
		}
		finally {
			ingestGate.unlock();
		}
	}

	private boolean isWorkspace(Path path) {
		return Files.isDirectory(path.resolve("runs")) || Files.isDirectory(path.resolve("config"));
	}

	private static Path normalizeWorkspacesDir(String value) {
		if (value == null || value.isBlank()) {
			throw invalidConfig("metricsviewer.workspaces-dir", value, "non-empty local path");
		}
		final String trimmed = value.trim();
		final Path normalized = Paths.get(trimmed).toAbsolutePath().normalize();
		final String normalizedText = normalized.toString();
		if (trimmed.startsWith("\\\\") || trimmed.startsWith("//")
				|| normalizedText.startsWith("\\\\") || normalizedText.startsWith("//")) {
			throw invalidConfig("metricsviewer.workspaces-dir", value, "local path without UNC root");
		}
		return normalized;
	}

	private static void validateInitialWorkspace(String value, Path workspacesDir) {
		if (value == null || value.isBlank() || value.contains("/") || value.contains("\\")
				|| value.contains("#") || value.contains("//")
				|| value.equals(".") || value.equals("..")
				|| value.matches("^[A-Za-z]:.*")) {
			throw invalidConfig(
					"metricsviewer.initial-workspace", value, "single direct child directory name");
		}
		final Path candidate = Paths.get(value);
		final Path resolved = workspacesDir.resolve(candidate).normalize();
		if (candidate.isAbsolute() || candidate.getNameCount() != 1
				|| !resolved.getParent().equals(workspacesDir)) {
			throw invalidConfig(
					"metricsviewer.initial-workspace", value, "single direct child directory name");
		}
	}

	private static IllegalArgumentException invalidConfig(String key, Object value, String expected) {
		return new IllegalArgumentException(
				"Invalid configuration: key=" + key + " value=" + value + " expected=" + expected);
	}

	public enum SwitchResult {
		SWITCHED,
		NO_OP,
		UNKNOWN
	}

	public static final class Lease implements AutoCloseable {
		private WorkspaceSnapshot snapshot;

		private Lease(WorkspaceSnapshot snapshot) {
			this.snapshot = snapshot;
		}

		public long epoch() {
			return snapshot.epoch();
		}

		public String name() {
			return snapshot.name();
		}

		public RunScanner runScanner() {
			return snapshot.runScanner();
		}

		public LodPageCache pageCache() {
			return snapshot.pageCache();
		}

		public MetricsRepository repository() {
			return snapshot.repository();
		}

		@Override
		public void close() {
			if (snapshot == null) return;
			final WorkspaceSnapshot releasing = snapshot;
			snapshot = null;
			releasing.release();
		}
	}

	private static final class WorkspaceSnapshot {
		private final long epoch;
		private final String name;
		private final Path root;
		private final RunScanner runScanner;
		private final LodPageCache pageCache;
		private final MetricsRepository repository;
		private int leaseCount;
		private boolean retired;
		private boolean closed;

		private WorkspaceSnapshot(
				long epoch,
				String name,
				Path root,
				RunScanner runScanner,
				LodPageCache pageCache, MetricsRepository repository) {
			this.epoch = epoch;
			this.name = name;
			this.root = root;
			this.runScanner = runScanner;
			this.pageCache = pageCache;
			this.repository = repository;
		}

		private synchronized Lease tryAcquire() {
			// retire判定と利用者登録を同じmonitor内で行い、closeとの競合を作らない。
			if (retired) return null;
			leaseCount++;
			return new Lease(this);
		}

		private synchronized void retire() {
			// 新規leaseを禁止した後、利用者が無ければclose-on-zeroを直ちに実行する。
			if (retired) return;
			retired = true;
			closeIfUnused();
		}

		private synchronized void release() {
			// 最後のlease解放だけがretired snapshotのresource closeへ到達する。
			if (leaseCount <= 0) throw new IllegalStateException("Workspace lease released twice");
			leaseCount--;
			closeIfUnused();
		}

		private void closeIfUnused() {
			// query leaseの寿命だけに従い、取り込みstreamの寿命には触れない。
			if (!retired || leaseCount != 0 || closed) return;
			closed = true;
			pageCache.retainRuns(Set.of());
		}

		private long epoch() {
			return epoch;
		}

		private String name() {
			return name;
		}

		private Path root() {
			return root;
		}

		private RunScanner runScanner() {
			return runScanner;
		}

		private LodPageCache pageCache() {
			return pageCache;
		}

		private MetricsRepository repository() {
			return repository;
		}
	}
}
