package io.github.kazukin123.anetlab.metricsviewer.service;

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
	private final Supplier<GzipInputSessions> gzipSessionsFactory;
	private final AtomicReference<WorkspaceSnapshot> current;
	private final ReentrantLock ingestGate = new ReentrantLock(true);
	private final AtomicInteger pendingSwitchRequests = new AtomicInteger();
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
		this.workspacesDir = normalizeWorkspacesDir(workspacesDirPath);
		validateInitialWorkspace(initialWorkspace, this.workspacesDir);
		this.database = database;
		this.settings = settings;
		this.queryCoordinator = queryCoordinator;
		this.gzipSessionsFactory = gzipSessionsFactory;
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
				previous.retire();
				cancelledEpoch = previous.epoch();
				result = SwitchResult.SWITCHED;
			}
		} finally {
			pendingSwitchRequests.decrementAndGet();
			ingestGate.unlock();
		}
		if (result == SwitchResult.SWITCHED) queryCoordinator.cancelWorkspace(cancelledEpoch);
		return result;
	}

	public boolean runIngestCycle() {
		try (Lease lease = acquireLease()) {
			boolean didWork = false;
			for (int slot = 0; slot < 4; slot++) {
				ingestGate.lock();
				try {
					// shutdown開始後は現在blockより先へ進まず、取得済みleaseの解放へ移る。
					if (shutdown) return didWork;
					// 切替済みなら旧snapshotへ残りblockを割り当てず、次cycleを即時実行させる。
					if (current.get().epoch() != lease.epoch()) return true;
					didWork |= lease.ingestScheduler().runNextBlock();
				} finally {
					ingestGate.unlock();
				}
			}
			return didWork;
		}
	}

	@PreDestroy
	public void shutdown() {
		ingestGate.lock();
		try {
			if (shutdown) return;
			// terminal状態を先に公開し、新規leaseと切替を止めてからcurrentをretireする。
			shutdown = true;
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
		final RunWarningRegistry warningRegistry = new RunWarningRegistry();
		final GzipInputSessions gzipSessions = gzipSessionsFactory.get();
		final LodIngestWriter lodWriter = new LodIngestWriter();
		final MetricsIngestor ingestor = new MetricsIngestor(
				database,
				gzipSessions,
				lodWriter,
				warningRegistry,
				MetricsIngestor.MAX_BLOCK_LINES,
				() -> pendingSwitchRequests.get() > 0);
		final IngestScheduler scheduler = new IngestScheduler(
				runScanner, ingestor, gzipSessions, warningRegistry);
		final MetricsRangeProjector projector = new MetricsRangeProjector(pageCache);
		final MetricsRepository repository = new MetricsRepository(
				runScanner, database, settings, projector);
		return new WorkspaceSnapshot(
				epoch, name, root, runScanner, pageCache, gzipSessions, scheduler, repository);
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

		public IngestScheduler ingestScheduler() {
			return snapshot.ingestScheduler();
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
		private final GzipInputSessions gzipSessions;
		private final IngestScheduler ingestScheduler;
		private final MetricsRepository repository;
		private int leaseCount;
		private boolean retired;
		private boolean closed;

		private WorkspaceSnapshot(
				long epoch,
				String name,
				Path root,
				RunScanner runScanner,
				LodPageCache pageCache,
				GzipInputSessions gzipSessions,
				IngestScheduler ingestScheduler,
				MetricsRepository repository) {
			this.epoch = epoch;
			this.name = name;
			this.root = root;
			this.runScanner = runScanner;
			this.pageCache = pageCache;
			this.gzipSessions = gzipSessions;
			this.ingestScheduler = ingestScheduler;
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
			// switch/shutdownとreleaseのどちらが最後でもcloseAllは一度だけ実行する。
			if (!retired || leaseCount != 0 || closed) return;
			closed = true;
			gzipSessions.closeAll();
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

		private IngestScheduler ingestScheduler() {
			return ingestScheduler;
		}

		private MetricsRepository repository() {
			return repository;
		}
	}
}
