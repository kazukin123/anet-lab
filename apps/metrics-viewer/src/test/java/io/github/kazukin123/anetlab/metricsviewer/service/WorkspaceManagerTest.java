package io.github.kazukin123.anetlab.metricsviewer.service;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyBoolean;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.io.IOException;
import java.lang.reflect.Field;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.SQLException;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.zip.GZIPOutputStream;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.springframework.boot.test.system.CapturedOutput;
import org.springframework.boot.test.system.OutputCaptureExtension;

import io.github.kazukin123.anetlab.metricsviewer.config.MetricsViewerSettings;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsCacheDatabase;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsCacheDatabase.CachePreparation;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsCacheDatabase.ConnectionHandle;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsSource;

class WorkspaceManagerTest {

	@Test
	@ExtendWith(OutputCaptureExtension.class)
	void initialWorkspaceMustBeOneDirectChildName(CapturedOutput output) throws Exception {
		final Path root = Files.createTempDirectory("workspace-manager-validation-");
		for (String invalid : List.of(
				"", " ", ".", "..", "a/b", "a\\b", "C:relative", "a#b",
				root.resolve("absolute").toString())) {
			assertThrows(IllegalArgumentException.class,
					() -> new WorkspaceManager(root.toString(), invalid, database(), settings()),
					invalid);
		}

		final WorkspaceManager manager = new WorkspaceManager(
				root.toString(), "missing", database(), settings());
		try {
			assertEquals("missing", manager.currentName());
			assertEquals(List.of(), manager.listWorkspaceNames());
			assertEquals(WorkspaceManager.SwitchResult.NO_OP,
					manager.switchWorkspace("missing"));
			assertEquals(0L, manager.currentEpoch());
			assertFalse(Files.exists(root.resolve("missing")));
			assertTrue(output.getAll().contains("Initial workspace directory not found"));
		} finally {
			manager.shutdown();
		}
	}

	@Test
	void uncWorkspacesDirectoryIsRejectedAtStartup() {
		assertThrows(IllegalArgumentException.class,
				() -> new WorkspaceManager("\\\\server\\share", "ws", database(), settings()));
		assertThrows(IllegalArgumentException.class,
				() -> new WorkspaceManager("//server/share", "ws", database(), settings()));
	}

	@Test
	void concurrentSwitchesToSameTargetSwapExactlyOnce() throws Exception {
		final Path root = Files.createTempDirectory("workspace-manager-concurrent-switch-");
		Files.createDirectories(root.resolve("ws-a").resolve("runs"));
		Files.createDirectories(root.resolve("ws-b").resolve("runs"));
		final WorkspaceManager manager = new WorkspaceManager(
				root.toString(), "ws-a", database(), settings());
		final ExecutorService executor = Executors.newFixedThreadPool(6);
		final CountDownLatch ready = new CountDownLatch(6);
		final CountDownLatch start = new CountDownLatch(1);
		try {
			final List<Future<WorkspaceManager.SwitchResult>> results = new ArrayList<>();
			for (int i = 0; i < 6; i++) {
				results.add(executor.submit(() -> {
					ready.countDown();
					start.await(2, TimeUnit.SECONDS);
					return manager.switchWorkspace("ws-b");
				}));
			}
			assertTrue(ready.await(2, TimeUnit.SECONDS));
			start.countDown();

			int switched = 0;
			for (Future<WorkspaceManager.SwitchResult> result : results) {
				if (result.get(2, TimeUnit.SECONDS) == WorkspaceManager.SwitchResult.SWITCHED) switched++;
			}
			assertEquals(1, switched);
			assertEquals(1L, manager.currentEpoch());
			assertEquals("ws-b", manager.currentName());
		} finally {
			start.countDown();
			executor.shutdownNow();
			manager.shutdown();
		}
	}

	@Test
	void concurrentSwitchesToDifferentTargetsAreSerializedInArrivalOrder() throws Exception {
		final Path root = Files.createTempDirectory("workspace-manager-different-switches-");
		createRawRun(root, "ws-a", "same-run");
		Files.createDirectories(root.resolve("ws-b").resolve("runs"));
		Files.createDirectories(root.resolve("ws-c").resolve("runs"));
		final BlockingWriteDatabase database = new BlockingWriteDatabase();
		final WorkspaceManager manager = new WorkspaceManager(
				root.toString(), "ws-a", database, settings());
		final ExecutorService executor = Executors.newFixedThreadPool(3);
		final AtomicReference<Thread> firstSwitchThread = new AtomicReference<>();
		final AtomicReference<Thread> secondSwitchThread = new AtomicReference<>();
		try {
			final Future<Boolean> cycle = executor.submit(manager::runIngestCycle);
			assertTrue(database.awaitWrite());
			final Future<WorkspaceManager.SwitchResult> firstSwitch = executor.submit(() -> {
				firstSwitchThread.set(Thread.currentThread());
				return manager.switchWorkspace("ws-b");
			});
			assertTrue(awaitThreadState(firstSwitchThread, Thread.State.WAITING));
			final Future<WorkspaceManager.SwitchResult> secondSwitch = executor.submit(() -> {
				secondSwitchThread.set(Thread.currentThread());
				return manager.switchWorkspace("ws-c");
			});
			assertTrue(awaitThreadState(secondSwitchThread, Thread.State.WAITING));

			database.releaseWrite();
			assertEquals(WorkspaceManager.SwitchResult.SWITCHED,
					firstSwitch.get(2, TimeUnit.SECONDS));
			assertEquals(WorkspaceManager.SwitchResult.SWITCHED,
					secondSwitch.get(2, TimeUnit.SECONDS));
			cycle.get(2, TimeUnit.SECONDS);
			assertEquals("ws-c", manager.currentName());
			assertEquals(2L, manager.currentEpoch());
		} finally {
			database.releaseWrite();
			executor.shutdownNow();
			manager.shutdown();
		}
	}

	@Test
	void shutdownIsTerminalAndClosesResourcesAfterTheLastLease() throws Exception {
		final Path root = Files.createTempDirectory("workspace-manager-terminal-shutdown-");
		Files.createDirectories(root.resolve("ws-a").resolve("runs"));
		Files.createDirectories(root.resolve("ws-b").resolve("runs"));
		final CountingGzipInputSessions sessions = new CountingGzipInputSessions();
		final WorkspaceManager manager = new WorkspaceManager(
				root.toString(), "ws-a", database(), settings(), () -> sessions);
		final WorkspaceManager.Lease lease = manager.acquireLease();
		try {
			manager.shutdown();
			assertEquals(0, sessions.closeCount);
			assertTimeoutPreemptively(Duration.ofSeconds(1),
					() -> assertThrows(IllegalStateException.class, manager::acquireLease));
			assertThrows(IllegalStateException.class, () -> manager.switchWorkspace("ws-b"));
			assertEquals("ws-a", manager.currentName());
			assertEquals(0L, manager.currentEpoch());
		} finally {
			lease.close();
			manager.shutdown();
		}
		assertEquals(1, sessions.closeCount);
	}

	@Test
	void acquireLeaseRetriesWhenSnapshotRetiresBetweenReadAndAcquire() throws Exception {
		final Path root = Files.createTempDirectory("workspace-manager-lease-race-");
		Files.createDirectories(root.resolve("ws-a").resolve("runs"));
		Files.createDirectories(root.resolve("ws-b").resolve("runs"));
		final WorkspaceManager manager = new WorkspaceManager(
				root.toString(), "ws-a", database(), settings());
		// 本体へtest hookを増やさず、read直後・tryAcquire前の競合だけをwhite-boxで固定する。
		final Object oldSnapshot = currentSnapshot(manager);
		final ExecutorService executor = Executors.newSingleThreadExecutor();
		final AtomicReference<Thread> worker = new AtomicReference<>();
		final CountDownLatch started = new CountDownLatch(1);
		Future<WorkspaceManager.Lease> acquiring = null;
		try {
			synchronized (oldSnapshot) {
				acquiring = executor.submit(() -> {
					worker.set(Thread.currentThread());
					started.countDown();
					return manager.acquireLease();
				});
				assertTrue(started.await(2, TimeUnit.SECONDS));
				assertTrue(awaitThreadState(worker, Thread.State.BLOCKED));
				assertEquals(WorkspaceManager.SwitchResult.SWITCHED,
						manager.switchWorkspace("ws-b"));
			}

			try (WorkspaceManager.Lease lease = acquiring.get(2, TimeUnit.SECONDS)) {
				assertEquals("ws-b", lease.name());
				assertEquals(1L, lease.epoch());
			}
		} finally {
			if (acquiring != null) acquiring.cancel(true);
			executor.shutdownNow();
			manager.shutdown();
		}
	}

	@Test
	void returningToWorkspaceUsesTheProcessGlobalDatabaseLifecycleLock() throws Exception {
		final Path root = Files.createTempDirectory("workspace-manager-return-lock-");
		createRawRun(root, "ws-a", "same-run");
		Files.createDirectories(root.resolve("ws-b").resolve("runs"));
		final Path runA = root.resolve("ws-a").resolve("runs").resolve("same-run");
		final PrepareAwareDatabase database = new PrepareAwareDatabase();
		final MetricsSource source = MetricsSource.select(runA).orElseThrow();
		new MetricsIngestor(database).ingestBlock("same-run", runA, source);

		final WorkspaceManager manager = new WorkspaceManager(
				root.toString(), "ws-a", database, settings());
		final WorkspaceManager.Lease oldLease = manager.acquireLease();
		ConnectionHandle oldQuery = database.openRead(runA);
		final ExecutorService executor = Executors.newSingleThreadExecutor();
		WorkspaceManager.Lease returnedLease = null;
		Future<Boolean> ingest = null;
		try {
			assertEquals(WorkspaceManager.SwitchResult.SWITCHED,
					manager.switchWorkspace("ws-b"));
			assertEquals(WorkspaceManager.SwitchResult.SWITCHED,
					manager.switchWorkspace("ws-a"));
			returnedLease = manager.acquireLease();
			assertTrue(returnedLease.runScanner().getRunsDir().startsWith(root.resolve("ws-a")));

			database.armNextPrepare();
			ingest = executor.submit(manager::runIngestCycle);
			assertTrue(database.awaitPrepare());
			final Future<Boolean> blockedIngest = ingest;
			assertThrows(TimeoutException.class,
					() -> blockedIngest.get(150, TimeUnit.MILLISECONDS));

			oldQuery.close();
			oldQuery = null;
			ingest.get(2, TimeUnit.SECONDS);
		} finally {
			if (oldQuery != null) oldQuery.close();
			if (ingest != null) ingest.cancel(true);
			if (returnedLease != null) returnedLease.close();
			oldLease.close();
			executor.shutdownNow();
			manager.shutdown();
		}
	}

	@Test
	void switchingClosesOldGzipStreamSoWorkspaceCanBeMoved() throws Exception {
		final Path root = Files.createTempDirectory("workspace-manager-gzip-release-");
		final Path runA = root.resolve("ws-a").resolve("runs").resolve("same-run");
		Files.createDirectories(runA);
		final Path gzipFile = runA.resolve("metrics.jsonl.gz");
		try (GZIPOutputStream output = new GZIPOutputStream(Files.newOutputStream(gzipFile))) {
			output.write("{\"type\":\"scalar\",\"tag\":\"loss\",\"step\":0,\"value\":1}\n"
					.getBytes(StandardCharsets.UTF_8));
		}
		Files.createDirectories(root.resolve("ws-b").resolve("runs"));
		final GzipInputSessions oldSessions = new GzipInputSessions();
		final GzipInputSessions newSessions = new GzipInputSessions();
		final java.util.ArrayDeque<GzipInputSessions> sessions =
				new java.util.ArrayDeque<>(List.of(oldSessions, newSessions));
		final WorkspaceManager manager = new WorkspaceManager(
				root.toString(), "ws-a", database(), settings(), sessions::removeFirst);
		final MetricsSource source = MetricsSource.select(runA).orElseThrow();
		oldSessions.acquire(runA, source, UUID.randomUUID().toString());
		assertTrue(oldSessions.hasSession(runA, source));
		try {
			assertEquals(WorkspaceManager.SwitchResult.SWITCHED,
					manager.switchWorkspace("ws-b"));
			assertFalse(oldSessions.hasSession(runA, source));

			final Path moved = root.resolve("ws-a-moved");
			Files.move(root.resolve("ws-a"), moved);
			assertTrue(Files.isRegularFile(
					moved.resolve("runs").resolve("same-run").resolve("metrics.jsonl.gz")));
		} finally {
			manager.shutdown();
		}
	}

	@Test
	void switchWaitsForCurrentIngestCycleAndNextCycleUsesNewWorkspace() throws Exception {
		final Path root = Files.createTempDirectory("workspace-manager-gate-");
		createRawRun(root, "ws-a", "same-run");
		createRawRun(root, "ws-b", "same-run");

		final CountDownLatch entered = new CountDownLatch(1);
		final CountDownLatch release = new CountDownLatch(1);
		final List<Path> preparedRuns = new CopyOnWriteArrayList<>();
		final MetricsCacheDatabase database = database();
		when(database.prepare(any(Path.class), any(), anyBoolean())).thenAnswer(invocation -> {
			final Path runDir = invocation.getArgument(0, Path.class);
			preparedRuns.add(runDir);
			if (runDir.startsWith(root.resolve("ws-a")) && entered.getCount() > 0L) {
				entered.countDown();
				release.await(2, TimeUnit.SECONDS);
			}
			throw new IOException("Injected preparation failure");
		});
		final WorkspaceManager manager = new WorkspaceManager(
				root.toString(), "ws-a", database, settings());
		final ExecutorService executor = Executors.newFixedThreadPool(2);
		try {
			final Future<Boolean> cycle = executor.submit(manager::runIngestCycle);
			assertTrue(entered.await(2, TimeUnit.SECONDS));
			final Future<WorkspaceManager.SwitchResult> switching =
					executor.submit(() -> manager.switchWorkspace("ws-b"));
			assertFalse(switching.isDone());

			release.countDown();
			cycle.get(2, TimeUnit.SECONDS);
			assertEquals(WorkspaceManager.SwitchResult.SWITCHED,
					switching.get(2, TimeUnit.SECONDS));

			preparedRuns.clear();
			manager.runIngestCycle();
			assertFalse(preparedRuns.isEmpty());
			assertTrue(preparedRuns.stream().allMatch(path -> path.startsWith(root.resolve("ws-b"))));
		} finally {
			release.countDown();
			executor.shutdownNow();
			manager.shutdown();
		}
	}

	@Test
	void switchDoesNotWaitForRemainingBlocksInCurrentCycle() throws Exception {
		final Path root = Files.createTempDirectory("workspace-manager-block-gate-");
		createRawRun(root, "ws-a", "same-run");
		createRawRun(root, "ws-b", "same-run");

		final CountDownLatch firstBlockEntered = new CountDownLatch(1);
		final CountDownLatch releaseFirstBlock = new CountDownLatch(1);
		final CountDownLatch secondBlockEntered = new CountDownLatch(1);
		final CountDownLatch releaseSecondBlock = new CountDownLatch(1);
		final AtomicInteger oldWorkspaceAttempts = new AtomicInteger();
		final MetricsCacheDatabase database = database();
		when(database.prepare(any(Path.class), any(), anyBoolean())).thenAnswer(invocation -> {
			final Path runDir = invocation.getArgument(0, Path.class);
			if (runDir.startsWith(root.resolve("ws-a"))) {
				final int attempt = oldWorkspaceAttempts.incrementAndGet();
				if (attempt == 1) {
					firstBlockEntered.countDown();
					releaseFirstBlock.await(2, TimeUnit.SECONDS);
				} else if (attempt == 2) {
					secondBlockEntered.countDown();
					releaseSecondBlock.await(2, TimeUnit.SECONDS);
				}
			}
			throw new IOException("Injected preparation failure");
		});
		final WorkspaceManager manager = new WorkspaceManager(
				root.toString(), "ws-a", database, settings());
		final ExecutorService executor = Executors.newFixedThreadPool(2);
		try {
			final Future<Boolean> cycle = executor.submit(manager::runIngestCycle);
			assertTrue(firstBlockEntered.await(2, TimeUnit.SECONDS));
			final Future<WorkspaceManager.SwitchResult> switching =
					executor.submit(() -> manager.switchWorkspace("ws-b"));
			assertFalse(switching.isDone());

			releaseFirstBlock.countDown();
			assertEquals(WorkspaceManager.SwitchResult.SWITCHED,
					switching.get(2, TimeUnit.SECONDS));
			assertFalse(secondBlockEntered.await(150, TimeUnit.MILLISECONDS));
			assertTrue(cycle.get(2, TimeUnit.SECONDS));
			assertEquals("ws-b", manager.currentName());
		} finally {
			releaseFirstBlock.countDown();
			releaseSecondBlock.countDown();
			executor.shutdownNow();
			manager.shutdown();
		}
	}

	@Test
	void shutdownStopsAnActiveCycleAtTheNextBlockBoundary() throws Exception {
		final Path root = Files.createTempDirectory("workspace-manager-shutdown-cycle-");
		createRawRun(root, "ws-a", "same-run");
		final BlockingWriteDatabase database = new BlockingWriteDatabase();
		final WorkspaceManager manager = new WorkspaceManager(
				root.toString(), "ws-a", database, settings());
		final ExecutorService executor = Executors.newFixedThreadPool(2);
		try {
			final Future<Boolean> cycle = executor.submit(manager::runIngestCycle);
			assertTrue(database.awaitWrite());
			final Future<?> shuttingDown = executor.submit(manager::shutdown);
			assertFalse(shuttingDown.isDone());

			database.releaseWrite();
			shuttingDown.get(2, TimeUnit.SECONDS);
			assertTrue(cycle.get(2, TimeUnit.SECONDS));
			assertEquals(1, database.writeCount());
		} finally {
			database.releaseWrite();
			executor.shutdownNow();
			manager.shutdown();
		}
	}

	@Test
	void pendingSwitchYieldsTheCurrentBlockAtACompleteLine() throws Exception {
		final Path root = Files.createTempDirectory("workspace-manager-switch-yield-");
		final Path runA = root.resolve("ws-a").resolve("runs").resolve("same-run");
		Files.createDirectories(runA);
		final StringBuilder lines = new StringBuilder();
		for (int step = 0; step < 5; step++) {
			lines.append("{\"type\":\"scalar\",\"tag\":\"loss\",\"step\":")
					.append(step)
					.append(",\"value\":1.0}\n");
		}
		Files.writeString(runA.resolve("metrics.jsonl"), lines, StandardCharsets.UTF_8);
		Files.createDirectories(root.resolve("ws-b").resolve("runs"));

		final BlockingWriteDatabase database = new BlockingWriteDatabase();
		final WorkspaceManager manager = new WorkspaceManager(
				root.toString(), "ws-a", database, settings());
		final ExecutorService executor = Executors.newFixedThreadPool(2);
		final AtomicReference<Thread> switchThread = new AtomicReference<>();
		final CountDownLatch switchStarted = new CountDownLatch(1);
		try {
			final Future<Boolean> cycle = executor.submit(manager::runIngestCycle);
			assertTrue(database.awaitWrite());
			final Future<WorkspaceManager.SwitchResult> switching = executor.submit(() -> {
				switchThread.set(Thread.currentThread());
				switchStarted.countDown();
				return manager.switchWorkspace("ws-b");
			});
			assertTrue(switchStarted.await(2, TimeUnit.SECONDS));
			assertTrue(awaitThreadState(switchThread, Thread.State.WAITING));

			database.releaseWrite();
			assertEquals(WorkspaceManager.SwitchResult.SWITCHED,
					switching.get(2, TimeUnit.SECONDS));
			cycle.get(2, TimeUnit.SECONDS);

			try (ConnectionHandle handle = database.openRead(runA);
					java.sql.Statement statement = handle.connection().createStatement();
					java.sql.ResultSet result = statement.executeQuery(
							"SELECT COUNT(*) FROM scalars")) {
				result.next();
				assertEquals(1L, result.getLong(1));
			}
		} finally {
			database.releaseWrite();
			executor.shutdownNow();
			manager.shutdown();
		}
	}

	@Test
	void retiredSnapshotClosesGzipSessionsOnceAfterItsLastLease() throws Exception {
		final Path root = Files.createTempDirectory("workspace-manager-lease-");
		Files.createDirectories(root.resolve("ws-a").resolve("runs"));
		Files.createDirectories(root.resolve("ws-b").resolve("runs"));
		final CountingGzipInputSessions oldSessions = new CountingGzipInputSessions();
		final CountingGzipInputSessions newSessions = new CountingGzipInputSessions();
		final java.util.ArrayDeque<GzipInputSessions> sessions =
				new java.util.ArrayDeque<>(List.of(oldSessions, newSessions));
		final WorkspaceManager manager = new WorkspaceManager(
				root.toString(), "ws-a", database(), settings(), sessions::removeFirst);
		final WorkspaceManager.Lease oldLease = manager.acquireLease();
		try {
			assertEquals(WorkspaceManager.SwitchResult.SWITCHED,
					manager.switchWorkspace("ws-b"));
			assertTrue(oldLease.runScanner().getRunsDir().startsWith(root.resolve("ws-a")));
			try (WorkspaceManager.Lease newLease = manager.acquireLease()) {
				assertTrue(newLease.runScanner().getRunsDir().startsWith(root.resolve("ws-b")));
			}
			assertEquals(0, oldSessions.closeCount);

			oldLease.close();
			assertEquals(1, oldSessions.closeCount);
			oldLease.close();
			assertEquals(1, oldSessions.closeCount);
		} finally {
			oldLease.close();
			manager.shutdown();
		}
		assertEquals(1, newSessions.closeCount);
	}

	private static MetricsCacheDatabase database() {
		return mock(MetricsCacheDatabase.class);
	}

	private static MetricsViewerSettings settings() {
		final MetricsViewerSettings settings = mock(MetricsViewerSettings.class);
		when(settings.getCacheMemoryBytes()).thenReturn(0L);
		when(settings.getTargetPointsPerSeries()).thenReturn(8_000);
		when(settings.getMaxPointsPerRequest()).thenReturn(500_000);
		return settings;
	}

	private static Object currentSnapshot(WorkspaceManager manager) throws Exception {
		final Field field = WorkspaceManager.class.getDeclaredField("current");
		field.setAccessible(true);
		return ((AtomicReference<?>) field.get(manager)).get();
	}

	private static boolean awaitThreadState(
			AtomicReference<Thread> thread,
			Thread.State expected) throws Exception {
		final long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(2);
		while (System.nanoTime() < deadline) {
			if (thread.get() != null && thread.get().getState() == expected) return true;
			Thread.sleep(1L);
		}
		return false;
	}

	private static void createRawRun(Path root, String workspace, String runId) throws Exception {
		final Path runDir = root.resolve(workspace).resolve("runs").resolve(runId);
		Files.createDirectories(runDir);
		Files.writeString(
				runDir.resolve("metrics.jsonl"),
				"{\"type\":\"scalar\",\"tag\":\"loss\",\"step\":0,\"value\":1.0}\n",
				java.nio.charset.StandardCharsets.UTF_8);
	}

	private static final class CountingGzipInputSessions extends GzipInputSessions {
		private int closeCount;

		@Override
		synchronized void closeAll() {
			closeCount++;
			super.closeAll();
		}
	}

	private static final class PrepareAwareDatabase extends MetricsCacheDatabase {
		private final AtomicBoolean prepareArmed = new AtomicBoolean();
		private volatile CountDownLatch prepareEntered = new CountDownLatch(0);

		private void armNextPrepare() {
			prepareEntered = new CountDownLatch(1);
			prepareArmed.set(true);
		}

		private boolean awaitPrepare() throws InterruptedException {
			return prepareEntered.await(2, TimeUnit.SECONDS);
		}

		@Override
		public CachePreparation prepare(Path runDir, MetricsSource source, boolean activeGzipSession)
				throws IOException, SQLException {
			if (prepareArmed.compareAndSet(true, false)) prepareEntered.countDown();
			return super.prepare(runDir, source, activeGzipSession);
		}
	}

	private static final class BlockingWriteDatabase extends MetricsCacheDatabase {
		private final AtomicBoolean blockNextWrite = new AtomicBoolean(true);
		private final AtomicInteger writeCount = new AtomicInteger();
		private final CountDownLatch writeEntered = new CountDownLatch(1);
		private final CountDownLatch releaseWrite = new CountDownLatch(1);

		private boolean awaitWrite() throws InterruptedException {
			return writeEntered.await(2, TimeUnit.SECONDS);
		}

		private void releaseWrite() {
			releaseWrite.countDown();
		}

		private int writeCount() {
			return writeCount.get();
		}

		@Override
		public ConnectionHandle openWrite(Path runDir) throws SQLException {
			writeCount.incrementAndGet();
			if (blockNextWrite.compareAndSet(true, false)) {
				writeEntered.countDown();
				try {
					releaseWrite.await(2, TimeUnit.SECONDS);
				} catch (InterruptedException e) {
					Thread.currentThread().interrupt();
					throw new SQLException("Interrupted while waiting to open write connection", e);
				}
			}
			return super.openWrite(runDir);
		}
	}
}
