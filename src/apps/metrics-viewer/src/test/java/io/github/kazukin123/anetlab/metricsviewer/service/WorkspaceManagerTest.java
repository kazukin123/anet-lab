package io.github.kazukin123.anetlab.metricsviewer.service;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

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
import java.util.function.Supplier;
import java.util.zip.GZIPOutputStream;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.springframework.boot.test.system.CapturedOutput;
import org.springframework.boot.test.system.OutputCaptureExtension;

import io.github.kazukin123.anetlab.metricsviewer.config.MetricsViewerSettings;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsCacheDatabase;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsCacheDatabase.ConnectionHandle;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsSource;

class WorkspaceManagerTest {

	@Test
	void latePriorityUpdateFromTheOldEpochCannotChangeTheNewWorkspace() throws Exception {
		final Path root = Files.createTempDirectory("priority-epoch-");
		createRawRun(root, "a", "same");
		createRawRun(root, "b", "same");
		final WorkspaceManager manager = workspaceManager(root.toString(), "a", new MetricsCacheDatabase(), settings());
		try {
			final long oldEpoch = manager.currentEpoch();
			manager.replacePriority(oldEpoch, java.util.Set.of(root.resolve("a/runs/same")));
			manager.switchWorkspace("b");
			final Path selected = root.resolve("b/runs/same").toAbsolutePath().normalize();
			manager.replacePriority(manager.currentEpoch(), java.util.Set.of(selected));
			manager.replacePriority(oldEpoch, java.util.Set.of(root.resolve("a/runs/same")));
			assertEquals(java.util.Set.of(selected), manager.priorityRunPaths());
		}
		finally {
			manager.shutdown();
		}
	}

	@Test
	void unopenedWorkspaceRebuildsOldCacheAndIsVisibleThroughExistingApi() throws Exception {
		final Path root = Files.createTempDirectory("workspace-background-rebuild-");
		Files.createDirectories(root.resolve("ws-a/runs"));
		createRawRun(root, "ws-b", "old-run");
		final Path run = root.resolve("ws-b/runs/old-run");
		try (java.sql.Connection c = java.sql.DriverManager
				.getConnection("jdbc:sqlite:" + run.resolve("metrics_cache.db"));
				java.sql.Statement s = c.createStatement()) {
			s.execute("PRAGMA user_version=1");
		}
		final MetricsViewerSettings settings = settings();
		final WorkspaceManager manager = workspaceManager(root.toString(), "ws-a", new MetricsCacheDatabase(),
				settings);
		try {
			manager.runIngestCycle();
			assertTrue(Files.size(run.resolve("metrics_cache.db")) > 4096);
			manager.switchWorkspace("ws-b");
			final MetricsService service = new MetricsService(manager, mock(LoadingThread.class), settings,
					new MetricsQueryCoordinator(1));
			final var runs = service.getRuns().getRuns();
			assertEquals(1, runs.size());
			assertEquals("ready", runs.get(0).getIngest().state());
			assertNotNull(runs.get(0).getGeneration());
			assertEquals(1L, runs.get(0).getTags().get(0).getStats().getCount());
			assertEquals(1.0, runs.get(0).getTags().get(0).getStats().getLastValue());
		}
		finally {
			manager.shutdown();
		}
	}

	@Test
	void malformedCurrentSourceMetaIsRebuiltRatherThanHeldAsAWriteFailure() throws Exception {
		final Path root = Files.createTempDirectory("malformed-source-meta-");
		createRawRun(root, "a", "cached");
		final Path run = root.resolve("a/runs/cached");
		final MetricsCacheDatabase database = new MetricsCacheDatabase();
		new MetricsIngestor(database).ingestBlock("cached", run, MetricsSource.select(run).orElseThrow());
		try (final var handle = database.openWrite(run); final var statement = handle.connection().createStatement()) {
			statement.execute("ALTER TABLE source_meta RENAME COLUMN k TO broken");
		}
		final WorkspaceManager manager = workspaceManager(root.toString(), "a", database, settings());
		try {
			manager.runIngestCycle();
			try (final var handle = database.openRead(run);
					final var statement = handle.connection().createStatement();
					final var rows = statement.executeQuery("SELECT v FROM source_meta WHERE k='state'")) {
				assertTrue(rows.next());
				assertEquals("ready", rows.getString(1));
			}
		}
		finally {
			manager.shutdown();
		}
	}

	@Test
	void finishingFourSmallRunsDoesNotIdleWhileAnotherBackgroundRunIsQueued() throws Exception {
		final Path root = Files.createTempDirectory("queued-background-");
		Files.createDirectories(root.resolve("display/runs"));
		for (int index = 0; index < 5; index++)
			createRawRun(root, "background" + index, "small");
		final WorkspaceManager manager = workspaceManager(root.toString(), "display", new MetricsCacheDatabase(),
				settings());
		try {
			assertTrue(manager.runIngestCycle());
			assertFalse(manager.runIngestCycle());
			for (int index = 0; index < 5; index++)
				assertTrue(Files.isRegularFile(root.resolve("background" + index + "/runs/small/metrics_cache.db")));
		}
		finally {
			manager.shutdown();
		}
	}

	@Test
	void identicalPriorityPathsDoNotRequestAnotherScanOrWake() throws Exception {
		final Path root = Files.createTempDirectory("priority-unchanged-");
		createRawRun(root, "a", "selected");
		final WorkspaceManager manager = workspaceManager(root.toString(), "a", new MetricsCacheDatabase(), settings());
		final AtomicInteger wakes = new AtomicInteger();
		manager.setIngestWakeup(wakes::incrementAndGet);
		try {
			final Path selected = root.resolve("a/runs/selected");
			manager.replacePriority(manager.currentEpoch(), java.util.Set.of(selected));
			final int first = wakes.get();
			manager.replacePriority(manager.currentEpoch(), java.util.Set.of(selected.resolve(".")));
			assertEquals(first, wakes.get());
		}
		finally {
			manager.shutdown();
		}
	}

	@Test
	void unreadableWorkspaceListingDoesNotPreventInspectingOtherWorkspaces() throws Exception {
		final Path root = Files.createTempDirectory("workspace-listing-failure-");
		Files.createDirectories(root.resolve("bad/config"));
		Files.writeString(root.resolve("bad/runs"), "not a directory");
		createRawRun(root, "good", "healthy");
		final WorkspaceManager manager = workspaceManager(root.toString(), "good", new MetricsCacheDatabase(),
				settings());
		try {
			manager.runIngestCycle();
			assertTrue(Files.isRegularFile(root.resolve("good/runs/healthy/metrics_cache.db")));
		}
		finally {
			manager.shutdown();
		}
	}

	@Test
	void readyCommitDoesNotRequestAnImmediateFollowUpCycle() throws Exception {
		final Path root = Files.createTempDirectory("workspace-manager-ready-cycle-");
		createRawRun(root, "ws-a", "ready-run");
		final WorkspaceManager manager = workspaceManager(
				root.toString(), "ws-a", new MetricsCacheDatabase(), settings());
		try {
			assertFalse(manager.runIngestCycle());
		} finally {
			manager.shutdown();
		}
	}

	@Test
	void switchedWorkspaceCancelsQueriesBoundToThePreviousEpoch() throws Exception {
		final Path root = Files.createTempDirectory("workspace-manager-cancel-query-");
		Files.createDirectories(root.resolve("ws-a").resolve("runs"));
		Files.createDirectories(root.resolve("ws-b").resolve("runs"));
		final MetricsViewerSettings settings = settings();
		final MetricsQueryCoordinator coordinator = mock(MetricsQueryCoordinator.class);
		final WorkspaceManager manager = new WorkspaceManager(
				root.toString(), "ws-a", database(), settings,
				coordinator, GzipInputSessions::new);
		try {
			assertEquals(WorkspaceManager.SwitchResult.SWITCHED,
					manager.switchWorkspace("ws-b"));
			assertEquals(1L, manager.currentEpoch());
			verify(coordinator).cancelWorkspace(0L);
		} finally {
			manager.shutdown();
		}
	}

	@Test
	void noOpAndUnknownSwitchesDoNotCancelCurrentWorkspaceQueries() throws Exception {
		final Path root = Files.createTempDirectory("workspace-manager-no-cancel-");
		Files.createDirectories(root.resolve("ws-a").resolve("runs"));
		final MetricsViewerSettings settings = settings();
		final MetricsQueryCoordinator coordinator = new MetricsQueryCoordinator(1);
		final WorkspaceManager manager = new WorkspaceManager(
				root.toString(), "ws-a", database(), settings,
				coordinator, GzipInputSessions::new);
		final CountDownLatch started = new CountDownLatch(1);
		final CountDownLatch release = new CountDownLatch(1);
		final ExecutorService executor = Executors.newSingleThreadExecutor();
		try {
			final Future<String> running = executor.submit(() -> coordinator.run(
					new QueryChannel("browser-tab"),
					0L,
					execution -> {
						execution.bindWorkspace(manager.currentEpoch());
						started.countDown();
						try {
							if (!release.await(2, TimeUnit.SECONDS)) {
								throw new AssertionError("Timed out waiting to release query");
							}
						} catch (InterruptedException e) {
							Thread.currentThread().interrupt();
							throw new AssertionError("Query was interrupted", e);
						}
						execution.checkpoint();
						return "current";
					}));
			assertTrue(started.await(2, TimeUnit.SECONDS));

			assertEquals(WorkspaceManager.SwitchResult.NO_OP,
					manager.switchWorkspace("ws-a"));
			assertEquals(WorkspaceManager.SwitchResult.UNKNOWN,
					manager.switchWorkspace("missing"));
			release.countDown();

			assertEquals("current", running.get(2, TimeUnit.SECONDS));
		} finally {
			release.countDown();
			executor.shutdownNow();
			manager.shutdown();
		}
	}

	@Test
	@ExtendWith(OutputCaptureExtension.class)
	void initialWorkspaceMustBeOneDirectChildName(CapturedOutput output) throws Exception {
		final Path root = Files.createTempDirectory("workspace-manager-validation-");
		for (String invalid : List.of(
				"", " ", ".", "..", "a/b", "a\\b", "C:relative", "a#b",
				root.resolve("absolute").toString())) {
			assertThrows(IllegalArgumentException.class,
					() -> workspaceManager(root.toString(), invalid, database(), settings()),
					invalid);
		}

		final WorkspaceManager manager = workspaceManager(
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
				() -> workspaceManager("\\\\server\\share", "ws", database(), settings()));
		assertThrows(IllegalArgumentException.class,
				() -> workspaceManager("//server/share", "ws", database(), settings()));
	}

	@Test
	void concurrentSwitchesToSameTargetSwapExactlyOnce() throws Exception {
		final Path root = Files.createTempDirectory("workspace-manager-concurrent-switch-");
		Files.createDirectories(root.resolve("ws-a").resolve("runs"));
		Files.createDirectories(root.resolve("ws-b").resolve("runs"));
		final WorkspaceManager manager = workspaceManager(
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
		final WorkspaceManager manager = workspaceManager(
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
	void shutdownClosesIngestResourcesIndependentlyOfQueryLeases() throws Exception {
		final Path root = Files.createTempDirectory("workspace-manager-terminal-shutdown-");
		Files.createDirectories(root.resolve("ws-a").resolve("runs"));
		Files.createDirectories(root.resolve("ws-b").resolve("runs"));
		final CountingGzipInputSessions sessions = new CountingGzipInputSessions();
		final WorkspaceManager manager = workspaceManager(
				root.toString(), "ws-a", database(), settings(), () -> sessions);
		final WorkspaceManager.Lease lease = manager.acquireLease();
		try {
			manager.shutdown();
			assertEquals(1, sessions.closeCount);
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
		final WorkspaceManager manager = workspaceManager(
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

		final WorkspaceManager manager = workspaceManager(
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
	void workspaceRoundTripPreservesGenerationAndBufferedGzipProgress() throws Exception {
		final Path root = Files.createTempDirectory("workspace-manager-gzip-continue-");
		final Path run = root.resolve("ws-a/runs/same-run");
		Files.createDirectories(run);
		StringBuilder lines = new StringBuilder();
		for (int i = 0; i < 10; i++)
			lines.append("{\"type\":\"scalar\",\"tag\":\"loss\",\"step\":").append(i).append(",\"value\":1}\n");
		try (var gzip = new GZIPOutputStream(Files.newOutputStream(run.resolve("metrics.jsonl.gz")))) {
			gzip.write(lines.toString().getBytes(StandardCharsets.UTF_8));
		}
		Files.createDirectories(root.resolve("ws-b/runs"));
		final GzipInputSessions sessions = new GzipInputSessions();
		final MetricsCacheDatabase database = new MetricsCacheDatabase();
		final WorkspaceManager manager = new WorkspaceManager(root.toString(), "ws-a", database, settings(),
				new MetricsQueryCoordinator(1), () -> sessions, 1);
		try {
			manager.runIngestCycle();
			String generation;
			try (var h = database.openRead(run)) {
				generation = MetricsCacheDatabase.SourceMeta.read(h.connection(), -1).generation();
			}
			assertTrue(sessions.hasSession(run, MetricsSource.select(run).orElseThrow()));
			manager.switchWorkspace("ws-b");
			assertTrue(sessions.hasSession(run, MetricsSource.select(run).orElseThrow()));
			manager.runIngestCycle();
			try (var h = database.openRead(run);
					var st = h.connection().createStatement();
					var rs = st.executeQuery("SELECT COUNT(*) FROM scalars")) {
				rs.next();
				assertEquals(8, rs.getLong(1));
				assertEquals(generation, MetricsCacheDatabase.SourceMeta.read(h.connection(), -1).generation());
			}
			manager.switchWorkspace("ws-a");
			manager.runIngestCycle();
			try (var h = database.openRead(run);
					var st = h.connection().createStatement();
					var rs = st.executeQuery("SELECT COUNT(*) FROM scalars")) {
				rs.next();
				assertEquals(10, rs.getLong(1));
				assertEquals(generation, MetricsCacheDatabase.SourceMeta.read(h.connection(), -1).generation());
			}
			assertFalse(sessions.hasSession(run, MetricsSource.select(run).orElseThrow()));
		}
		finally {
			manager.shutdown();
		}
	}

	@Test
	void switchWaitsForCurrentBlockAndNewPriorityCanReactivateChangedSource() throws Exception {
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
		final WorkspaceManager manager = workspaceManager(
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
			Files.writeString(root.resolve("ws-b/runs/same-run/metrics.jsonl"), "{}\n{}\n");
			manager.replacePriority(manager.currentEpoch(), java.util.Set.of(root.resolve("ws-b/runs/same-run")));
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
		final WorkspaceManager manager = workspaceManager(
				root.toString(), "ws-a", database, settings());
		final ExecutorService executor = Executors.newFixedThreadPool(2);
		final AtomicReference<Thread> switchThread = new AtomicReference<>();
		final CountDownLatch switchStarted = new CountDownLatch(1);
		try {
			final Future<Boolean> cycle = executor.submit(manager::runIngestCycle);
			assertTrue(firstBlockEntered.await(2, TimeUnit.SECONDS));
			final Future<WorkspaceManager.SwitchResult> switching = executor.submit(() -> {
				switchThread.set(Thread.currentThread());
				switchStarted.countDown();
				return manager.switchWorkspace("ws-b");
			});
			assertTrue(switchStarted.await(2, TimeUnit.SECONDS));
			assertTrue(awaitThreadState(switchThread, Thread.State.WAITING));
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
		final WorkspaceManager manager = workspaceManager(
				root.toString(), "ws-a", database, settings());
		final ExecutorService executor = Executors.newFixedThreadPool(2);
		try {
			final Future<Boolean> cycle = executor.submit(manager::runIngestCycle);
			assertTrue(database.awaitWrite());
			final Future<?> shuttingDown = executor.submit(manager::shutdown);
			assertFalse(shuttingDown.isDone());

			database.releaseWrite();
			shuttingDown.get(2, TimeUnit.SECONDS);
			assertFalse(cycle.get(2, TimeUnit.SECONDS));
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
		final WorkspaceManager manager = workspaceManager(
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
	void retiringQuerySnapshotsDoesNotCloseProcessIngestSessions() throws Exception {
		final Path root = Files.createTempDirectory("workspace-manager-process-resource-");
		Files.createDirectories(root.resolve("ws-a/runs"));
		Files.createDirectories(root.resolve("ws-b/runs"));
		final CountingGzipInputSessions sessions = new CountingGzipInputSessions();
		final WorkspaceManager manager = workspaceManager(root.toString(), "ws-a", database(), settings(),
				() -> sessions);
		final WorkspaceManager.Lease old = manager.acquireLease();
		try {
			manager.switchWorkspace("ws-b");
			assertTrue(old.runScanner().getRunsDir().startsWith(root.resolve("ws-a")));
			assertEquals(0, sessions.closeCount);
			old.close();
			old.close();
			assertEquals(0, sessions.closeCount);
		}
		finally {
			old.close();
			manager.shutdown();
		}
		assertEquals(1, sessions.closeCount);
	}

	private static MetricsCacheDatabase database() {
		return mock(MetricsCacheDatabase.class);
	}

	private static WorkspaceManager workspaceManager(
			String workspacesDirPath,
			String initialWorkspace,
			MetricsCacheDatabase database,
			MetricsViewerSettings settings) {
		return workspaceManager(
				workspacesDirPath,
				initialWorkspace,
				database,
				settings,
				GzipInputSessions::new);
	}

	private static WorkspaceManager workspaceManager(
			String workspacesDirPath,
			String initialWorkspace,
			MetricsCacheDatabase database,
			MetricsViewerSettings settings,
			Supplier<GzipInputSessions> gzipSessionsFactory) {
		return new WorkspaceManager(
				workspacesDirPath,
				initialWorkspace,
				database,
				settings,
				new MetricsQueryCoordinator(settings),
				gzipSessionsFactory);
	}

	private static MetricsViewerSettings settings() {
		final MetricsViewerSettings settings = mock(MetricsViewerSettings.class);
		when(settings.getCacheMemoryBytes()).thenReturn(0L);
		when(settings.getMaxConcurrentQueries()).thenReturn(1);
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
