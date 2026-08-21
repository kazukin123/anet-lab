package io.github.kazukin123.anetlab.metricsviewer.service;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;

import java.sql.Statement;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

import org.junit.jupiter.api.Test;

class MetricsQueryCoordinatorTest {

	@Test
	void aNewerQueryInTheSameChannelSupersedesTheRunningQuery() throws Exception {
		final MetricsQueryCoordinator coordinator = new MetricsQueryCoordinator(1);
		final QueryChannel channel = new QueryChannel("browser-tab");
		final CountDownLatch firstStarted = new CountDownLatch(1);
		final ExecutorService executor = Executors.newSingleThreadExecutor();
		try {
			final Future<String> first = executor.submit(() -> coordinator.run(channel, 1L, query -> {
				firstStarted.countDown();
				while (true) {
					query.checkpoint();
					Thread.onSpinWait();
				}
			}));
			assertEquals(true, firstStarted.await(2, TimeUnit.SECONDS));

			assertEquals("new", coordinator.run(channel, 2L, query -> "new"));
			final ExecutionException error = org.junit.jupiter.api.Assertions.assertThrows(
					ExecutionException.class,
					() -> first.get(2, TimeUnit.SECONDS));
			assertInstanceOf(QueryCancelledException.class, error.getCause());
		} finally {
			executor.shutdownNow();
		}
	}

	@Test
	void aNewerQueryInTheSameChannelWakesTheSupersededPermitWaiter() throws Exception {
		final MetricsQueryCoordinator coordinator = new MetricsQueryCoordinator(1);
		final CountDownLatch blockerStarted = new CountDownLatch(1);
		final CountDownLatch releaseBlocker = new CountDownLatch(1);
		final CountDownLatch waiterSubmitted = new CountDownLatch(1);
		final ExecutorService executor = Executors.newFixedThreadPool(3);
		try {
			final Future<String> blocker = executor.submit(() -> coordinator.run(
					new QueryChannel("other-tab"),
					0L,
					query -> {
						blockerStarted.countDown();
						try {
							if (!releaseBlocker.await(10, TimeUnit.SECONDS)) {
								throw new AssertionError("Timed out waiting to release blocker");
							}
						} catch (InterruptedException e) {
							Thread.currentThread().interrupt();
							throw new AssertionError("Blocker was interrupted", e);
						}
						return "blocker";
					}));
			assertEquals(true, blockerStarted.await(2, TimeUnit.SECONDS));

			final QueryChannel channel = new QueryChannel("browser-tab");
			final Future<String> waiting = executor.submit(() -> {
				waiterSubmitted.countDown();
				return coordinator.run(channel, 1L, query -> "old");
			});
			assertEquals(true, waiterSubmitted.await(2, TimeUnit.SECONDS));
			org.junit.jupiter.api.Assertions.assertThrows(
					java.util.concurrent.TimeoutException.class,
					() -> waiting.get(100, TimeUnit.MILLISECONDS));

			final Future<String> newest = executor.submit(
					() -> coordinator.run(channel, 2L, query -> "new"));
			final ExecutionException error = org.junit.jupiter.api.Assertions.assertThrows(
					ExecutionException.class,
					() -> waiting.get(2, TimeUnit.SECONDS));
			assertInstanceOf(QueryCancelledException.class, error.getCause());

			releaseBlocker.countDown();
			assertEquals("blocker", blocker.get(2, TimeUnit.SECONDS));
			assertEquals("new", newest.get(2, TimeUnit.SECONDS));
		} finally {
			releaseBlocker.countDown();
			executor.shutdownNow();
		}
	}

	@Test
	void permitWaitCancellationCannotLeaveAnInterruptOnTheRequestThread() throws Exception {
		final MetricsQueryCoordinator coordinator = new MetricsQueryCoordinator(1);
		final CountDownLatch blockerStarted = new CountDownLatch(1);
		final CountDownLatch releaseBlocker = new CountDownLatch(1);
		final CountDownLatch interruptStarted = new CountDownLatch(1);
		final CountDownLatch deliverInterrupt = new CountDownLatch(1);
		final CountDownLatch waiterFinished = new CountDownLatch(1);
		final AtomicReference<Throwable> waiterFailure = new AtomicReference<>();
		final AtomicBoolean interruptedAfterRun = new AtomicBoolean();
		final ExecutorService executor = Executors.newFixedThreadPool(2);
		final QueryChannel channel = new QueryChannel("browser-tab");
		final Thread waiter = new Thread(() -> {
			try {
				coordinator.run(channel, 1L, query -> "old");
			} catch (Throwable error) {
				waiterFailure.set(error);
			} finally {
				interruptedAfterRun.set(Thread.currentThread().isInterrupted());
				waiterFinished.countDown();
			}
		}, "controlled-permit-waiter") {
			@Override
			public void interrupt() {
				interruptStarted.countDown();
				await(deliverInterrupt);
				super.interrupt();
			}
		};
		try {
			final Future<String> blocker = executor.submit(() -> coordinator.run(
					new QueryChannel("other-tab"),
					0L,
					query -> {
						blockerStarted.countDown();
						await(releaseBlocker);
						return "blocker";
					}));
			assertTrue(blockerStarted.await(2, TimeUnit.SECONDS));

			waiter.start();
			awaitState(waiter, Thread.State.TIMED_WAITING);
			final Future<String> newest = executor.submit(
					() -> coordinator.run(channel, 2L, query -> "new"));
			assertTrue(interruptStarted.await(2, TimeUnit.SECONDS));

			// permit を得ても、Coordinator の interrupt 送出が完了するまではcleanupを通過できない。
			releaseBlocker.countDown();
			assertEquals("blocker", blocker.get(2, TimeUnit.SECONDS));
			assertFalse(waiterFinished.await(100, TimeUnit.MILLISECONDS));

			deliverInterrupt.countDown();
			assertTrue(waiterFinished.await(2, TimeUnit.SECONDS));
			assertInstanceOf(QueryCancelledException.class, waiterFailure.get());
			assertFalse(interruptedAfterRun.get());
			assertEquals("new", newest.get(2, TimeUnit.SECONDS));
		} finally {
			releaseBlocker.countDown();
			deliverInterrupt.countDown();
			waiter.join(2_000L);
			executor.shutdownNow();
		}
	}

	@Test
	void differentChannelsRunConcurrentlyWithoutCancellingEachOther() throws Exception {
		final MetricsQueryCoordinator coordinator = new MetricsQueryCoordinator(2);
		final CountDownLatch bothStarted = new CountDownLatch(2);
		final CountDownLatch release = new CountDownLatch(1);
		final ExecutorService executor = Executors.newFixedThreadPool(2);
		try {
			final Future<String> first = executor.submit(() -> coordinator.run(
					new QueryChannel("tab-a"),
					0L,
					execution -> {
						bothStarted.countDown();
						await(release);
						return "a";
					}));
			final Future<String> second = executor.submit(() -> coordinator.run(
					new QueryChannel("tab-b"),
					0L,
					execution -> {
						bothStarted.countDown();
						await(release);
						return "b";
					}));

			assertEquals(true, bothStarted.await(2, TimeUnit.SECONDS));
			release.countDown();
			assertEquals("a", first.get(2, TimeUnit.SECONDS));
			assertEquals("b", second.get(2, TimeUnit.SECONDS));
		} finally {
			release.countDown();
			executor.shutdownNow();
		}
	}

	@Test
	void aLateSequenceIsRejectedWithoutRunningItsWork() {
		final MetricsQueryCoordinator coordinator = new MetricsQueryCoordinator(1);
		final QueryChannel channel = new QueryChannel("browser-tab");
		assertEquals("new", coordinator.run(channel, 2L, execution -> "new"));

		org.junit.jupiter.api.Assertions.assertThrows(
				QueryCancelledException.class,
				() -> coordinator.run(channel, 1L, execution -> {
					throw new AssertionError("Late work must not execute");
				}));
	}

	@Test
	void lateCleanupDoesNotDetachTheNewerTicket() throws Exception {
		final MetricsQueryCoordinator coordinator = new MetricsQueryCoordinator(2);
		final QueryChannel channel = new QueryChannel("browser-tab");
		final CountDownLatch oldStarted = new CountDownLatch(1);
		final CountDownLatch releaseOld = new CountDownLatch(1);
		final CountDownLatch newStarted = new CountDownLatch(1);
		final CountDownLatch releaseNew = new CountDownLatch(1);
		final ExecutorService executor = Executors.newFixedThreadPool(2);
		try {
			final Future<String> old = executor.submit(() -> coordinator.run(
					channel,
					1L,
					execution -> {
						oldStarted.countDown();
						await(releaseOld);
						return "old";
					}));
			assertEquals(true, oldStarted.await(2, TimeUnit.SECONDS));

			final Future<String> newer = executor.submit(() -> coordinator.run(
					channel,
					2L,
					execution -> {
						newStarted.countDown();
						while (!await(releaseNew, 10, TimeUnit.MILLISECONDS)) {
							execution.checkpoint();
						}
						return "new";
					}));
			assertEquals(true, newStarted.await(2, TimeUnit.SECONDS));

			releaseOld.countDown();
			final ExecutionException oldError = org.junit.jupiter.api.Assertions.assertThrows(
					ExecutionException.class,
					() -> old.get(2, TimeUnit.SECONDS));
			assertInstanceOf(QueryCancelledException.class, oldError.getCause());

			assertEquals("latest", coordinator.run(channel, 3L, execution -> "latest"));
			releaseNew.countDown();
			final ExecutionException newerError = org.junit.jupiter.api.Assertions.assertThrows(
					ExecutionException.class,
					() -> newer.get(2, TimeUnit.SECONDS));
			assertInstanceOf(QueryCancelledException.class, newerError.getCause());
		} finally {
			releaseOld.countDown();
			releaseNew.countDown();
			executor.shutdownNow();
		}
	}

	@Test
	void anEvictedChannelCanExecuteOneLateQueryAsANewChannel() {
		final MetricsQueryCoordinator coordinator = new MetricsQueryCoordinator(1);
		final QueryChannel evicted = new QueryChannel("channel-0");
		assertEquals("initial", coordinator.run(evicted, 1L, query -> "initial"));
		for (int i = 1; i <= 64; i++) {
			final int expected = i;
			final QueryChannel channel = new QueryChannel("channel-" + i);
			assertEquals(expected, coordinator.<Integer>run(channel, 1L, query -> expected));
		}

		assertEquals("late", coordinator.run(evicted, 0L, query -> "late"));
	}

	@Test
	void workspaceCancellationCrossingEpochBindingCancelsTheQuery() throws Exception {
		final MetricsQueryCoordinator coordinator = new MetricsQueryCoordinator(1);
		final CountDownLatch leaseAcquired = new CountDownLatch(1);
		final CountDownLatch bindEpoch = new CountDownLatch(1);
		final ExecutorService executor = Executors.newSingleThreadExecutor();
		try {
			final Future<String> query = executor.submit(() -> coordinator.run(
					new QueryChannel("browser-tab"),
					0L,
					execution -> {
						leaseAcquired.countDown();
						try {
							if (!bindEpoch.await(2, TimeUnit.SECONDS)) {
								throw new AssertionError("Timed out waiting to bind workspace epoch");
							}
						} catch (InterruptedException e) {
							Thread.currentThread().interrupt();
							throw new AssertionError("Epoch binding was interrupted", e);
						}
						execution.bindWorkspace(7L);
						return "old-workspace";
					}));
			assertEquals(true, leaseAcquired.await(2, TimeUnit.SECONDS));

			coordinator.cancelWorkspace(7L);
			bindEpoch.countDown();
			final ExecutionException error = org.junit.jupiter.api.Assertions.assertThrows(
					ExecutionException.class,
					() -> query.get(2, TimeUnit.SECONDS));
			assertInstanceOf(QueryCancelledException.class, error.getCause());
		} finally {
			bindEpoch.countDown();
			executor.shutdownNow();
		}
	}

	@Test
	void cancelAllIsTerminalAndCancelsTheRunningQuery() throws Exception {
		final MetricsQueryCoordinator coordinator = new MetricsQueryCoordinator(1);
		final CountDownLatch started = new CountDownLatch(1);
		final ExecutorService executor = Executors.newSingleThreadExecutor();
		try {
			final Future<String> running = executor.submit(() -> coordinator.run(
					new QueryChannel("browser-tab"),
					0L,
					execution -> {
						started.countDown();
						while (true) {
							execution.checkpoint();
							Thread.onSpinWait();
						}
					}));
			assertEquals(true, started.await(2, TimeUnit.SECONDS));

			coordinator.cancelAll();

			final ExecutionException runningError = org.junit.jupiter.api.Assertions.assertThrows(
					ExecutionException.class,
					() -> running.get(2, TimeUnit.SECONDS));
			assertInstanceOf(QueryCancelledException.class, runningError.getCause());
			org.junit.jupiter.api.Assertions.assertThrows(
					QueryCancelledException.class,
					() -> coordinator.run(new QueryChannel("new-tab"), 0L, execution -> "new"));
		} finally {
			executor.shutdownNow();
		}
	}

	@Test
	void aStatementCannotBeRegisteredAfterWorkspaceCancellation() throws Exception {
		final MetricsQueryCoordinator coordinator = new MetricsQueryCoordinator(1);
		final Statement statement = mock(Statement.class);
		final CountDownLatch epochBound = new CountDownLatch(1);
		final CountDownLatch registerStatement = new CountDownLatch(1);
		final ExecutorService executor = Executors.newSingleThreadExecutor();
		try {
			final Future<String> running = executor.submit(() -> coordinator.run(
					new QueryChannel("browser-tab"),
					0L,
					execution -> {
						execution.bindWorkspace(7L);
						epochBound.countDown();
						try {
							if (!registerStatement.await(2, TimeUnit.SECONDS)) {
								throw new AssertionError("Timed out waiting to register statement");
							}
						} catch (InterruptedException e) {
							Thread.currentThread().interrupt();
							throw new AssertionError("Statement registration was interrupted", e);
						}
						execution.registerStatement(statement);
						return "old-workspace";
					}));
			assertEquals(true, epochBound.await(2, TimeUnit.SECONDS));

			coordinator.cancelWorkspace(7L);
			registerStatement.countDown();

			final ExecutionException error = org.junit.jupiter.api.Assertions.assertThrows(
					ExecutionException.class,
					() -> running.get(2, TimeUnit.SECONDS));
			assertInstanceOf(QueryCancelledException.class, error.getCause());
			verifyNoInteractions(statement);
		} finally {
			registerStatement.countDown();
			executor.shutdownNow();
		}
	}

	@Test
	void supersedingAQueryCancelsItsRegisteredStatement() throws Exception {
		final MetricsQueryCoordinator coordinator = new MetricsQueryCoordinator(1);
		final QueryChannel channel = new QueryChannel("browser-tab");
		final Statement statement = mock(Statement.class);
		final CountDownLatch statementRegistered = new CountDownLatch(1);
		final ExecutorService executor = Executors.newSingleThreadExecutor();
		try {
			final Future<String> running = executor.submit(() -> coordinator.run(
					channel,
					0L,
					execution -> {
						try (StatementRegistration ignored = execution.registerStatement(statement)) {
							statementRegistered.countDown();
							while (true) {
								execution.checkpoint();
								Thread.onSpinWait();
							}
						}
					}));
			assertEquals(true, statementRegistered.await(2, TimeUnit.SECONDS));

			assertEquals("new", coordinator.run(channel, 1L, execution -> "new"));

			final ExecutionException error = org.junit.jupiter.api.Assertions.assertThrows(
					ExecutionException.class,
					() -> running.get(2, TimeUnit.SECONDS));
			assertInstanceOf(QueryCancelledException.class, error.getCause());
			verify(statement).cancel();
		} finally {
			executor.shutdownNow();
		}
	}

	@Test
	void nestedStatementsCanBeRegisteredAndClosedInLifoOrder() {
		final MetricsQueryCoordinator coordinator = new MetricsQueryCoordinator(1);
		final Statement outer = mock(Statement.class);
		final Statement inner = mock(Statement.class);

		assertEquals("done", coordinator.run(
				new QueryChannel("browser-tab"),
				0L,
				execution -> {
					try (StatementRegistration ignoredOuter = execution.registerStatement(outer)) {
						try (StatementRegistration ignoredInner = execution.registerStatement(inner)) {
							execution.checkpoint();
						}
					}
					return "done";
				}));
		verifyNoInteractions(outer, inner);
	}

	@Test
	void nestedStatementsRejectOutOfOrderClose() {
		final MetricsQueryCoordinator coordinator = new MetricsQueryCoordinator(1);
		final Statement outer = mock(Statement.class);
		final Statement inner = mock(Statement.class);

		assertEquals("done", coordinator.run(
				new QueryChannel("browser-tab"),
				0L,
				execution -> {
					final StatementRegistration outerRegistration =
							execution.registerStatement(outer);
					final StatementRegistration innerRegistration =
							execution.registerStatement(inner);
					org.junit.jupiter.api.Assertions.assertThrows(
							IllegalStateException.class,
							outerRegistration::close);
					innerRegistration.close();
					outerRegistration.close();
					return "done";
				}));
	}

	@Test
	void supersedingNestedStatementsCancelsOnlyTheInnermostStatement() throws Exception {
		final MetricsQueryCoordinator coordinator = new MetricsQueryCoordinator(2);
		final QueryChannel channel = new QueryChannel("browser-tab");
		final Statement outer = mock(Statement.class);
		final Statement inner = mock(Statement.class);
		final CountDownLatch statementsRegistered = new CountDownLatch(1);
		final ExecutorService executor = Executors.newSingleThreadExecutor();
		try {
			final Future<String> running = executor.submit(() -> coordinator.run(
					channel,
					0L,
					execution -> {
						try (StatementRegistration ignoredOuter = execution.registerStatement(outer)) {
							try (StatementRegistration ignoredInner = execution.registerStatement(inner)) {
								statementsRegistered.countDown();
								while (true) {
									execution.checkpoint();
									Thread.onSpinWait();
								}
							}
						}
					}));
			assertTrue(statementsRegistered.await(2, TimeUnit.SECONDS));

			assertEquals("new", coordinator.run(channel, 1L, execution -> "new"));

			final ExecutionException error = org.junit.jupiter.api.Assertions.assertThrows(
					ExecutionException.class,
					() -> running.get(2, TimeUnit.SECONDS));
			assertInstanceOf(QueryCancelledException.class, error.getCause());
			verify(inner).cancel();
			verifyNoInteractions(outer);
		} finally {
			executor.shutdownNow();
		}
	}

	private static void await(CountDownLatch latch) {
		if (!await(latch, 2, TimeUnit.SECONDS)) {
			throw new AssertionError("Timed out waiting for test release");
		}
	}

	private static boolean await(CountDownLatch latch, long timeout, TimeUnit unit) {
		try {
			return latch.await(timeout, unit);
		} catch (InterruptedException e) {
			Thread.currentThread().interrupt();
			throw new AssertionError("Test worker was interrupted", e);
		}
	}

	private static void awaitState(Thread thread, Thread.State expected) {
		final long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(2);
		while (thread.getState() != expected) {
			if (System.nanoTime() >= deadline) {
				throw new AssertionError(
						"Timed out waiting for thread state " + expected + ", actual=" + thread.getState());
			}
			Thread.onSpinWait();
		}
	}
}
