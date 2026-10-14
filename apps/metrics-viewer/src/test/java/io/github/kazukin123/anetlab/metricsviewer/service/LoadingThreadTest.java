package io.github.kazukin123.anetlab.metricsviewer.service;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.springframework.boot.test.system.CapturedOutput;
import org.springframework.boot.test.system.OutputCaptureExtension;

class LoadingThreadTest {

	@Test
	void selectedRunIsProbedWithinOneSecondEvenWhenAllWorkWasReady() throws Exception {
		final WorkspaceManager manager = mock(WorkspaceManager.class);
		final CountDownLatch cycles = new CountDownLatch(2);
		when(manager.priorityRunPaths()).thenReturn(java.util.Set.of(java.nio.file.Path.of("selected")));
		when(manager.runIngestCycle()).thenAnswer(call -> {
			cycles.countDown();
			return false;
		});
		final LoadingThread thread = new LoadingThread(manager, 5000);
		thread.start();
		try {
			assertTrue(cycles.await(2500, TimeUnit.MILLISECONDS));
		}
		finally {
			thread.terminateAndWait(2000);
		}
	}

	@Test
	@ExtendWith(OutputCaptureExtension.class)
	void unexpectedRuntimeFailureStopsWriterWithoutRetry(CapturedOutput output) throws Exception {
		final AtomicInteger cycleCount = new AtomicInteger();
		final WorkspaceManager workspaceManager = mock(WorkspaceManager.class);
		when(workspaceManager.runIngestCycle()).thenAnswer(invocation -> {
			cycleCount.incrementAndGet();
			throw new IllegalStateException("Injected cycle failure");
		});
		final LoadingThread thread = new LoadingThread(workspaceManager, 1);
		thread.start();
		try {
			thread.join(2_000);
			assertFalse(thread.isAlive());
			org.junit.jupiter.api.Assertions.assertEquals(1, cycleCount.get());
		}
		finally {
			thread.terminateAndWait(2_000);
		}
		assertTrue(output.getAll().contains("LoadingThread failed; writer stopped."));
	}
}
