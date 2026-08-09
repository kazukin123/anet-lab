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
	@ExtendWith(OutputCaptureExtension.class)
	void runtimeFailureIsLoggedAndNextCycleRuns(CapturedOutput output) throws Exception {
		final CountDownLatch secondCycleStarted = new CountDownLatch(1);
		final AtomicInteger cycleCount = new AtomicInteger();
		final IngestScheduler scheduler = mock(IngestScheduler.class);
		when(scheduler.runCycle()).thenAnswer(invocation -> {
			if (cycleCount.incrementAndGet() == 1) {
				throw new IllegalStateException("Injected cycle failure");
			}
			secondCycleStarted.countDown();
			return false;
		});
		final LoadingThread thread = new LoadingThread(scheduler, 1);

		thread.start();
		try {
			assertTrue(secondCycleStarted.await(2, TimeUnit.SECONDS));
		} finally {
			thread.terminateAndWait(2_000);
		}

		assertFalse(thread.isAlive());
		assertTrue(output.getAll().contains(
				"LoadingThread cycle failed; retrying in 1ms."));
		assertTrue(output.getAll().contains(
				"java.lang.IllegalStateException: Injected cycle failure"));
	}
}
