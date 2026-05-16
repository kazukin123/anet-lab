package io.github.kazukin123.anetlab.metricsviewer.service;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsFileReader;
import io.github.kazukin123.anetlab.metricsviewer.infra.RunScanner;

class LoadingThreadTest {

	@TempDir
	Path runsDir;

	@Test
	void terminateAndWaitSavesLoadedCachesOnShutdown() throws Exception {
		final RecordingRepository repository = new RecordingRepository(List.of("run1", "run2"));
		final LoadingThread loadingThread = new LoadingThread(
				new RunScanner(runsDir.toString()),
				repository,
				new MetricsFileReader());

		loadingThread.start();
		assertTrue(repository.awaitLoadCache());

		loadingThread.terminateAndWait(5000);

		assertFalse(loadingThread.isAlive());
		assertEquals(List.of(
				runsDir.resolve("run1").toAbsolutePath().normalize(),
				runsDir.resolve("run2").toAbsolutePath().normalize()),
				repository.savedRunDirs);
		assertEquals(List.of("run1", "run2"), repository.savedRunIds);
	}

	private static class RecordingRepository extends MetricsRepository {
		private final List<String> loadedRunIds;
		private final CountDownLatch loadCacheCalled = new CountDownLatch(1);
		private final List<Path> savedRunDirs = new CopyOnWriteArrayList<>();
		private final List<String> savedRunIds = new CopyOnWriteArrayList<>();

		RecordingRepository(List<String> loadedRunIds) {
			this.loadedRunIds = loadedRunIds;
		}

		@Override
		public void loadCache(Path runsDir) {
			loadCacheCalled.countDown();
		}

		@Override
		public void loadCacheForRun(Path runDir) {
		}

		@Override
		public List<String> listAllRunIds() {
			return loadedRunIds;
		}

		@Override
		public void saveCache(Path runDir, String runId) {
			savedRunDirs.add(runDir.toAbsolutePath().normalize());
			savedRunIds.add(runId);
		}

		boolean awaitLoadCache() throws InterruptedException {
			return loadCacheCalled.await(2, TimeUnit.SECONDS);
		}
	}
}
