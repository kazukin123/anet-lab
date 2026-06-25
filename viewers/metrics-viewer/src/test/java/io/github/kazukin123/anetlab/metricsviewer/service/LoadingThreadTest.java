package io.github.kazukin123.anetlab.metricsviewer.service;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsFileReader;
import io.github.kazukin123.anetlab.metricsviewer.infra.RunScanner;
import io.github.kazukin123.anetlab.metricsviewer.infra.model.MetricsFileBlock;

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

	@Test
	void loadsMetricsFromConfiguredRunsDir() throws Exception {
		final Path runDir = runsDir.resolve("runA");
		Files.createDirectories(runDir);
		Files.writeString(runDir.resolve("metrics.jsonl"),
				"{\"type\":\"scalar\",\"tag\":\"tagA\",\"step\":1,\"value\":2.0}\n",
				StandardCharsets.UTF_8);

		final RecordingRepository repository = new RecordingRepository(List.of());
		final LoadingThread loadingThread = new LoadingThread(
				new RunScanner(runsDir.toString()),
				repository,
				new MetricsFileReader());

		loadingThread.start();
		try {
			assertTrue(repository.awaitMerge());
		} finally {
			loadingThread.terminateAndWait(5000);
		}

		assertEquals(List.of("runA"), repository.mergedRunIds);
	}

	@Test
	void skipsMetricsFileWhenSizeMatchesLastReadPosition() throws Exception {
		final Path runDir = runsDir.resolve("runClean");
		Files.createDirectories(runDir);
		final Path metricsFile = runDir.resolve("metrics.jsonl");
		Files.writeString(metricsFile,
				"{\"type\":\"scalar\",\"tag\":\"tagA\",\"step\":1,\"value\":2.0}\n",
				StandardCharsets.UTF_8);

		final RecordingRepository repository = new RecordingRepository(List.of());
		repository.setLastReadPosition("runClean", Files.size(metricsFile));
		final CountingMetricsFileReader fileReader = new CountingMetricsFileReader();
		final LoadingThread loadingThread = new LoadingThread(
				new RunScanner(runsDir.toString()),
				repository,
				fileReader);

		loadingThread.start();
		try {
			assertTrue(repository.awaitLastReadPosition());
		} finally {
			loadingThread.terminateAndWait(5000);
		}

		assertEquals(0, fileReader.parseDiffCalls.get());
		assertTrue(repository.mergedRunIds.isEmpty());
	}

	@Test
	void skipsMetricsFileWhenCachedOffsetExceedsFileSize() throws Exception {
		final Path runDir = runsDir.resolve("runTruncated");
		Files.createDirectories(runDir);
		final Path metricsFile = runDir.resolve("metrics.jsonl");
		Files.writeString(metricsFile,
				"{\"type\":\"scalar\",\"tag\":\"tagA\",\"step\":1,\"value\":2.0}\n",
				StandardCharsets.UTF_8);

		final RecordingRepository repository = new RecordingRepository(List.of());
		repository.setLastReadPosition("runTruncated", Files.size(metricsFile) + 1);
		final CountingMetricsFileReader fileReader = new CountingMetricsFileReader();
		final LoadingThread loadingThread = new LoadingThread(
				new RunScanner(runsDir.toString()),
				repository,
				fileReader);

		loadingThread.start();
		try {
			assertTrue(repository.awaitLastReadPosition());
		} finally {
			loadingThread.terminateAndWait(5000);
		}

		assertEquals(0, fileReader.parseDiffCalls.get());
		assertTrue(repository.mergedRunIds.isEmpty());
	}

	private static class CountingMetricsFileReader extends MetricsFileReader {
		private final AtomicInteger parseDiffCalls = new AtomicInteger();

		@Override
		public MetricsFileBlock parseDiff(Path jsonlFile, long startOffset, int maxLines) throws IOException {
			parseDiffCalls.incrementAndGet();
			return new MetricsFileBlock(startOffset, startOffset, List.of(), 0L, true);
		}
	}

	private static class RecordingRepository extends MetricsRepository {
		private final List<String> loadedRunIds;
		private final CountDownLatch loadCacheCalled = new CountDownLatch(1);
		private final CountDownLatch mergeCalled = new CountDownLatch(1);
		private final CountDownLatch lastReadPositionCalled = new CountDownLatch(1);
		private final Map<String, Long> lastReadPositions = new ConcurrentHashMap<>();
		private final List<Path> savedRunDirs = new CopyOnWriteArrayList<>();
		private final List<String> savedRunIds = new CopyOnWriteArrayList<>();
		private final List<String> mergedRunIds = new CopyOnWriteArrayList<>();

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

		void setLastReadPosition(String runId, long lastReadPosition) {
			lastReadPositions.put(runId, lastReadPosition);
		}

		@Override
		public long getLastReadPosition(String runId) {
			lastReadPositionCalled.countDown();
			return lastReadPositions.getOrDefault(runId, 0L);
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

		@Override
		public void mergeMetrics(String runId, MetricsFileBlock block) {
			mergedRunIds.add(runId);
			mergeCalled.countDown();
		}

		boolean awaitLoadCache() throws InterruptedException {
			return loadCacheCalled.await(2, TimeUnit.SECONDS);
		}

		boolean awaitMerge() throws InterruptedException {
			return mergeCalled.await(2, TimeUnit.SECONDS);
		}

		boolean awaitLastReadPosition() throws InterruptedException {
			return lastReadPositionCalled.await(2, TimeUnit.SECONDS);
		}
	}
}
