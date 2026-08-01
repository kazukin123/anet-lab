package io.github.kazukin123.anetlab.metricsviewer.service;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import io.github.kazukin123.anetlab.metricsviewer.config.MetricsViewerSettings;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsCacheDatabase;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsCacheDatabase.SourceMeta;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsSource;
import io.github.kazukin123.anetlab.metricsviewer.infra.RunScanner;
import io.github.kazukin123.anetlab.metricsviewer.view.model.MetricsSeriesRequest;
import io.github.kazukin123.anetlab.metricsviewer.view.model.MetricsSeriesResult;

class MetricsRepositorySnapshotIntegrationTest {

	@TempDir
	private Path tempDir;

	@Test
	void fullRebuildWaitsUntilPlanningAndProjectionFinishOnTheSameSnapshot()
			throws Exception {
		final Path runDir = tempDir.resolve("run-live");
		Files.createDirectories(runDir);
		final Path sourcePath = runDir.resolve("metrics.jsonl");
		Files.writeString(sourcePath, scalarLine(1, 1.0), StandardCharsets.UTF_8);

		final MetricsCacheDatabase database = new MetricsCacheDatabase();
		final MetricsSource initialSource = MetricsSource.select(runDir).orElseThrow();
		new MetricsIngestor(database).ingestBlock("run-live", runDir, initialSource);
		final String initialGeneration = generation(database, runDir);
		final MetricsViewerSettings settings = new MetricsViewerSettings(3, 100, 0, 1);
		final MetricsRepository repository = new MetricsRepository(
				new RunScanner(tempDir.toString()),
				database,
				settings,
				new MetricsRangeProjector(new LodPageCache(settings)));

		final CountDownLatch planningCompleted = new CountDownLatch(1);
		final CountDownLatch allowProjection = new CountDownLatch(1);
		final BlockingPlanningRequest request =
				new BlockingPlanningRequest(planningCompleted, allowProjection);
		final ExecutorService executor = Executors.newFixedThreadPool(2);
		try {
			final Future<List<MetricsSeriesResult>> query =
					executor.submit(() -> repository.query(List.of(request)));
			assertTrue(planningCompleted.await(2, TimeUnit.SECONDS));

			Files.writeString(
					sourcePath,
					scalarLine(1, 9.0) + scalarLine(2, 2.0),
					StandardCharsets.UTF_8);
			final MetricsSource replacementSource = MetricsSource.select(runDir).orElseThrow();
			final Future<MetricsCacheDatabase.CachePreparation> rebuild = executor.submit(
					() -> database.prepare(runDir, replacementSource, false));

			assertThrows(TimeoutException.class, () -> rebuild.get(1, TimeUnit.SECONDS));
			allowProjection.countDown();

			final MetricsSeriesResult result = query.get(2, TimeUnit.SECONDS).get(0);
			final MetricsCacheDatabase.CachePreparation rebuilt = rebuild.get(2, TimeUnit.SECONDS);
			assertEquals(initialGeneration, result.getGeneration().toString());
			assertNotEquals(initialGeneration, rebuilt.generation());
		} finally {
			allowProjection.countDown();
			executor.shutdownNow();
		}
	}

	private static String generation(MetricsCacheDatabase database, Path runDir) throws Exception {
		try (MetricsCacheDatabase.ConnectionHandle handle = database.openRead(runDir)) {
			return SourceMeta.read(handle.connection(), -1L).generation();
		}
	}

	private static String scalarLine(long step, double value) {
		return "{\"type\":\"scalar\",\"tag\":\"loss\",\"step\":"
				+ step + ",\"value\":" + value + "}\n";
	}

	private static final class BlockingPlanningRequest extends MetricsSeriesRequest {
		private final CountDownLatch planningCompleted;
		private final CountDownLatch allowProjection;

		private BlockingPlanningRequest(
				CountDownLatch planningCompleted,
				CountDownLatch allowProjection) {
			this.planningCompleted = planningCompleted;
			this.allowProjection = allowProjection;
			setRunId("run-live");
			setTagKey("loss");
			setFromStep(0L);
			setToStep(10L);
			setMaxPoints(3);
		}

		@Override
		public Integer getMaxPoints() {
			planningCompleted.countDown();
			try {
				if (!allowProjection.await(5, TimeUnit.SECONDS)) {
					throw new IllegalStateException("Timed out waiting to allow projection");
				}
			} catch (InterruptedException e) {
				Thread.currentThread().interrupt();
				throw new IllegalStateException("Interrupted while waiting to allow projection", e);
			}
			return super.getMaxPoints();
		}
	}
}
