package io.github.kazukin123.anetlab.metricsviewer.service;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.Base64;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Stream;

import org.junit.jupiter.api.Test;
import org.springframework.http.HttpStatus;

import io.github.kazukin123.anetlab.metricsviewer.config.MetricsViewerSettings;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsCacheDatabase;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsSource;
import io.github.kazukin123.anetlab.metricsviewer.view.model.GetMetricsRequest;
import io.github.kazukin123.anetlab.metricsviewer.view.model.MetricsSeriesRequest;
import io.github.kazukin123.anetlab.metricsviewer.view.model.MetricsSeriesResult;
import io.github.kazukin123.anetlab.metricsviewer.view.model.RawProjection;

class WorkspaceSnapshotIntegrationTest {

	@Test
	void switchingWorkspaceCancelsTheOldQueryAndTheNewWorkspaceQuerySucceeds() throws Exception {
		final Path root = Files.createTempDirectory("workspace-snapshot-query-");
		final Path runA = createRun(root, "ws-a", 1.0);
		final BlockingReadDatabase database = new BlockingReadDatabase();
		final MetricsSource sourceA = MetricsSource.select(runA).orElseThrow();
		new MetricsIngestor(database).ingestBlock("same-run", runA, sourceA);
		checkpoint(runA.resolve(MetricsCacheDatabase.DATABASE_FILENAME));

		final Path runB = root.resolve("ws-b").resolve("runs").resolve("same-run");
		copyTree(runA, runB);
		updateScalar(runB.resolve(MetricsCacheDatabase.DATABASE_FILENAME), 2.0);

		final MetricsViewerSettings settings = settings();
		final MetricsQueryCoordinator coordinator = new MetricsQueryCoordinator(settings);
		final WorkspaceManager manager = new WorkspaceManager(
				root.toString(), "ws-a", database, settings, coordinator, GzipInputSessions::new);
		final MetricsService service = new MetricsService(
				manager, mock(LoadingThread.class), settings, coordinator);
		final ExecutorService executor = Executors.newSingleThreadExecutor();
		database.blockNextRead(runA);
		try {
			final Future<MetricsSeriesResult> oldQuery = executor.submit(() -> service.getMetrics(
					request(), "old-browser-tab", "0").data().get(0));
			assertTrue(database.awaitBlockedRead());

			assertEquals(WorkspaceManager.SwitchResult.SWITCHED,
					manager.switchWorkspace("ws-b"));
			database.releaseBlockedRead();

			final ExecutionException oldError = org.junit.jupiter.api.Assertions.assertThrows(
					ExecutionException.class,
					() -> oldQuery.get(2, TimeUnit.SECONDS));
			final MetricsApiException superseded =
					(MetricsApiException) oldError.getCause();
			assertEquals(HttpStatus.CONFLICT, superseded.getStatus());
			assertEquals("superseded", ((java.util.Map<?, ?>) superseded.getBody()).get("code"));

			final MetricsSeriesResult newResult = service.getMetrics(
					request(), "new-browser-tab", "0").data().get(0);
			assertEquals(2.0f, firstValue(newResult));
		} finally {
			database.releaseBlockedRead();
			executor.shutdownNow();
			manager.shutdown();
		}
	}

	private static Path createRun(Path root, String workspace, double value) throws Exception {
		final Path runDir = root.resolve(workspace).resolve("runs").resolve("same-run");
		Files.createDirectories(runDir);
		Files.writeString(
				runDir.resolve("metrics.jsonl"),
				"{\"type\":\"scalar\",\"tag\":\"loss\",\"step\":0,\"value\":"
						+ value + "}\n",
				StandardCharsets.UTF_8);
		return runDir;
	}

	private static void copyTree(Path source, Path target) throws Exception {
		try (Stream<Path> paths = Files.walk(source)) {
			for (Path path : paths.toList()) {
				final Path destination = target.resolve(source.relativize(path));
				if (Files.isDirectory(path)) Files.createDirectories(destination);
				else Files.copy(path, destination, StandardCopyOption.COPY_ATTRIBUTES);
			}
		}
	}

	private static void checkpoint(Path database) throws Exception {
		try (Connection connection = open(database);
				Statement statement = connection.createStatement()) {
			statement.execute("PRAGMA wal_checkpoint(TRUNCATE)");
		}
	}

	private static void updateScalar(Path database, double value) throws Exception {
		try (Connection connection = open(database);
				Statement statement = connection.createStatement()) {
			statement.executeUpdate("UPDATE scalars SET value=" + value);
		}
	}

	private static Connection open(Path database) throws SQLException {
		return DriverManager.getConnection("jdbc:sqlite:" + database.toAbsolutePath().normalize());
	}

	private static MetricsViewerSettings settings() {
		return new MetricsViewerSettings(8_000, 500_000, 0, 2);
	}

	private static GetMetricsRequest request() {
		final MetricsSeriesRequest series = new MetricsSeriesRequest();
		series.setRunId("same-run");
		series.setTagKey("loss");
		series.setFromStep(0L);
		series.setToStep(0L);
		series.setMaxPoints(3);
		final GetMetricsRequest request = new GetMetricsRequest();
		request.setSeries(List.of(series));
		return request;
	}

	private static float firstValue(MetricsSeriesResult result) {
		final RawProjection projection = (RawProjection) result.getProjection();
		final byte[] bytes = Base64.getDecoder().decode(projection.values().get(0));
		return ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).getFloat();
	}

	private static final class BlockingReadDatabase extends MetricsCacheDatabase {
		private final AtomicBoolean blockNextRead = new AtomicBoolean();
		private final CountDownLatch readBlocked = new CountDownLatch(1);
		private final CountDownLatch releaseRead = new CountDownLatch(1);
		private Path blockedRun;

		private void blockNextRead(Path runDir) {
			blockedRun = runDir.toAbsolutePath().normalize();
			blockNextRead.set(true);
		}

		private boolean awaitBlockedRead() throws InterruptedException {
			return readBlocked.await(2, TimeUnit.SECONDS);
		}

		private void releaseBlockedRead() {
			releaseRead.countDown();
		}

		@Override
		public ConnectionHandle openRead(Path runDir) throws SQLException {
			if (runDir.toAbsolutePath().normalize().equals(blockedRun)
					&& blockNextRead.compareAndSet(true, false)) {
				readBlocked.countDown();
				try {
					if (!releaseRead.await(2, TimeUnit.SECONDS)) {
						throw new SQLException("Timed out waiting to release blocked read");
					}
				} catch (InterruptedException e) {
					Thread.currentThread().interrupt();
					throw new SQLException("Interrupted while waiting to release blocked read", e);
				}
			}
			return super.openRead(runDir);
		}
	}
}
