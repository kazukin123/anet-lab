package io.github.kazukin123.anetlab.metricsviewer.service;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.Mockito.inOrder;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InOrder;
import org.springframework.boot.test.system.CapturedOutput;
import org.springframework.boot.test.system.OutputCaptureExtension;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.test.util.ReflectionTestUtils;

import io.github.kazukin123.anetlab.metricsviewer.config.MetricsViewerSettings;
import io.github.kazukin123.anetlab.metricsviewer.view.MetricsViewerController;
import io.github.kazukin123.anetlab.metricsviewer.view.model.GetMetricsRequest;
import io.github.kazukin123.anetlab.metricsviewer.view.model.MetricsSeriesRequest;

class MetricsQueryConcurrencyTest {

	@Test
	void shutdownCancelsQueriesBeforeStoppingLoadingAndClosingTheWorkspace() {
		final WorkspaceManager workspaceManager = mock(WorkspaceManager.class);
		final LoadingThread loadingThread = mock(LoadingThread.class);
		final MetricsQueryCoordinator coordinator = mock(MetricsQueryCoordinator.class);
		final MetricsService service = new MetricsService(
				workspaceManager,
				loadingThread,
				mock(MetricsViewerSettings.class),
				coordinator);

		ReflectionTestUtils.invokeMethod(service, "shutdown");

		final InOrder order = inOrder(coordinator, loadingThread, workspaceManager);
		order.verify(coordinator).cancelAll();
		order.verify(loadingThread).terminateAndWait(30_000L);
		order.verify(workspaceManager).shutdown();
	}

	@Test
	@ExtendWith(OutputCaptureExtension.class)
	void aQueryThatCannotAcquireASlotReturnsAndLogsTheRetryContract(CapturedOutput output)
			throws Exception {
		final CountDownLatch enteredRepository = new CountDownLatch(1);
		final CountDownLatch releaseRepository = new CountDownLatch(1);
		final MetricsRepository repository = mock(MetricsRepository.class);
		when(repository.query(anyList(), any(QueryExecution.class))).thenAnswer(ignored -> {
			enteredRepository.countDown();
			releaseRepository.await(10, TimeUnit.SECONDS);
			return List.of();
		});
		final MetricsViewerSettings settings = mock(MetricsViewerSettings.class);
		when(settings.getMaxConcurrentQueries()).thenReturn(1);
		when(settings.getTargetPointsPerSeries()).thenReturn(4000);
		when(settings.getMaxPointsPerRequest()).thenReturn(500_000);
		final WorkspaceManager workspaceManager = mock(WorkspaceManager.class);
		final WorkspaceManager.Lease lease = mock(WorkspaceManager.Lease.class);
		when(workspaceManager.acquireLease()).thenReturn(lease);
		when(lease.repository()).thenReturn(repository);
		final MetricsQueryCoordinator coordinator = new MetricsQueryCoordinator(settings);
		final MetricsService service = new MetricsService(
				workspaceManager,
				mock(LoadingThread.class),
				settings,
				coordinator);
		final MetricsViewerController controller = new MetricsViewerController(service);

		final ExecutorService executor = Executors.newFixedThreadPool(2);
		try {
			final Future<?> first = executor.submit(
					() -> service.getMetrics(validRequest(), "first-tab", "0"));
			enteredRepository.await(2, TimeUnit.SECONDS);
			final Future<ResponseEntity<?>> second = executor.submit(() -> controller.getMetrics(
					validRequest(), "second-tab", "0"));

			final ResponseEntity<?> response = second.get(7, TimeUnit.SECONDS);
			assertEquals(HttpStatus.SERVICE_UNAVAILABLE, response.getStatusCode());
			assertEquals("2", response.getHeaders().getFirst("Retry-After"));
			assertEquals("query_busy", ((java.util.Map<?, ?>) response.getBody()).get("code"));
			assertTrue(output.getAll().contains(
					"Metrics query rejected: code=query_busy reason=timeout series=1"));
			releaseRepository.countDown();
			first.get(2, TimeUnit.SECONDS);
		} finally {
			releaseRepository.countDown();
			executor.shutdownNow();
		}
	}

	private static GetMetricsRequest validRequest() {
		final MetricsSeriesRequest series = new MetricsSeriesRequest();
		series.setRunId("run");
		series.setTagKey("loss");
		series.setFromStep(0L);
		series.setToStep(1L);
		series.setMaxPoints(3);
		final GetMetricsRequest request = new GetMetricsRequest();
		request.setSeries(List.of(series));
		return request;
	}
}
