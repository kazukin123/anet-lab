package io.github.kazukin123.anetlab.metricsviewer.service;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.anyList;
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
import org.springframework.boot.test.system.CapturedOutput;
import org.springframework.boot.test.system.OutputCaptureExtension;
import org.springframework.http.HttpStatus;

import io.github.kazukin123.anetlab.metricsviewer.config.MetricsViewerSettings;
import io.github.kazukin123.anetlab.metricsviewer.view.model.GetMetricsRequest;
import io.github.kazukin123.anetlab.metricsviewer.view.model.MetricsSeriesRequest;

class MetricsQueryConcurrencyTest {

	@Test
	@ExtendWith(OutputCaptureExtension.class)
	void aQueryThatCannotAcquireASlotReturnsAndLogsTheRetryContract(CapturedOutput output)
			throws Exception {
		final CountDownLatch enteredRepository = new CountDownLatch(1);
		final CountDownLatch releaseRepository = new CountDownLatch(1);
		final MetricsRepository repository = mock(MetricsRepository.class);
		when(repository.query(anyList())).thenAnswer(ignored -> {
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
		final MetricsService service = new MetricsService(
				workspaceManager,
				mock(LoadingThread.class),
				settings);

		final ExecutorService executor = Executors.newFixedThreadPool(2);
		try {
			final Future<?> first = executor.submit(() -> service.getMetrics(validRequest()));
			enteredRepository.await(2, TimeUnit.SECONDS);
			final Future<MetricsApiException> second = executor.submit(() -> {
				try {
					service.getMetrics(validRequest());
					throw new AssertionError("Expected query_busy");
				} catch (MetricsApiException e) {
					return e;
				}
			});

			final MetricsApiException error = second.get(7, TimeUnit.SECONDS);
			assertEquals(HttpStatus.SERVICE_UNAVAILABLE, error.getStatus());
			assertEquals("2", error.getHeaders().getFirst("Retry-After"));
			assertEquals("query_busy", ((java.util.Map<?, ?>) error.getBody()).get("code"));
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
