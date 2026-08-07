package io.github.kazukin123.anetlab.metricsviewer.service;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;
import org.springframework.http.HttpStatus;

import io.github.kazukin123.anetlab.metricsviewer.config.MetricsViewerSettings;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsCacheDatabase.CacheMetadata;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsCacheDatabase.IngestState;
import io.github.kazukin123.anetlab.metricsviewer.view.model.MetricsSeriesRequest;

class MetricsQueryPlannerTest {

	@Test
	void evenlyAllocatesTheRemainderInRequestOrder() {
		final MetricsQueryPlanner planner = planner(101);

		final MetricsQueryPlanner.SeriesPlan[] plans = planner.plan(List.of(
				availableSeries(0, "a", 80L, null),
				availableSeries(1, "b", 80L, null)));

		assertEquals(51, plans[0].pointBudget());
		assertEquals(50, plans[1].pointBudget());
	}

	@Test
	void redistributesBudgetLeftByACappedSeries() {
		final MetricsQueryPlanner planner = planner(140);

		final MetricsQueryPlanner.SeriesPlan[] plans = planner.plan(List.of(
				availableSeries(0, "a", 100L, 55),
				availableSeries(1, "b", 100L, 100)));

		assertEquals(55, plans[0].pointBudget());
		assertEquals(85, plans[1].pointBudget());
	}

	@Test
	void capsAMinimumQuotaBelowFiftyAndRedistributesTheRest() {
		final MetricsQueryPlanner planner = planner(100);

		final MetricsQueryPlanner.SeriesPlan[] plans = planner.plan(List.of(
				availableSeries(0, "a", 100L, 20),
				availableSeries(1, "b", 100L, 80)));

		assertEquals(20, plans[0].pointBudget());
		assertEquals(80, plans[1].pointBudget());
	}

	@Test
	void unavailableSeriesConsumeNoBudgetAndDoNotFailPlanning() {
		final MetricsQueryPlanner planner = planner(100);
		final MetricsQueryPlanner.SeriesInput notFound = new MetricsQueryPlanner.SeriesInput(
				0, request("missing", null), false, null, null, null);
		final MetricsQueryPlanner.SeriesInput empty = new MetricsQueryPlanner.SeriesInput(
				1,
				request("empty", null),
				true,
				readyMetadata(),
				new MetricsQueryPlanner.TagInput(2L, 0L, 0L, 0L, false, null, null),
				null);

		final MetricsQueryPlanner.SeriesPlan[] plans = planner.plan(List.of(notFound, empty));

		assertEquals(SeriesAvailability.NOT_FOUND, plans[0].availability());
		assertEquals(SeriesAvailability.EMPTY, plans[1].availability());
		assertEquals(0, plans[0].pointBudget());
		assertEquals(0, plans[1].pointBudget());
	}

	@Test
	void rejectsWhenTheMinimumQuotaExceedsTheRequestLimit() {
		final MetricsQueryPlanner planner = planner(149);

		final MetricsApiException error = assertThrows(
				MetricsApiException.class,
				() -> planner.plan(List.of(
						availableSeries(0, "a", 100L, 60),
						availableSeries(1, "b", 100L, 60),
						availableSeries(2, "c", 100L, 60))));

		assertEquals(HttpStatus.UNPROCESSABLE_ENTITY, error.getStatus());
		assertEquals(Map.of(
				"seriesCount", 3,
				"requiredMinimumPoints", 150L,
				"maxPointsPerRequest", 149), error.getBody());
	}

	private static MetricsQueryPlanner planner(int maxPointsPerRequest) {
		return new MetricsQueryPlanner(
				new MetricsViewerSettings(60, maxPointsPerRequest, 0, 1));
	}

	private static MetricsQueryPlanner.SeriesInput availableSeries(
			int index,
			String tagKey,
			long availableVertices,
			Integer maxPoints) {
		return new MetricsQueryPlanner.SeriesInput(
				index,
				request(tagKey, maxPoints),
				true,
				readyMetadata(),
				new MetricsQueryPlanner.TagInput(
						1L,
						availableVertices,
						0L,
						availableVertices,
						false,
						null,
						null),
				null);
	}

	private static CacheMetadata readyMetadata() {
		return new CacheMetadata(
				"00000000-0000-0000-0000-000000000001",
				IngestState.READY,
				null,
				100L,
				100L,
				null,
				null);
	}

	private static MetricsSeriesRequest request(String tagKey, Integer maxPoints) {
		final MetricsSeriesRequest request = new MetricsSeriesRequest();
		request.setRunId("run-ready");
		request.setTagKey(tagKey);
		request.setFromStep(0L);
		request.setToStep(100L);
		request.setMaxPoints(maxPoints);
		return request;
	}
}
