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
	void fourRunsKeepTheirRequestedResolutionWithoutFairSharing() {
		final MetricsQueryPlanner planner=planner(200);
		for(int runCount=1;runCount<=4;runCount++) {
			final List<MetricsQueryPlanner.SeriesInput> inputs=new java.util.ArrayList<>();
			for(int run=0;run<runCount;run++) inputs.add(availableSeries(run,"loss",40L,60));
			for(var plan:planner.plan(inputs)) assertEquals(40,plan.pointBudget());
		}
	}

	@Test
	void rejectsAnExplicitSeriesCapInsteadOfIncreasingStepWidth() {
		final MetricsApiException error=assertThrows(MetricsApiException.class,
				()->planner(200).plan(List.of(availableSeries(0,"loss",100L,20))));
		assertEquals(HttpStatus.UNPROCESSABLE_ENTITY,error.getStatus());
		assertEquals(Map.of("seriesIndex",0,"requiredPoints",100L,"maxPoints",20),error.getBody());
	}

	@Test
	void lodBudgetUsesTheExplicitStepBinCount() {
		final var input=availableSeries(0,"loss",100L,30);
		input.request().setStepBucketWidth(10L);
		assertEquals(30,planner(200).plan(List.of(input))[0].pointBudget());
	}

	@Test
	void unavailableSeriesConsumeNoBudgetAndDoNotFailPlanning() {
		final MetricsQueryPlanner.SeriesInput notFound=new MetricsQueryPlanner.SeriesInput(
				0,request("missing",null),false,null,null,null);
		final MetricsQueryPlanner.SeriesInput empty=new MetricsQueryPlanner.SeriesInput(
				1,request("empty",null),true,readyMetadata(),
				new MetricsQueryPlanner.TagInput(2L,0L,0L,0L,false,null,null),null);
		final var plans=planner(100).plan(List.of(notFound,empty));
		assertEquals(SeriesAvailability.NOT_FOUND,plans[0].availability());
		assertEquals(SeriesAvailability.EMPTY,plans[1].availability());
		assertEquals(0,plans[0].pointBudget());
		assertEquals(0,plans[1].pointBudget());
	}

	@Test
	void rejectsWhenFullResolutionExceedsTheRequestLimit() {
		final MetricsApiException error=assertThrows(MetricsApiException.class,
				()->planner(149).plan(List.of(availableSeries(0,"a",60L,60),
						availableSeries(1,"b",60L,60),availableSeries(2,"c",60L,60))));
		assertEquals(HttpStatus.UNPROCESSABLE_ENTITY,error.getStatus());
		assertEquals(Map.of("seriesCount",3,"requiredPoints",180L,"maxPointsPerRequest",149),error.getBody());
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
		final MetricsSeriesRequest request=request(tagKey,maxPoints);
		request.setToStep(availableVertices-1L);
		return new MetricsQueryPlanner.SeriesInput(
				index,
				request,
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
		request.setStepBucketWidth(1L);
		request.setToStep(100L);
		request.setMaxPoints(maxPoints);
		return request;
	}
}
