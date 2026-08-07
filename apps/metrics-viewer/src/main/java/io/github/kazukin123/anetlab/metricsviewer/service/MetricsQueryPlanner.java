package io.github.kazukin123.anetlab.metricsviewer.service;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

import org.springframework.http.HttpStatus;

import io.github.kazukin123.anetlab.metricsviewer.config.MetricsViewerSettings;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsCacheDatabase.CacheMetadata;
import io.github.kazukin123.anetlab.metricsviewer.view.model.Issue;
import io.github.kazukin123.anetlab.metricsviewer.view.model.MetricsSeriesRequest;

final class MetricsQueryPlanner {
	private static final int MIN_POINTS_PER_SERIES = 50;

	private final MetricsViewerSettings settings;

	MetricsQueryPlanner(MetricsViewerSettings settings) {
		this.settings = settings;
	}

	SeriesPlan[] plan(List<SeriesInput> inputs) {
		final SeriesPlan[] plans = inputs.stream()
				.map(this::inspect)
				.toArray(SeriesPlan[]::new);
		allocatePointBudgets(plans);
		return plans;
	}

	private SeriesPlan inspect(SeriesInput input) {
		final SeriesPlan plan = new SeriesPlan(
				input.index(),
				input.request(),
				input.metadata() == null ? null : parseUuid(input.metadata().generation()),
				input.metadata() == null ? null : input.metadata().generation());
		if (!input.runExists()) {
			plan.availability = SeriesAvailability.NOT_FOUND;
			return plan;
		}
		if (input.metadata() == null) {
			plan.availability = SeriesAvailability.PENDING;
			return plan;
		}
		if (input.metadata().errorCode() != null) {
			plan.issues.add(new Issue(
					"run", input.metadata().errorCode(), input.metadata().errorMessage()));
		}
		if (input.queryError() != null) {
			plan.availability = SeriesAvailability.PENDING;
			plan.issues.add(new Issue("run", "query_error", input.queryError()));
			return plan;
		}

		final boolean stillIngesting;
		try {
			stillIngesting = input.metadata().stateOrThrow().isStillIngesting();
		} catch (IllegalArgumentException e) {
			plan.availability = SeriesAvailability.PENDING;
			plan.issues.add(new Issue("run", "query_error", e.getMessage()));
			return plan;
		}

		final TagInput tag = input.tag();
		if (tag == null) {
			plan.availability = stillIngesting
					? SeriesAvailability.PENDING
					: SeriesAvailability.NOT_FOUND;
			return plan;
		}
		plan.tagId = tag.tagId();
		if (tag.hasError()) {
			plan.issues.add(new Issue("tag", tag.errorCode(), tag.errorMessage()));
		}
		if (tag.totalCount() == null || tag.totalCount() == 0L) {
			plan.availability = stillIngesting
					? SeriesAvailability.PENDING
					: SeriesAvailability.EMPTY;
			return plan;
		}
		plan.ordinalFrom = tag.ordinalFrom();
		plan.ordinalTo = tag.ordinalTo();
		plan.availableVertices = Math.max(0L, plan.ordinalTo - plan.ordinalFrom);
		plan.availability = plan.availableVertices > 0L
				? SeriesAvailability.OK
				: (stillIngesting ? SeriesAvailability.PENDING : SeriesAvailability.EMPTY);
		return plan;
	}

	private void allocatePointBudgets(SeriesPlan[] plans) {
		long requiredMinimum = 0L;
		for (SeriesPlan plan : plans) {
			if (plan.availability != SeriesAvailability.OK) continue;
			final int requested = plan.request.getMaxPoints() == null
					? settings.getTargetPointsPerSeries()
					: plan.request.getMaxPoints();
			plan.cap = (int) Math.min((long) requested, plan.availableVertices);
			plan.pointBudget = Math.min(MIN_POINTS_PER_SERIES, plan.cap);
			requiredMinimum += plan.pointBudget;
		}
		if (requiredMinimum > settings.getMaxPointsPerRequest()) {
			final Map<String, Object> body = new LinkedHashMap<>();
			body.put("seriesCount", plans.length);
			body.put("requiredMinimumPoints", requiredMinimum);
			body.put("maxPointsPerRequest", settings.getMaxPointsPerRequest());
			throw new MetricsApiException(HttpStatus.UNPROCESSABLE_ENTITY, body);
		}

		int remaining = settings.getMaxPointsPerRequest() - (int) requiredMinimum;
		while (remaining > 0) {
			final List<SeriesPlan> active = java.util.Arrays.stream(plans)
					.filter(plan -> plan.pointBudget < plan.cap)
					.toList();
			if (active.isEmpty()) break;
			final int fairShare = Math.max(1, remaining / active.size());
			boolean allocated = false;
			for (SeriesPlan plan : active) {
				final int addition =
						Math.min(plan.cap - plan.pointBudget, Math.min(fairShare, remaining));
				if (addition <= 0) continue;
				plan.pointBudget += addition;
				remaining -= addition;
				allocated = true;
				if (remaining == 0) break;
			}
			if (!allocated) break;
		}
	}

	private static UUID parseUuid(String value) {
		try {
			return value == null ? null : UUID.fromString(value);
		} catch (IllegalArgumentException e) {
			return null;
		}
	}

	record SeriesInput(
			int index,
			MetricsSeriesRequest request,
			boolean runExists,
			CacheMetadata metadata,
			TagInput tag,
			String queryError) {
	}

	record TagInput(
			long tagId,
			Long totalCount,
			long ordinalFrom,
			long ordinalTo,
			boolean hasError,
			String errorCode,
			String errorMessage) {
	}

	/**
	 * index、request、issues、availability、pointBudgetは応答値として常に有効で、
	 * availabilityがokでない系列のpointBudgetは0となる。
	 * generationValueはmetadataを読めた場合の生値、generationはそのUUID変換に成功した場合に有効となる。
	 * tagIdはtagを読めた場合に有効となる。
	 * ordinal範囲とavailableVerticesは正のtag統計を読めた場合に有効となり、
	 * capとpointBudgetはavailabilityがokの系列だけで配分値として有効となる。
	 */
	static final class SeriesPlan {
		private final int index;
		private final MetricsSeriesRequest request;
		private final List<Issue> issues = new ArrayList<>();
		private final UUID generation;
		private final String generationValue;
		private long tagId;
		private long ordinalFrom;
		private long ordinalTo;
		private long availableVertices;
		private SeriesAvailability availability;
		private int cap;
		private int pointBudget;

		private SeriesPlan(
				int index,
				MetricsSeriesRequest request,
				UUID generation,
				String generationValue) {
			this.index = index;
			this.request = request;
			this.generation = generation;
			this.generationValue = generationValue;
		}

		int index() {
			return index;
		}

		MetricsSeriesRequest request() {
			return request;
		}

		List<Issue> issues() {
			return issues;
		}

		UUID generation() {
			return generation;
		}

		String generationValue() {
			return generationValue;
		}

		long tagId() {
			return tagId;
		}

		long ordinalFrom() {
			return ordinalFrom;
		}

		long ordinalTo() {
			return ordinalTo;
		}

		int pointBudget() {
			return pointBudget;
		}

		SeriesAvailability availability() {
			return availability;
		}
	}
}
