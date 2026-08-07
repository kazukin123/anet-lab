package io.github.kazukin123.anetlab.metricsviewer.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

@Component
public class MetricsViewerSettings {

	private static final int MAX_CONFIGURED_POINTS = 1_000_000;

	private final int targetPointsPerSeries;
	private final int maxPointsPerRequest;
	private final long cacheMemoryBytes;
	private final int maxConcurrentQueries;

	public MetricsViewerSettings(
			@Value("${metricsviewer.target-points-per-series:8000}") int targetPointsPerSeries,
			@Value("${metricsviewer.max-points-per-request:500000}") int maxPointsPerRequest,
			@Value("${metricsviewer.cache-memory-mb:256}") int cacheMemoryMb,
			@Value("${metricsviewer.max-concurrent-queries:2}") int maxConcurrentQueries) {
		validateRange("metricsviewer.max-points-per-request", maxPointsPerRequest, 3, MAX_CONFIGURED_POINTS);
		validateRange("metricsviewer.target-points-per-series", targetPointsPerSeries, 3, maxPointsPerRequest);
		validateRange("metricsviewer.max-concurrent-queries", maxConcurrentQueries, 1, 4);
		if (cacheMemoryMb < 0) {
			throw invalid("metricsviewer.cache-memory-mb", cacheMemoryMb, "0 or greater");
		}

		final long requestedBytes = Math.multiplyExact((long) cacheMemoryMb, 1024L * 1024L);
		final long heapLimit = Runtime.getRuntime().maxMemory() / 2L;
		if (requestedBytes > heapLimit) {
			throw invalid("metricsviewer.cache-memory-mb", cacheMemoryMb,
					"no more than 50% of max heap (" + heapLimit / (1024L * 1024L) + " MiB)");
		}

		this.targetPointsPerSeries = targetPointsPerSeries;
		this.maxPointsPerRequest = maxPointsPerRequest;
		this.cacheMemoryBytes = requestedBytes;
		this.maxConcurrentQueries = maxConcurrentQueries;
	}

	private static void validateRange(String key, int value, int minimum, int maximum) {
		if (value < minimum || value > maximum) {
			throw invalid(key, value, minimum + ".." + maximum);
		}
	}

	private static IllegalArgumentException invalid(String key, Object value, String expected) {
		return new IllegalArgumentException(
				"Invalid configuration: key=" + key + " value=" + value + " expected=" + expected);
	}

	public int getTargetPointsPerSeries() {
		return targetPointsPerSeries;
	}

	public int getMaxPointsPerRequest() {
		return maxPointsPerRequest;
	}

	public long getCacheMemoryBytes() {
		return cacheMemoryBytes;
	}

	public int getMaxConcurrentQueries() {
		return maxConcurrentQueries;
	}
}
