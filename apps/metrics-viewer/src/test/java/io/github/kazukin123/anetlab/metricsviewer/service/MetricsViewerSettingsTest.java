package io.github.kazukin123.anetlab.metricsviewer.service;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

import io.github.kazukin123.anetlab.metricsviewer.config.MetricsViewerSettings;

class MetricsViewerSettingsTest {

	@Test
	void acceptsTheDocumentedDefaultsAndDisabledCache() {
		final MetricsViewerSettings defaults = new MetricsViewerSettings(4000, 500000, 0, 2);

		assertEquals(4000, defaults.getTargetPointsPerSeries());
		assertEquals(500000, defaults.getMaxPointsPerRequest());
		assertEquals(0L, defaults.getCacheMemoryBytes());
		assertEquals(2, defaults.getMaxConcurrentQueries());
	}

	@Test
	void rejectsEveryOutOfRangePublicSetting() {
		assertThrows(
				IllegalArgumentException.class,
				() -> new MetricsViewerSettings(2, 500000, 0, 2));
		assertThrows(
				IllegalArgumentException.class,
				() -> new MetricsViewerSettings(4001, 4000, 0, 2));
		assertThrows(
				IllegalArgumentException.class,
				() -> new MetricsViewerSettings(4000, 1_000_001, 0, 2));
		assertThrows(
				IllegalArgumentException.class,
				() -> new MetricsViewerSettings(4000, 500000, -1, 2));
		assertThrows(
				IllegalArgumentException.class,
				() -> new MetricsViewerSettings(4000, 500000, Integer.MAX_VALUE, 2));
		assertThrows(
				IllegalArgumentException.class,
				() -> new MetricsViewerSettings(4000, 500000, 0, 5));
	}
}
