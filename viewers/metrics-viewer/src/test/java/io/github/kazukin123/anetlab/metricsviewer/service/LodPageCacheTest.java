package io.github.kazukin123.anetlab.metricsviewer.service;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.Statement;
import java.util.Set;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import io.github.kazukin123.anetlab.metricsviewer.config.MetricsViewerSettings;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsCacheDatabase;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsCacheDatabase.ConnectionHandle;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsSource;

class LodPageCacheTest {

	private static final int POINT_COUNT = (LodPageCache.LOD_PAGE_BUCKETS + 1) * 16;

	@TempDir
	private Path tempDir;

	@Test
	void overlappingRangesReuseFixedPagesAndIdentityChangesInvalidateThem() throws Exception {
		final Fixture fixture = createFixture();
		final MetricsViewerSettings settings = mock(MetricsViewerSettings.class);
		when(settings.getCacheMemoryBytes()).thenReturn(1024L * 1024L);
		final LodPageCache cache = new LodPageCache(settings);
		final MetricsRangeProjector projector = new MetricsRangeProjector(cache);

		try (ConnectionHandle handle = fixture.database().openRead(fixture.runDir())) {
			projectBoundary(projector, handle.connection(), fixture, 6);
			assertEquals(2, cache.pageCount());
			projectBoundary(projector, handle.connection(), fixture, 6);
			assertEquals(2, cache.pageCount());
		}

		cache.invalidateGeneration("run-cache", "different-generation");
		assertEquals(0, cache.pageCount());
		try (ConnectionHandle handle = fixture.database().openRead(fixture.runDir())) {
			projectBoundary(projector, handle.connection(), fixture, 6);
		}
		assertEquals(2, cache.pageCount());
		cache.retainRuns(Set.of());
		assertEquals(0, cache.pageCount());
	}

	@Test
	void evictionUsesThePrimitiveArrayByteCount() throws Exception {
		final Fixture fixture = createFixture();
		final MetricsViewerSettings settings = mock(MetricsViewerSettings.class);
		when(settings.getCacheMemoryBytes()).thenReturn(98_350L);
		final LodPageCache cache = new LodPageCache(settings);
		final MetricsRangeProjector projector = new MetricsRangeProjector(cache);

		try (ConnectionHandle handle = fixture.database().openRead(fixture.runDir())) {
			projectBoundary(projector, handle.connection(), fixture, 6);
		}

		assertEquals(1, cache.pageCount());
		assertEquals(96L, cache.usedBytes());
	}

	private Fixture createFixture() throws Exception {
		final Path runDir = tempDir.resolve("run-cache-" + System.nanoTime());
		Files.createDirectories(runDir);
		final StringBuilder jsonl = new StringBuilder();
		for (int step = 0; step < POINT_COUNT; step++) {
			jsonl.append("{\"type\":\"scalar\",\"tag\":\"loss\",\"step\":")
					.append(step)
					.append(",\"value\":")
					.append(step % 31)
					.append(".0}\n");
		}
		Files.writeString(runDir.resolve("metrics.jsonl"), jsonl, StandardCharsets.UTF_8);

		final MetricsCacheDatabase database = new MetricsCacheDatabase();
		new MetricsIngestor(database).ingestBlock(
				"run-cache",
				runDir,
				MetricsSource.select(runDir).orElseThrow());
		try (ConnectionHandle handle = database.openRead(runDir);
				Statement statement = handle.connection().createStatement()) {
			return new Fixture(
					runDir,
					database,
					queryString(statement, "SELECT v FROM source_meta WHERE k='generation'"),
					queryLong(statement, "SELECT id FROM tags WHERE key='loss'"));
		}
	}

	private static void projectBoundary(
			MetricsRangeProjector projector,
			Connection connection,
			Fixture fixture,
			int pointBudget) throws Exception {
		final long ordinalFrom = (LodPageCache.LOD_PAGE_BUCKETS - 1L) * 16L;
		final long ordinalTo = (LodPageCache.LOD_PAGE_BUCKETS + 1L) * 16L;
		projector.project(
				connection,
				fixture.generation(),
				"run-cache",
				fixture.tagId(),
				ordinalFrom,
				ordinalTo,
				pointBudget);
	}

	private static long queryLong(Statement statement, String sql) throws Exception {
		try (ResultSet result = statement.executeQuery(sql)) {
			result.next();
			return result.getLong(1);
		}
	}

	private static String queryString(Statement statement, String sql) throws Exception {
		try (ResultSet result = statement.executeQuery(sql)) {
			result.next();
			return result.getString(1);
		}
	}

	private record Fixture(
			Path runDir,
			MetricsCacheDatabase database,
			String generation,
			long tagId) {
	}
}
