package io.github.kazukin123.anetlab.metricsviewer.service;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.Statement;
import java.util.Set;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.mockito.ArgumentCaptor;

import io.github.kazukin123.anetlab.metricsviewer.config.MetricsViewerSettings;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsCacheDatabase;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsCacheDatabase.ConnectionHandle;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsSource;

class LodPageCacheTest {

	private static final int POINT_COUNT = (LodPageCache.LOD_PAGE_BUCKETS + 1) * 16;

	@TempDir
	private Path tempDir;

	@Test
	void onlyCompletePagesAreReusedAndTrailingPageReloadsAfterAppend() throws Exception {
		final Fixture fixture = createFixture();
		final MetricsViewerSettings settings = mock(MetricsViewerSettings.class);
		when(settings.getCacheMemoryBytes()).thenReturn(1024L * 1024L);
		final LodPageCache cache = new LodPageCache(settings);
		final MetricsRangeProjector projector = new MetricsRangeProjector(cache);

		final Connection closedConnection;
		try (ConnectionHandle handle = fixture.database().openRead(fixture.runDir())) {
			closedConnection = handle.connection();
			projectBoundary(projector, handle.connection(), fixture, 6);
			assertEquals(1, cache.pageCount());
			projectBoundary(projector, handle.connection(), fixture, 6);
			assertEquals(1, cache.pageCount());
		}
		assertEquals(16L, cache.find(
				closedConnection,
				fixture.generation(),
				"run-cache",
				fixture.tagId(),
				1,
				0L).count());

		appendPoints(fixture, 16);
		try (ConnectionHandle handle = fixture.database().openRead(fixture.runDir())) {
			final LodBucket appended = cache.find(
					handle.connection(),
					fixture.generation(),
					"run-cache",
					fixture.tagId(),
					1,
					LodPageCache.LOD_PAGE_BUCKETS + 1L);
			assertNotNull(appended);
			assertEquals(16L, appended.count());
			assertEquals(POINT_COUNT + 16L, appended.ordinalTo());
		}
		assertEquals(1, cache.pageCount());

		cache.invalidateGeneration("run-cache", "different-generation");
		assertEquals(0, cache.pageCount());
		try (ConnectionHandle handle = fixture.database().openRead(fixture.runDir())) {
			projectBoundary(projector, handle.connection(), fixture, 6);
		}
		assertEquals(1, cache.pageCount());
		cache.retainRuns(Set.of());
		assertEquals(0, cache.pageCount());
	}

	@Test
	void disabledCacheLoadsOnlyTheRequestedBucket() throws Exception {
		final MetricsViewerSettings settings = mock(MetricsViewerSettings.class);
		when(settings.getCacheMemoryBytes()).thenReturn(0L);
		final LodPageCache cache = new LodPageCache(settings);
		final Connection connection = mock(Connection.class);
		final PreparedStatement statement = mock(PreparedStatement.class);
		final ResultSet result = mock(ResultSet.class);
		final long bucket = 10L;

		when(connection.prepareStatement(anyString())).thenReturn(statement);
		when(statement.executeQuery()).thenReturn(result);
		when(result.next()).thenReturn(true, false);
		when(result.getLong("bucket")).thenReturn(bucket);
		when(result.getLong("cnt")).thenReturn(16L);
		when(result.getLong("step_first")).thenReturn(160L);
		when(result.getLong("step_last")).thenReturn(175L);
		when(result.getLong("min_ordinal")).thenReturn(160L);
		when(result.getLong("min_step")).thenReturn(160L);
		when(result.getLong("max_ordinal")).thenReturn(175L);
		when(result.getLong("max_step")).thenReturn(175L);
		when(result.getDouble("vmin")).thenReturn(1.0);
		when(result.getDouble("vmax")).thenReturn(2.0);
		when(result.getDouble("vmean")).thenReturn(1.5);
		when(result.getDouble("vlast")).thenReturn(1.75);

		final LodBucket loaded = cache.find(
				connection, "generation", "run-cache", 1L, 1, bucket);

		assertNotNull(loaded);
		assertEquals(160L, loaded.ordinalFrom());
		assertEquals(176L, loaded.ordinalTo());
		assertEquals(0, cache.pageCount());
		final ArgumentCaptor<String> sql = ArgumentCaptor.forClass(String.class);
		verify(connection).prepareStatement(sql.capture());
		final String normalizedSql = sql.getValue().replaceAll("\\s+", " ").trim();
		assertTrue(normalizedSql.contains("WHERE tag_id=? AND level=? AND bucket=?"));
		assertFalse(normalizedSql.contains("bucket>=?"));
		verify(statement).setLong(3, bucket);
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
		assertEquals(98_304L, cache.usedBytes());
	}

	private static void appendPoints(Fixture fixture, int count) throws Exception {
		final StringBuilder jsonl = new StringBuilder();
		for (int step = POINT_COUNT; step < POINT_COUNT + count; step++) {
			jsonl.append("{\"type\":\"scalar\",\"tag\":\"loss\",\"step\":")
					.append(step)
					.append(",\"value\":")
					.append(step % 31)
					.append(".0}\n");
		}
		Files.writeString(
				fixture.runDir().resolve("metrics.jsonl"),
				jsonl,
				StandardCharsets.UTF_8,
				StandardOpenOption.APPEND);
		new MetricsIngestor(fixture.database()).ingestBlock(
				"run-cache",
				fixture.runDir(),
				MetricsSource.select(fixture.runDir()).orElseThrow());
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
