package io.github.kazukin123.anetlab.metricsviewer.service;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
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
import java.sql.SQLException;
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
	void trailingPageIsRetainedAndReloadedOnlyAfterNewBucketsAreCompleted() throws Exception {
		final Fixture fixture = createFixture();
		final MetricsViewerSettings settings = mock(MetricsViewerSettings.class);
		when(settings.getCacheMemoryBytes()).thenReturn(1024L * 1024L);
		final LodPageCache cache = new LodPageCache(settings);
		final MetricsRangeProjector projector = new MetricsRangeProjector(cache);

		final Connection closedConnection;
		try (ConnectionHandle handle = fixture.database().openRead(fixture.runDir())) {
			closedConnection = handle.connection();
			projectBoundary(projector, handle.connection(), fixture, POINT_COUNT, 6);
			// 満杯のpage 0と、完成bucketが1件だけの末尾page 1を両方保持する。
			assertEquals(2, cache.pageCount());
		}
		// 保持済みpageだけで答えられるので、閉じた接続でもDBへ触れずに射影できる。
		projectBoundary(projector, closedConnection, fixture, POINT_COUNT, 6);
		assertEquals(16L, find(cache,
				closedConnection,
				fixture.generation(),
				"run-cache",
				fixture.tagId(),
				POINT_COUNT,
				1,
				LodPageCache.LOD_PAGE_BUCKETS).count());

		appendPoints(fixture, 16);
		try (ConnectionHandle handle = fixture.database().openRead(fixture.runDir())) {
			// 末尾pageの完成bucketが増えたsnapshotでは、保持中のpageが足りないので読み直す。
			final LodBucket appended = find(cache,
					handle.connection(),
					fixture.generation(),
					"run-cache",
					fixture.tagId(),
					POINT_COUNT + 16L,
					1,
					LodPageCache.LOD_PAGE_BUCKETS + 1L);
			assertNotNull(appended);
			assertEquals(16L, appended.count());
			assertEquals(POINT_COUNT + 16L, appended.ordinalTo());
		}
		assertEquals(2, cache.pageCount());
		// 読み直したpageは追記前のsnapshotにもそのまま使え、追記後のbucketはそのsnapshotでは見えない。
		assertEquals(16L, find(cache,
				closedConnection,
				fixture.generation(),
				"run-cache",
				fixture.tagId(),
				POINT_COUNT,
				1,
				LodPageCache.LOD_PAGE_BUCKETS).count());
		assertNull(find(cache,
				closedConnection,
				fixture.generation(),
				"run-cache",
				fixture.tagId(),
				POINT_COUNT,
				1,
				LodPageCache.LOD_PAGE_BUCKETS + 1L));

		cache.invalidateGeneration("run-cache", "different-generation");
		assertEquals(0, cache.pageCount());
		try (ConnectionHandle handle = fixture.database().openRead(fixture.runDir())) {
			projectBoundary(projector, handle.connection(), fixture, POINT_COUNT + 16L, 6);
		}
		assertEquals(2, cache.pageCount());
		cache.retainRuns(Set.of());
		assertEquals(0, cache.pageCount());
	}

	@Test
	void bucketsAfterTheCompletePrefixAreNotLookedUp() throws Exception {
		final Connection connection = mock(Connection.class);
		when(connection.prepareStatement(anyString()))
				.thenThrow(new SQLException("incomplete buckets must not be queried"));
		for (long capacityBytes : new long[] {0L, 1024L * 1024L}) {
			final MetricsViewerSettings settings = mock(MetricsViewerSettings.class);
			when(settings.getCacheMemoryBytes()).thenReturn(capacityBytes);
			final LodPageCache cache = new LodPageCache(settings);

			// 175点ならlevel 1の完成bucketは0〜9で、10番は子が15件しかない未完成bucketである。
			assertNull(find(cache, connection, "generation", "run-cache", 1L, 175L, 1, 10L));
			assertNull(find(cache, connection, "generation", "run-cache", 1L, 255L, 2, 0L));
			assertEquals(0, cache.pageCount());
		}
	}

	@Test
	void missingCompleteBucketFailsFast() throws Exception {
		final Fixture fixture = createFixture();
		try (ConnectionHandle handle = fixture.database().openWrite(fixture.runDir());
				PreparedStatement delete = handle.connection().prepareStatement(
						"DELETE FROM scalars_lod WHERE tag_id=? AND level=1 AND bucket=5")) {
			delete.setLong(1, fixture.tagId());
			assertEquals(1, delete.executeUpdate());
		}
		for (long capacityBytes : new long[] {0L, 1024L * 1024L}) {
			final MetricsViewerSettings settings = mock(MetricsViewerSettings.class);
			when(settings.getCacheMemoryBytes()).thenReturn(capacityBytes);
			final LodPageCache cache = new LodPageCache(settings);

			try (ConnectionHandle handle = fixture.database().openRead(fixture.runDir())) {
				final IllegalStateException error = assertThrows(
						IllegalStateException.class,
						() -> find(cache,
								handle.connection(),
								fixture.generation(),
								"run-cache",
								fixture.tagId(),
								POINT_COUNT,
								1,
								5L));
				assertTrue(error.getMessage().contains("missing")
						&& error.getMessage().contains("level=1"), error.getMessage());
			}
			assertEquals(0, cache.pageCount());
		}
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

		final LodBucket loaded = find(cache,
				connection, "generation", "run-cache", 1L, (bucket + 1L) * 16L, 1, bucket);

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
		// 満杯page(1024 bucket)と末尾page(1 bucket)は、1 bucketあたり96 byteで数える。
		final long bothPagesBytes = (LodPageCache.LOD_PAGE_BUCKETS + 1L) * 96L;
		final LodPageCache exactFit = projectBoundaryWithCapacity(fixture, bothPagesBytes);
		assertEquals(2, exactFit.pageCount());
		assertEquals(bothPagesBytes, exactFit.usedBytes());

		// 1 byte足りなければ、先に保持した満杯pageを追い出して末尾pageだけが残る。
		final LodPageCache oneByteShort = projectBoundaryWithCapacity(fixture, bothPagesBytes - 1L);
		assertEquals(1, oneByteShort.pageCount());
		assertEquals(96L, oneByteShort.usedBytes());
	}

	private static LodPageCache projectBoundaryWithCapacity(
			Fixture fixture,
			long capacityBytes) throws Exception {
		final MetricsViewerSettings settings = mock(MetricsViewerSettings.class);
		when(settings.getCacheMemoryBytes()).thenReturn(capacityBytes);
		final LodPageCache cache = new LodPageCache(settings);
		try (ConnectionHandle handle = fixture.database().openRead(fixture.runDir())) {
			projectBoundary(new MetricsRangeProjector(cache), handle.connection(), fixture, POINT_COUNT, 6);
		}
		return cache;
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
			long tagCount,
			int pointBudget) throws Exception {
		final long ordinalFrom = (LodPageCache.LOD_PAGE_BUCKETS - 1L) * 16L;
		final long ordinalTo = (LodPageCache.LOD_PAGE_BUCKETS + 1L) * 16L;
		new MetricsQueryCoordinator(1).run(
				new QueryChannel(java.util.UUID.randomUUID().toString()),
				0L,
				execution -> {
					try {
						projector.project(
								connection,
								fixture.generation(),
								"run-cache",
								fixture.tagId(),
								tagCount,
								ordinalFrom,
								ordinalTo,
								pointBudget,
								execution);
						return null;
					} catch (Exception e) {
						throw new IllegalStateException(e);
					}
				});
	}

	private static LodBucket find(
			LodPageCache cache,
			Connection connection,
			String generation,
			String runId,
			long tagId,
			long tagCount,
			int level,
			long bucket) {
		return new MetricsQueryCoordinator(1).run(
				new QueryChannel(java.util.UUID.randomUUID().toString()),
				0L,
				execution -> {
					try {
						return cache.find(
								connection, generation, runId, tagId, tagCount, level, bucket, execution);
					} catch (Exception e) {
						throw new IllegalStateException(e);
					}
				});
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
