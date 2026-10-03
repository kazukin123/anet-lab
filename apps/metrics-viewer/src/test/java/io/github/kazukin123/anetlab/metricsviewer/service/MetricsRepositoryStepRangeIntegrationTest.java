package io.github.kazukin123.anetlab.metricsviewer.service;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Base64;
import java.util.List;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import io.github.kazukin123.anetlab.metricsviewer.config.MetricsViewerSettings;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsCacheDatabase;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsSource;
import io.github.kazukin123.anetlab.metricsviewer.infra.RunScanner;
import io.github.kazukin123.anetlab.metricsviewer.view.model.MetricsSeriesRequest;
import io.github.kazukin123.anetlab.metricsviewer.view.model.MetricsSeriesResult;
import io.github.kazukin123.anetlab.metricsviewer.view.model.RawProjection;

/**
 * stepの閉区間を序数の半開区間へ写像する境界を固定する。
 * 先頭と末尾に同じstepが重なるtagで、step範囲の外側にある端(件数だけで決まる経路)と
 * 内側にある端(二分探索の経路)が同じ規則で序数を返すことを確かめる。
 */
class MetricsRepositoryStepRangeIntegrationTest {

	private static final String RUN_ID = "run-step-range";
	private static final long[] STEPS = {5, 5, 5, 6, 7, 7, 7};

	@TempDir
	private Path tempDir;

	private MetricsRepository repository;

	@BeforeEach
	void ingestTagWithDuplicatedBoundarySteps() throws Exception {
		final Path runDir = tempDir.resolve(RUN_ID);
		Files.createDirectories(runDir);
		final StringBuilder jsonl = new StringBuilder();
		for (int i = 0; i < STEPS.length; i++) {
			jsonl.append("{\"type\":\"scalar\",\"tag\":\"loss\",\"step\":")
					.append(STEPS[i])
					.append(",\"value\":")
					.append(i + 1)
					.append(".0}\n");
		}
		Files.writeString(runDir.resolve("metrics.jsonl"), jsonl, StandardCharsets.UTF_8);
		final MetricsCacheDatabase database = new MetricsCacheDatabase();
		new MetricsIngestor(database).ingestBlock(
				RUN_ID, runDir, MetricsSource.select(runDir).orElseThrow());

		final MetricsViewerSettings settings = new MetricsViewerSettings(100, 1000, 0, 1);
		repository = new MetricsRepository(
				new RunScanner(tempDir.toString()),
				database,
				settings,
				new MetricsRangeProjector(new LodPageCache(settings)));
	}

	@Test
	void rangesCoveringTheWholeTagIncludeEveryDuplicatedBoundaryPoint() {
		assertRawSteps(new long[] {5, 5, 5, 6, 7, 7, 7}, query(-100, 100));
		assertRawSteps(new long[] {5, 5, 5, 6, 7, 7, 7}, query(5, 7));
	}

	@Test
	void rangesWithOneInnerEndSearchOnlyThatEnd() {
		assertRawSteps(new long[] {5, 5, 5, 6}, query(5, 6));
		assertRawSteps(new long[] {6, 7, 7, 7}, query(6, 7));
		assertRawSteps(new long[] {6}, query(6, 6));
	}

	@Test
	void rangesOutsideTheTagAreEmpty() {
		for (MetricsSeriesResult result : List.of(query(8, 100), query(-100, 4))) {
			assertEquals(SeriesAvailability.EMPTY.externalName(), result.getAvailability());
			assertNull(result.getProjection());
		}
	}

	private MetricsSeriesResult query(long fromStep, long toStep) {
		final MetricsSeriesRequest request = new MetricsSeriesRequest();
		request.setRunId(RUN_ID);
		request.setTagKey("loss");
		request.setFromStep(fromStep);
		request.setToStep(toStep);
		return new MetricsQueryCoordinator(1).run(
				new QueryChannel(java.util.UUID.randomUUID().toString()),
				0L,
				execution -> repository.query(List.of(request), execution)).get(0);
	}

	private static void assertRawSteps(long[] expected, MetricsSeriesResult result) {
		assertEquals(SeriesAvailability.OK.externalName(), result.getAvailability());
		final RawProjection projection = (RawProjection) result.getProjection();
		final ByteBuffer buffer = ByteBuffer.wrap(Base64.getDecoder().decode(projection.steps().get(0)))
				.order(ByteOrder.LITTLE_ENDIAN);
		final long[] actual = new long[buffer.remaining() / Double.BYTES];
		for (int i = 0; i < actual.length; i++) actual[i] = (long) buffer.getDouble();
		assertArrayEquals(expected, actual);
	}
}
