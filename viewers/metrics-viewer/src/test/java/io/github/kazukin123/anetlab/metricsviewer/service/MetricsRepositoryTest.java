package io.github.kazukin123.anetlab.metricsviewer.service;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;
import org.springframework.test.util.ReflectionTestUtils;

import io.github.kazukin123.anetlab.metricsviewer.infra.model.MetricsFileBlock;
import io.github.kazukin123.anetlab.metricsviewer.infra.model.MetricsFileLine;
import io.github.kazukin123.anetlab.metricsviewer.view.model.TagTrace;

class MetricsRepositoryTest {

	@Test
	void findTagTraceDiffKeepsDecimatedStepsAscending() {
		final MetricsRepository repository = new MetricsRepository();
		ReflectionTestUtils.setField(repository, "decimationEnabled", true);
		ReflectionTestUtils.setField(repository, "maxTransferPointsInitial", 6);
		ReflectionTestUtils.setField(repository, "maxTransferPointsDelta", 6);

		repository.mergeMetrics("run1", MetricsFileBlock.builder()
				.startOffset(0)
				.endOffset(1)
				.lastModified(0)
				.isEOF(true)
				.lines(List.of(
						scalarLine("loss", 0, 10.0f),
						scalarLine("loss", 1, 0.0f),
						scalarLine("loss", 2, 5.0f),
						scalarLine("loss", 3, 20.0f),
						scalarLine("loss", 4, -1.0f),
						scalarLine("loss", 5, 7.0f),
						scalarLine("loss", 6, 8.0f)))
				.build());

		final List<TagTrace> traces = repository.findTagTraceDiff(Map.of("run1", Map.of("loss", 0)));

		assertEquals(1, traces.size());
		assertArrayEquals(new int[] {0, 1, 2, 3, 4, 6}, traces.get(0).getSteps());
	}

	private static MetricsFileLine scalarLine(String tag, int step, float value) {
		return MetricsFileLine.builder()
				.type("scalar")
				.tag(tag)
				.step(step)
				.value(value)
				.build();
	}
}
