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

	@Test
	void findTagTraceDiffReturnsUnrequestedKnownTagsFromBeginning() {
		final MetricsRepository repository = new MetricsRepository();

		repository.mergeMetrics("run1", MetricsFileBlock.builder()
				.startOffset(0)
				.endOffset(1)
				.lastModified(0)
				.isEOF(true)
				.lines(List.of(
						scalarLine("old_tag", 0, 10.0f),
						scalarLine("old_tag", 1, 11.0f),
						scalarLine("old_tag", 2, 12.0f),
						scalarLine("new_tag", 0, 20.0f),
						scalarLine("new_tag", 1, 21.0f)))
				.build());

		final List<TagTrace> traces = repository.findTagTraceDiff(Map.of("run1", Map.of("old_tag", 1)));

		assertEquals(2, traces.size());
		assertArrayEquals(new int[] {2}, findTrace(traces, "old_tag").getSteps());
		assertEquals(1, findTrace(traces, "old_tag").getBeginStep());
		assertArrayEquals(new int[] {0, 1}, findTrace(traces, "new_tag").getSteps());
		assertEquals(0, findTrace(traces, "new_tag").getBeginStep());
	}

	private static TagTrace findTrace(List<TagTrace> traces, String tagKey) {
		return traces.stream()
				.filter(trace -> tagKey.equals(trace.getTagKey()))
				.findFirst()
				.orElseThrow();
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
