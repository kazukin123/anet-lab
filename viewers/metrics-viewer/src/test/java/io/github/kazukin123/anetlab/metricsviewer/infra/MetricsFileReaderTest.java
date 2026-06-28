package io.github.kazukin123.anetlab.metricsviewer.infra;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import io.github.kazukin123.anetlab.metricsviewer.infra.model.MetricsFileBlock;

class MetricsFileReaderTest {

	@TempDir
	Path tempDir;

	@Test
	void estimateInitialLineCapacityUsesReadableBytesWithoutChangingMaxLinesLimit() {
		assertEquals(4096, MetricsFileReader.estimateInitialLineCapacity(0L, 1_000_000));
		assertEquals(9375, MetricsFileReader.estimateInitialLineCapacity(1_200_000L, 1_000_000));
		assertEquals(1_000_000, MetricsFileReader.estimateInitialLineCapacity(200_000_000L, 1_000_000));
		assertEquals(2, MetricsFileReader.estimateInitialLineCapacity(1_200_000L, 2));
		assertEquals(4096, MetricsFileReader.estimateInitialLineCapacity(1_200_000L, 0));
	}

	@Test
	void parseDiffKeepsSmallDeltaContent() throws Exception {
		final Path metricsFile = tempDir.resolve("metrics.jsonl");
		Files.writeString(metricsFile,
				"{\"type\":\"scalar\",\"tag\":\"tagA\",\"step\":1,\"value\":2.0}\n" +
						"{\"type\":\"scalar\",\"tag\":\"tagB\",\"step\":2,\"value\":3.0}\n",
				StandardCharsets.UTF_8);

		final MetricsFileBlock block = new MetricsFileReader().parseDiff(metricsFile, 0, 10);

		assertEquals(2, block.getLines().size());
		assertEquals("tagA", block.getLines().get(0).getTag());
		assertEquals("tagB", block.getLines().get(1).getTag());
	}

	@Test
	void parseDiffStillStopsAtMaxLines() throws Exception {
		final Path metricsFile = tempDir.resolve("metrics.jsonl");
		Files.writeString(metricsFile,
				"{\"type\":\"scalar\",\"tag\":\"tagA\",\"step\":1,\"value\":2.0}\n" +
						"{\"type\":\"scalar\",\"tag\":\"tagB\",\"step\":2,\"value\":3.0}\n" +
						"{\"type\":\"scalar\",\"tag\":\"tagC\",\"step\":3,\"value\":4.0}\n",
				StandardCharsets.UTF_8);

		final MetricsFileBlock block = new MetricsFileReader().parseDiff(metricsFile, 0, 2);

		assertEquals(2, block.getLines().size());
		assertEquals("tagA", block.getLines().get(0).getTag());
		assertEquals("tagB", block.getLines().get(1).getTag());
		assertFalse(block.isEOF());
	}
}
