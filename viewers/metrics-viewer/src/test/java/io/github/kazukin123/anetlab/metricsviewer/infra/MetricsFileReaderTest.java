package io.github.kazukin123.anetlab.metricsviewer.infra;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

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
	void parseDiffReadsFirstAppendedLineFromCommittedOffset() throws Exception {
		final Path metricsFile = tempDir.resolve("metrics.jsonl");
		final String lineA = "{\"type\":\"scalar\",\"tag\":\"tagA\",\"step\":1,\"value\":2.0}\n";
		final String lineB = "{\"type\":\"scalar\",\"tag\":\"tagB\",\"step\":2,\"value\":3.0}\n";
		final String lineC = "{\"type\":\"scalar\",\"tag\":\"tagC\",\"step\":3,\"value\":4.0}\n";
		final MetricsFileReader reader = new MetricsFileReader();

		Files.writeString(metricsFile, lineA + lineB, StandardCharsets.UTF_8);
		final MetricsFileBlock initial = reader.parseDiff(metricsFile, 0, 10);

		Files.writeString(metricsFile, lineC, StandardCharsets.UTF_8, StandardOpenOption.APPEND);
		final MetricsFileBlock delta = reader.parseDiff(metricsFile, initial.getEndOffset(), 10);

		assertEquals(1, delta.getLines().size());
		assertEquals("tagC", delta.getLines().get(0).getTag());
		assertEquals(Files.size(metricsFile), delta.getEndOffset());
	}

	@Test
	void parseDiffLeavesUnterminatedTrailingLineForNextRead() throws Exception {
		final Path metricsFile = tempDir.resolve("metrics.jsonl");
		final String lineA = "{\"type\":\"scalar\",\"tag\":\"tagA\",\"step\":1,\"value\":2.0}\n";
		final String lineBWithoutNewline = "{\"type\":\"scalar\",\"tag\":\"tagB\",\"step\":2,\"value\":3.0}";
		final MetricsFileReader reader = new MetricsFileReader();

		Files.writeString(metricsFile, lineA + lineBWithoutNewline, StandardCharsets.UTF_8);
		final MetricsFileBlock initial = reader.parseDiff(metricsFile, 0, 10);

		assertEquals(1, initial.getLines().size());
		assertEquals("tagA", initial.getLines().get(0).getTag());
		assertEquals(lineA.getBytes(StandardCharsets.UTF_8).length, initial.getEndOffset());
		assertTrue(initial.isEOF());

		Files.writeString(metricsFile, "\n", StandardCharsets.UTF_8, StandardOpenOption.APPEND);
		final MetricsFileBlock delta = reader.parseDiff(metricsFile, initial.getEndOffset(), 10);

		assertEquals(1, delta.getLines().size());
		assertEquals("tagB", delta.getLines().get(0).getTag());
		assertEquals(Files.size(metricsFile), delta.getEndOffset());
	}

	@Test
	void parseDiffStillStopsAtMaxLines() throws Exception {
		final Path metricsFile = tempDir.resolve("metrics.jsonl");
		final String lineA = "{\"type\":\"scalar\",\"tag\":\"tagA\",\"step\":1,\"value\":2.0}\n";
		final String lineB = "{\"type\":\"scalar\",\"tag\":\"tagB\",\"step\":2,\"value\":3.0}\n";
		final String lineC = "{\"type\":\"scalar\",\"tag\":\"tagC\",\"step\":3,\"value\":4.0}\n";
		Files.writeString(metricsFile, lineA + lineB + lineC, StandardCharsets.UTF_8);

		final MetricsFileBlock block = new MetricsFileReader().parseDiff(metricsFile, 0, 2);

		assertEquals(2, block.getLines().size());
		assertEquals("tagA", block.getLines().get(0).getTag());
		assertEquals("tagB", block.getLines().get(1).getTag());
		assertEquals((lineA + lineB).getBytes(StandardCharsets.UTF_8).length, block.getEndOffset());
		assertFalse(block.isEOF());
	}
}
