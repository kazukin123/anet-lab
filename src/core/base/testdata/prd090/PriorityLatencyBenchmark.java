package io.github.kazukin123.anetlab.metricsviewer.service;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import io.github.kazukin123.anetlab.metricsviewer.config.MetricsViewerSettings;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsCacheDatabase;

/** 隔離された大きい背景gzipと選択rawの反映待ちを公開閲覧APIから測定する。 */
public class PriorityLatencyBenchmark {
	public static void main(String[] args) throws Exception {
		final Path root = Path.of(args[0]).toAbsolutePath().normalize();
		final Path copiedGzip = Path.of(args[1]).toAbsolutePath().normalize();
		final Path selected = root.resolve("selected/runs/live");
		Files.createDirectories(selected);
		Files.writeString(selected.resolve("metrics.jsonl"), line(0), StandardCharsets.UTF_8);
		for (int index = 0; index < 3; index++) {
			final Path run = root.resolve("background" + index + "/runs/master");
			Files.createDirectories(run);
			Files.createLink(run.resolve("metrics.jsonl.gz"), copiedGzip);
		}
		final MetricsViewerSettings settings = new MetricsViewerSettings(8000, 500000, 256, 2);
		final MetricsQueryCoordinator coordinator = new MetricsQueryCoordinator(2);
		final GzipInputSessions streams = new GzipInputSessions();
		final WorkspaceManager manager = new WorkspaceManager(root.toString(), "selected", new MetricsCacheDatabase(),
				settings, coordinator, () -> streams);
		final LoadingThread writer = new LoadingThread(manager);
		final MetricsService api = new MetricsService(manager, writer, settings, coordinator);
		final List<Double> waits = new ArrayList<>();
		int activePeak = 0;
		try {
			manager.replacePriority(manager.currentEpoch(), Set.of(selected));
			writer.start();
			awaitCount(api, 1, writer);
			// 背景に処理可能な入力streamを確保した状態で、wakeや優先再要求なしの追記を測る。
			final long admissionDeadline = System.nanoTime() + 30_000_000_000L;
			while (streams.activeRuns().isEmpty() && System.nanoTime() < admissionDeadline)
				Thread.sleep(100);
			if (streams.activeRuns().isEmpty())
				throw new IllegalStateException("Background gzip was not admitted");
			for (int index = 1; index <= 20; index++) {
				final long start = System.nanoTime();
				Files.writeString(selected.resolve("metrics.jsonl"), line(index), StandardCharsets.UTF_8,
						StandardOpenOption.APPEND);
				awaitCount(api, index + 1, writer);
				final double milliseconds = (System.nanoTime() - start) / 1_000_000.0;
				waits.add(milliseconds);
				activePeak = Math.max(activePeak, streams.activeRuns().size());
				System.out.println("LATENCY sample=" + index + " milliseconds=" + milliseconds + " activeGzip="
						+ streams.activeRuns().size());
			}
			final List<Double> sorted = waits.stream().sorted().toList();
			final double p95 = sorted.get((int) Math.ceil(sorted.size() * 0.95) - 1);
			final long heap = java.lang.management.ManagementFactory.getMemoryPoolMXBeans().stream()
					.filter(pool -> pool.getType() == java.lang.management.MemoryType.HEAP)
					.mapToLong(pool -> pool.getPeakUsage().getUsed()).sum();
			System.out.println("PRIORITY samples=" + sorted.size() + " p50Ms=" + sorted.get(sorted.size() / 2)
					+ " p95Ms=" + p95 + " maxMs=" + sorted.get(sorted.size() - 1) + " heapPeakBytes=" + heap
					+ " activeGzipPeak=" + activePeak + " priorityStreamWaitMs=0 resource=raw source=admitted");
			if (p95 > 10000.0)
				throw new IllegalStateException("Priority p95 exceeds 10 seconds: " + p95);
		}
		finally {
			writer.terminateAndWait(30000);
			manager.shutdown();
			if (writer.isAlive() || !streams.activeRuns().isEmpty())
				throw new IllegalStateException("Input resources remain open");
		}
	}
	private static void awaitCount(MetricsService api, long expected, LoadingThread writer) throws Exception {
		final long deadline = System.nanoTime() + 15_000_000_000L;
		while (System.nanoTime() < deadline) {
			final var runs = api.getRuns().getRuns();
			if (!runs.isEmpty() && runs.get(0).getTags().stream()
					.anyMatch(tag -> tag.getStats() != null && tag.getStats().getCount() >= expected))
				return;
			if (!writer.isAlive())
				throw new IllegalStateException("Writer stopped before priority commit");
			Thread.sleep(100);
		}
		throw new IllegalStateException("Priority commit timed out: expected=" + expected);
	}
	private static String line(int step) {
		return "{\"type\":\"scalar\",\"tag\":\"live\",\"step\":" + step + ",\"value\":" + step + "}\n";
	}
}
