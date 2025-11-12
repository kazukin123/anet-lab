package io.github.kazukin123.anetlab.metricsviewer.service;

import java.io.IOException;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Stream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;

import io.github.kazukin123.anetlab.metricsviewer.infra.model.MetricsFileBlock;
import io.github.kazukin123.anetlab.metricsviewer.service.model.MetricsSnapshot;
import io.github.kazukin123.anetlab.metricsviewer.view.model.RunStats;
import io.github.kazukin123.anetlab.metricsviewer.view.model.TagInfo;
import io.github.kazukin123.anetlab.metricsviewer.view.model.TagTrace;

/**
 * Holds and manages all MetricsSnapshot objects for each run.
 * Thread-safe cache structure accessed by MetricsService and LoadingThread.
 */
@Component
public class MetricsRepository {

	private static final String SNAPSHOT_FILENAME = "metrics_cache.kryo";
	private static final String SNAPSHOT_FILEHEADER = "metrics_snapshot.kryo_v0.0.6";

	private static final Logger log = LoggerFactory.getLogger(MetricsRepository.class);

	private final Map<String, MetricsSnapshot> runSnapshotMap = new ConcurrentHashMap<>(); // runId → MetricsSnapshot
	private final Kryo kryo = new Kryo();

	/** コンストラクタ */
	public MetricsRepository() {
		kryo.setRegistrationRequired(false);
		//      kryo.register(MetricsSnapshot.class);
		//      kryo.register(Tag.class);
		//      kryo.register(MetricPoint.class);
		//      kryo.register(MetricFileBlock.class);
	}

	public long getLastReadPosition(String runId) {
		final long lastPos = Optional.ofNullable(runSnapshotMap.get(runId))
				.map(MetricsSnapshot::getLastReadPosition)
				.orElse(0L);
		return lastPos;
	}

	public void mergeMetrics(String runId, MetricsFileBlock block) {
		if (block == null || runId == null) return;

		final MetricsSnapshot snapshot = runSnapshotMap.computeIfAbsent(runId, id -> new MetricsSnapshot());
		snapshot.merge(block);
		snapshot.setLastReadPosition(block.getEndOffset());

		log.info("mergeMetrics run={} lines={} newPos={} tags={} points={} maxStep={}",
				runId,
				block.getLines() != null ? block.getLines().size() : 0,
				snapshot.getLastReadPosition(),
				snapshot.getTags().size(), snapshot.getTotalPoints(),
				snapshot.getRunStats().getMaxStep());
	}

	public List<TagTrace> findTagTrace(List<String> runIds, List<String> tagKeys) {
		final List<TagTrace> all = new ArrayList<>();

		if (runIds == null || runIds.isEmpty()) return all;

		for (String runId : runIds) {
			final MetricsSnapshot snapshot = runSnapshotMap.get(runId);
			if (snapshot == null) continue;

			final List<TagTrace> traces = snapshot.findTagTrace(tagKeys);
			for (TagTrace t : traces) {
				// runIdをここで埋める
				final TagTrace traceWithRun = TagTrace.builder()
						.runId(runId)
						.tagKey(t.getTagKey())
						.type(t.getType())
						.stats(t.getStats())
						.steps(t.getSteps())
						.values(t.getValues())
						.build();
				all.add(traceWithRun);
			}
		}

		log.debug("getTraces: runIds={} tagKeys={} traces={}", runIds, tagKeys, all.size());
		return all;
	}

	/**
	 * Returns all tags known for the given run.
	 */
	public List<TagInfo> findTagInfo(String runId) {
		final MetricsSnapshot snap = runSnapshotMap.get(runId);
		if (snap == null) {
			return Collections.emptyList();
		}
		return snap.getTags();
	}

	public RunStats getRunStats(String runId) {
		final MetricsSnapshot snapshot = runSnapshotMap.get(runId);
		if (snapshot == null) {
			return new RunStats();
		}
		final RunStats stats = snapshot.getRunStats();
		return stats;
	}

	/** 全キャッシュロード */
	public void loadCache(Path runsDir) {
		if (!Files.exists(runsDir)) {
			return;
		}

		try (Stream<Path> dirs = Files.list(runsDir)) {
			dirs.filter(Files::isDirectory).forEach(this::loadCacheForRun);
		} catch (IOException e) {
			log.warn("Failed to scan runs dir: {}", e.getMessage());
		}
	}

	/** 新しいRunを検出した際に個別ロード */
	public void loadCacheForRun(Path runDir) {
		final String runId = runDir.getFileName().toString();
		if (runSnapshotMap.containsKey(runId)) {
			return; // すでにロード済み
		}

		final Path file = runDir.resolve(SNAPSHOT_FILENAME);
		if (!Files.exists(file)) {
			return;
		}

		try (Input input = new Input(Files.newInputStream(file))) {
			// ヘッダチェック
			final String header = input.readString();
			log.debug("header={}", header);
			if (!SNAPSHOT_FILEHEADER.equals(header)) {
				throw new IOException("Header mismatch");
			}

			// Snapshotオブジェクト読込
			final MetricsSnapshot snapshot = kryo.readObject(input, MetricsSnapshot.class);
			runSnapshotMap.put(runId, snapshot);
			log.info("Loaded cache for run={} (offset={})", runId, snapshot.getLastReadPosition());
		} catch (Exception e) {
			log.warn("Failed to load cache for run {}: {}", runId, e.getMessage());
		}
	}

	public void saveCache(Path runDir, String runId) {
		final MetricsSnapshot snapshot = runSnapshotMap.get(runId);
		if (snapshot == null) return;

		final Path file = runDir.resolve(SNAPSHOT_FILENAME);
		final Path tmp = runDir.resolve(SNAPSHOT_FILENAME + ".tmp");

		// 一時ファイルを書き込み用に開く
		try (Output output = new Output(Files.newOutputStream(tmp))) {
			// ヘッダ書き込み
			output.writeString(SNAPSHOT_FILEHEADER);

			// snapshotオブジェクト書き込み
			kryo.writeObject(output, snapshot);

			// 本体と入れ替え
			Files.move(tmp, file,
					StandardCopyOption.REPLACE_EXISTING,
					StandardCopyOption.ATOMIC_MOVE);

			// ログ
			log.info("Snapshot saved. run={} pos={} points={}",
					runId, snapshot.getLastReadPosition(), snapshot.getTotalPoints());
		} catch (AtomicMoveNotSupportedException e) {
			try {
				Files.move(tmp, file, StandardCopyOption.REPLACE_EXISTING);
			} catch (IOException ex) {
				log.error("Failed to finalize snapshot move for {}: {}", runId, ex.getMessage());
			}
		} catch (Exception e) {
			log.error("Failed to save snapshot for {}: {}", runId, e.getMessage());
			try {
				Files.deleteIfExists(tmp);
			} catch (IOException ignored) {
				
			}
		}
	}

}
