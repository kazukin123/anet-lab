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
import io.github.kazukin123.anetlab.metricsviewer.service.model.Point;
import io.github.kazukin123.anetlab.metricsviewer.view.model.RunStats;
import io.github.kazukin123.anetlab.metricsviewer.view.model.TagInfo;
import io.github.kazukin123.anetlab.metricsviewer.view.model.TagStats;
import io.github.kazukin123.anetlab.metricsviewer.view.model.TagTrace;

/**
 * Holds and manages all MetricsSnapshot objects for each run.
 * Thread-safe cache structure accessed by MetricsService and LoadingThread.
 */
@Component
public class MetricsRepository {

	private static final String SNAPSHOT_FILENAME = "metrics_cache.kryo";
	private static final String SNAPSHOT_FILEHEADER = "metrics_snapshot.kryo_v0.1.0";

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

	/**
	 * 差分ロード用：Run×Tagごとに指定step以降のTagTraceを返す。
	 * 
	 * @param runTagMap runId → (tagKey → fromStep)
	 * @return 差分データ（TagTraceリスト）
	 */
	public List<TagTrace> findTagTraceDiff(Map<String, Map<String, Integer>> runTagMap) {
	    final List<TagTrace> traces = new ArrayList<>();

	    // requestのRunIdを回す
	    for (Map.Entry<String, Map<String, Integer>> runEntry : runTagMap.entrySet()) {
	        final String runId = runEntry.getKey();
	        final MetricsSnapshot snapshot = runSnapshotMap.get(runId);
	        if (snapshot == null) continue;

	        // requestのTagKeyを回す（空の場合は全Tag対象）
	        final Map<String, Integer> tagMap = runEntry.getValue();
	        final boolean allTags = (tagMap == null || tagMap.isEmpty());
	        final List<String> tagKeys = allTags ? snapshot.listTagKeys() : new ArrayList<>(tagMap.keySet());

	        for (String tagKey : tagKeys) {
	            final int fromStep = allTags ? 0 : tagMap.getOrDefault(tagKey, 0);

	            // 差分抽出
	            final List<Point> points = snapshot.getPointsSince(tagKey, fromStep);
	            if (points == null || points.isEmpty()) continue; // データなしはスキップ

	            // Trace元ネタを詰め直し
	            final int size = points.size();
	            final int[] steps = new int[size];
	            final float[] values = new float[size];
	            for (int i = 0; i < size; i++) {
	                final Point p = points.get(i);
	                steps[i] = p.getStep();
	                values[i] = p.getValue();
	            }

	            final TagStats stats = snapshot.getTagStats(tagKey);

	            final TagTrace trace = TagTrace.builder()
	                    .runId(runId)
	                    .tagKey(tagKey)
	                    .type("scalar")
	                    .stats(stats)
	                    .beginStep(fromStep)
	                    .endStep(steps[size - 1])
	                    .steps(steps)
	                    .values(values)
	                    .build();
	            traces.add(trace);
	        }
	    }

	    return traces;
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

	/**
	 * 現在ロードされている全RunIDを返す。
	 */
	public List<String> listAllRunIds() {
		if (runSnapshotMap.isEmpty()) return Collections.emptyList();
		return new ArrayList<>(runSnapshotMap.keySet());
	}

	/**
	 * 指定したRunに存在する全タグキーを返す。
	 */
	public List<String> listTagKeys(String runId) {
		if (runId == null) return Collections.emptyList();

		final MetricsSnapshot snapshot = runSnapshotMap.get(runId);
		if (snapshot == null) return Collections.emptyList();

		// Snapshot内部のタグ情報を取得
		final List<String> tagKeys = new ArrayList<>();
		snapshot.getTags().forEach(tagInfo -> tagKeys.add(tagInfo.getKey()));

		return tagKeys;
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
