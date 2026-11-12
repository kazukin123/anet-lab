package io.github.kazukin123.anetlab.metricsviewer.service;

import java.nio.file.Files;
import java.nio.file.Path;
import java.lang.management.ManagementFactory;
import java.sql.Connection;
import java.security.MessageDigest;
import java.util.HexFormat;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsCacheDatabase;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsSource;

/** 隔離した入力とDBだけを使う実Run性能・永続データ比較。 */
public class CacheRebuildBenchmark {
	public static void main(String[] args) throws Exception {
		Path run = Path.of(args[0]).toAbsolutePath().normalize();
		MetricsCacheDatabase db = new MetricsCacheDatabase();
		GzipInputSessions sessions = new GzipInputSessions();
		MetricsIngestor ingestor = new MetricsIngestor(db, sessions, new LodIngestWriter(), new RunWarningRegistry());
		long start = System.nanoTime();
		int blocks = 0;
		try {
			while (true) {
				var result = ingestor.ingestBlock("benchmark", run, MetricsSource.select(run).orElseThrow());
				blocks++;
				if (result.state() == MetricsCacheDatabase.IngestState.ERROR)
					throw new IllegalStateException("Ingest failed");
				if (!result.immediateRetry())
					break;
			}
		}
		finally {
			sessions.closeAll();
		}
		double seconds = (System.nanoTime() - start) / 1e9;
		long heap = ManagementFactory.getMemoryPoolMXBeans().stream()
				.filter(p -> p.getType() == java.lang.management.MemoryType.HEAP)
				.mapToLong(p -> p.getPeakUsage().getUsed()).sum();
		System.out.println("BENCH timeSeconds=" + seconds + " blocks=" + blocks + " heapPeakBytes=" + heap + " dbBytes="
				+ Files.size(run.resolve("metrics_cache.db")));
		// 世代、sourceパス・圧縮位置を除き、全永続データをPK順で厳密比較する。
		try (var handle = db.openRead(run)) {
			Connection c = handle.connection();
			String[] tables = { "tags", "scalars", "scalars_lod", "tag_stats", "json_lines" };
			String[] orders = { "id", "tag_id,ordinal", "tag_id,level,bucket", "tag_id", "ordinal" };
			for (int i = 0; i < tables.length; i++) {
				MessageDigest hash = MessageDigest.getInstance("SHA-256");
				long count = 0;
				try (var st = c.createStatement();
						var rs = st.executeQuery("SELECT * FROM " + tables[i] + " ORDER BY " + orders[i])) {
					int columns = rs.getMetaData().getColumnCount();
					while (rs.next()) {
						for (int j = 1; j <= columns; j++) {
							Object v = rs.getObject(j);
							byte[] bytes = (v == null ? "<null>" : v.toString())
									.getBytes(java.nio.charset.StandardCharsets.UTF_8);
							hash.update(java.nio.ByteBuffer.allocate(4).putInt(bytes.length).array());
							hash.update(bytes);
						}
						count++;
					}
				}
				System.out.println("DATA table=" + tables[i] + " rows=" + count + " sha256="
						+ HexFormat.of().formatHex(hash.digest()));
			}
		}
	}
}
