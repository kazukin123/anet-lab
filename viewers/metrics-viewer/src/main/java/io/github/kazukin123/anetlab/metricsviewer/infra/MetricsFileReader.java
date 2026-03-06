package io.github.kazukin123.anetlab.metricsviewer.infra;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.JsonToken;

import io.github.kazukin123.anetlab.metricsviewer.infra.model.MetricsFileBlock;
import io.github.kazukin123.anetlab.metricsviewer.infra.model.MetricsFileLine;

@Component
public class MetricsFileReader {

	private static final Logger log = LoggerFactory.getLogger(MetricsFileReader.class);
	private static final int BUFFER_SIZE = 1 << 16; // 64KB
	private static final int PROGRESS_INTERVAL = 100_000;
	private static final long PROGRESS_INTERVAL_MS = 2000;
	private static final JsonFactory JSON_FACTORY = new JsonFactory();

	/**
	 * ファイル全体をパースする。
	 */
	public MetricsFileBlock parseFull(Path jsonlFile, int maxLines) throws IOException {
		return readInternal(jsonlFile, 0, maxLines);
	}

	/**
	 * 指定オフセットからパースする（増分読み込み）。
	 */
	public MetricsFileBlock parseDiff(Path jsonlFile, long startOffset, int maxLines) throws IOException {
		return readInternal(jsonlFile, startOffset, maxLines);
	}

	/**
	 * 1行分のJSONLパース 
	 * @param line 位置行文字列
	 * @return パース結果
	 * @throws IOException
	 */
	private MetricsFileLine parseLine(String line) throws IOException {
		final JsonParser p = JSON_FACTORY.createParser(line);

		if (p.nextToken() != JsonToken.START_OBJECT) return null;

		String type = "";
		String tag = "";
		int step = 0;
		float value = 0.0f;

		while (p.nextToken() != JsonToken.END_OBJECT) {
			final String field = p.currentName();
			p.nextToken();

			switch (field) {
				case "type":
					type = p.getValueAsString();
					break;
				case "tag":
					tag = p.getValueAsString();
					break;
				case "step":
					step = p.getIntValue();
					break;
				case "value":
					value = p.getFloatValue();
					break;
				case "timestamp":
					break;
				default:
					p.skipChildren();
					break;
			}
		}
		return new MetricsFileLine(type, tag, step, value);
	}

	/**
	 * 実際の読み込み処理。
	 */
	private MetricsFileBlock readInternal(Path jsonlFile, long startOffset, int maxLines) throws IOException {
		if (!Files.exists(jsonlFile)) {
			log.warn("Metrics file not found: {}", jsonlFile);
			return new MetricsFileBlock(0, 0, List.of(), 0L, true);
		}

		// --- 行単位にオフセットを補正する ---
		if (startOffset > 0) {
			try (RandomAccessFile raf = new RandomAccessFile(jsonlFile.toFile(), "r")) {
				raf.seek(startOffset);
				int b;
				boolean skipped = false;
				while ((b = raf.read()) != -1) {
					if (b == '\n') {
						skipped = true;
						break;
					}
				}
				if (skipped) {
					final long newPos = raf.getFilePointer();
					if (newPos > startOffset) {
						log.debug("Offset adjusted from {} → {}", startOffset, newPos);
						startOffset = newPos;
					}
				}
			}
		}

		final long fileLastModified = Files.getLastModifiedTime(jsonlFile).toMillis();
		final List<MetricsFileLine> lines = new ArrayList<>(maxLines > 0 ? maxLines : 4096);

		long bytesRead = startOffset;
		long lastLogTime = System.currentTimeMillis();
		final long tStart = System.currentTimeMillis();

		log.debug("parse start file={} offset={} maxLines={}", jsonlFile, startOffset, maxLines);

		// バイト単位で読み込んで正確なオフセットを計算するため BufferedInputStream を使用
		try (java.io.BufferedInputStream in = new java.io.BufferedInputStream(Files.newInputStream(jsonlFile), BUFFER_SIZE)) {
			final long skipped = in.skip(startOffset);
			if (skipped < startOffset) {
				log.warn("Skipped {} < requested {} (file may have truncated) {}", skipped, startOffset, jsonlFile);
			}

			java.io.ByteArrayOutputStream buffer = new java.io.ByteArrayOutputStream(4096);
			int b;
			long lineCount = 0;
			boolean eof = true;

			while ((b = in.read()) != -1) {
				bytesRead++;
				if (b == '\n') {
					String line = buffer.toString(StandardCharsets.UTF_8.name());
					buffer.reset();

					// \r\n の \r を除去
					if (!line.isEmpty() && line.charAt(line.length() - 1) == '\r') {
						line = line.substring(0, line.length() - 1);
					}

					if (line.isEmpty()) continue;

					lineCount++;

					try {
						final MetricsFileLine obj = parseLine(line);
						if (obj != null) {
							lines.add(obj);
						}
					} catch (JsonProcessingException e) {
						final boolean looksTruncated = line.length() < 10 ||
								!line.trim().startsWith("{") ||
								!line.trim().endsWith("}");

						if (looksTruncated) {
							if (log.isDebugEnabled()) {
								log.debug("Skip incomplete JSON fragment near offset {} (len={}): {}",
										bytesRead, line.length(),
										line.substring(0, Math.min(80, line.length())));
							}
							continue;
						}

						final String excerpt = line.length() > 160 ? line.substring(0, 160) + "..." : line;
						log.error("JSON parse error (corrupted?) near offset {}: {} line={}",
								bytesRead, e.getOriginalMessage(), excerpt);
					}

					if (lineCount % PROGRESS_INTERVAL == 0 ||
							System.currentTimeMillis() - lastLogTime >= PROGRESS_INTERVAL_MS) {
						log.debug("reading {} lines={} offset={}MB", jsonlFile.getFileName(),
								lineCount, bytesRead / 1_000_000);
						lastLogTime = System.currentTimeMillis();
					}

					if (maxLines > 0 && lineCount >= maxLines) {
						eof = false;
						break;
					}
				} else {
					buffer.write(b);
				}
			}

			final long tEnd = System.currentTimeMillis();

			log.debug("parse done file={} lines={} readBytes={}MB duration={}ms eof={}",
					jsonlFile.getFileName(),
					lineCount, bytesRead / 1_000_000,
					(tEnd - tStart), eof);

			return new MetricsFileBlock(startOffset, bytesRead, lines, fileLastModified, eof);

		} catch (IOException e) {
			log.error("Error reading metrics file {}: {}", jsonlFile, e.getMessage());
			throw e;
		}
	}
}
