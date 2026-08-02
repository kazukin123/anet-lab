package io.github.kazukin123.anetlab.metricsviewer.service;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.sql.SQLException;

import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsCacheDatabase;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsCacheDatabase.CachePreparation;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsCacheDatabase.IngestState;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsSource;

/**
 * Metricsマスタから改行終端済みの完全行だけをblock単位で読み出す。
 * 未終端行はconsumerへ渡さず、commit可能offsetも最後の完全行から先へ進めない。
 * logical EOFはrawでは取得時点のsource末尾、gzipでは圧縮streamのEOFを表し、
 * その末尾が正常かcorruptかはsource固有の{@link #validate(ReadResult)}で判定する。
 */
interface SourceReader extends AutoCloseable {

	static SourceReader forSource(
			Path runDir,
			MetricsSource source,
			GzipInputSessions gzipSessions) {
		if ("jsonl.gz".equals(source.kind())) {
			return new GzipSessionReader(runDir, source, gzipSessions);
		}
		return new RawFileReader(runDir, source);
	}

	/**
	 * ソース同一性検査と入力準備を連続して行う。
	 * gzipではactive session確認、cache準備、session取得の順序をこの境界内で守る。
	 */
	Preparation prepare(MetricsCacheDatabase database) throws IOException, SQLException;

	int readByte() throws IOException;

	/**
	 * 現在までに消費したsource byte数を返す。
	 * rawでは非圧縮byte、gzipでは圧縮streamが消費したbyteを表す。
	 */
	long consumedSourceBytes();

	long initialCommittedSourceOffset();

	boolean reachedLogicalEnd(long consumedSourceBytes);

	default ReadResult readCompleteLines(int maxBlockLines, LineConsumer consumer)
			throws Exception {
		// 未終端行をblock外へ公開せず、改行まで読めた行だけをtransactionへ渡す。
		final ByteArrayOutputStream lineBuffer = new ByteArrayOutputStream(4096);
		long consumedSourceBytes = consumedSourceBytes();
		long committedSourceOffset = initialCommittedSourceOffset();
		int completedLines = 0;
		int value = -1;
		while (completedLines < maxBlockLines
				&& !reachedLogicalEnd(consumedSourceBytes)
				&& (value = readByte()) >= 0) {
			consumedSourceBytes = consumedSourceBytes();
			if (value == '\n') {
				completedLines++;
				committedSourceOffset = consumedSourceBytes;
				String line = lineBuffer.toString(StandardCharsets.UTF_8);
				lineBuffer.reset();
				if (line.endsWith("\r")) line = line.substring(0, line.length() - 1);
				consumer.accept(line, consumedSourceBytes);
			} else {
				lineBuffer.write(value);
			}
		}
		consumedSourceBytes = consumedSourceBytes();
		return new ReadResult(
				consumedSourceBytes,
				committedSourceOffset,
				completedLines,
				lineBuffer.size(),
				value < 0 || reachedLogicalEnd(consumedSourceBytes));
	}

	/** 論理EOFと未終端bytesの組合せがsource固有の終端条件を満たすか検証する。 */
	default void validate(ReadResult result) throws GzipCorruptException {
	}

	void finish(IngestState state);

	void fail();

	boolean treatsBlockIOExceptionAsCorrupt();

	boolean marksDatabaseFailureAsSourceReadError();

	@Override
	void close() throws IOException;

	record Preparation(CachePreparation cache, boolean readRequired) {
	}

	record ReadResult(
			long consumedSourceBytes,
			long committedSourceOffset,
			int completedLines,
			int trailingBytes,
			boolean logicalEof) {
	}

	@FunctionalInterface
	interface LineConsumer {
		void accept(String line, long sourceOffset) throws Exception;
	}

	final class GzipCorruptException extends IOException {
		GzipCorruptException(String message) {
			super(message);
		}

		GzipCorruptException(String message, IOException cause) {
			super(message, cause);
		}
	}
}
