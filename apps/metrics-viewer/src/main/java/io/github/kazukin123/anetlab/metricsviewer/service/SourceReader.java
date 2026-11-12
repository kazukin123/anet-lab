package io.github.kazukin123.anetlab.metricsviewer.service;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.sql.SQLException;
import java.util.function.BooleanSupplier;

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

	int readChunk(byte[] target) throws IOException;

	InputBuffer inputBuffer();

	default void advanceDecodedBytes(int count) {
	}

	/** gzipではsessionに属し、未消費の展開bytesをblock間で保持する。 */
	final class InputBuffer {
		final byte[] bytes = new byte[64 * 1024];
		int position;
		int limit;
		boolean eof;
	}

	/**
	 * 現在までに消費したsource byte数を返す。
	 * rawでは非圧縮byte、gzipでは圧縮streamが消費したbyteを表す。
	 */
	long consumedSourceBytes();

	long initialCommittedSourceOffset();

	boolean reachedLogicalEnd(long consumedSourceBytes);

	default ReadResult readCompleteLines(
			int maxBlockLines,
			BooleanSupplier yieldRequested,
			LineConsumer consumer)
			throws Exception {
		// chunk内の改行だけを探し、UTF-8は完全行になってからdecodeする。
		final InputBuffer buffer = inputBuffer();
		final ByteArrayOutputStream tail = new ByteArrayOutputStream(4096);
		long committed = initialCommittedSourceOffset();
		int lines = 0;
		while (lines < maxBlockLines) {
			if (buffer.position == buffer.limit) {
				final int read = readChunk(buffer.bytes);
				if (read < 0) {
					buffer.eof = true;
					break;
				}
				buffer.position = 0;
				buffer.limit = read;
			}
			final int from = buffer.position;
			int end = from;
			while (end < buffer.limit && buffer.bytes[end] != '\n')
				end++;
			final boolean complete = end < buffer.limit;
			buffer.position = complete ? end + 1 : end;
			advanceDecodedBytes(buffer.position - from);
			if (!complete) {
				tail.write(buffer.bytes, from, end - from);
				continue;
			}
			String line;
			if (tail.size() == 0)
				line = new String(buffer.bytes, from, end - from, StandardCharsets.UTF_8);
			else {
				tail.write(buffer.bytes, from, end - from);
				line = tail.toString(StandardCharsets.UTF_8);
				tail.reset();
			}
			if (line.endsWith("\r"))
				line = line.substring(0, line.length() - 1);
			lines++;
			committed = consumedSourceBytes();
			consumer.accept(line, committed);
			// 停止要求と時間予算は完全行境界でだけ評価する。
			if (yieldRequested.getAsBoolean())
				break;
		}
		return new ReadResult(consumedSourceBytes(), committed, lines, tail.size(),
				buffer.eof || reachedLogicalEnd(consumedSourceBytes()));
	}

	/** 論理EOFと未終端bytesの組合せがsource固有の終端条件を満たすか検証する。 */
	default void validate(ReadResult result) throws GzipCorruptException {
	}

	void finish(IngestState state);

	void fail();

	boolean treatsBlockIOExceptionAsCorrupt();

	boolean marksSourceReadFailureAsError();

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
