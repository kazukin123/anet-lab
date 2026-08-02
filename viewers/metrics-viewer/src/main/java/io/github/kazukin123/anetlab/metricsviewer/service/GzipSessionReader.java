package io.github.kazukin123.anetlab.metricsviewer.service;

import java.io.IOException;
import java.nio.file.Path;
import java.sql.SQLException;

import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsCacheDatabase;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsCacheDatabase.CachePreparation;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsCacheDatabase.IngestState;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsSource;
import io.github.kazukin123.anetlab.metricsviewer.service.SourceReader.GzipCorruptException;

/**
 * 同じgzip streamの展開済みbufferをblock間で引き継ぐreader。
 * 単一writer threadが1 Runのblockを順番に処理し、同じsessionを並行読取しない前提とする。
 */
final class GzipSessionReader implements SourceReader {

	private final Path runDir;
	private final MetricsSource source;
	private final GzipInputSessions sessions;
	private GzipInputSessions.Session session;
	private long initialCommittedSourceOffset;

	GzipSessionReader(Path runDir, MetricsSource source, GzipInputSessions sessions) {
		this.runDir = runDir;
		this.source = source;
		this.sessions = sessions;
	}

	@Override
	public Preparation prepare(MetricsCacheDatabase database) throws IOException, SQLException {
		// 展開済みbufferを再利用できる場合だけ、converting cacheを同じ世代で継続する。
		final CachePreparation preparation = database.prepare(
				runDir,
				source,
				sessions.hasSession(runDir, source));
		initialCommittedSourceOffset = preparation.committedOffset();
		if (preparation.sourceUnchangedError()
				|| preparation.state() == IngestState.READY) {
			sessions.close(runDir);
			return new Preparation(preparation, false);
		}
		try {
			session = sessions.acquire(runDir, source, preparation.generation());
		} catch (IOException e) {
			throw new GzipCorruptException(e.getMessage(), e);
		}
		return new Preparation(preparation, true);
	}

	@Override
	public int readByte() throws IOException {
		try {
			return session.input().read();
		} catch (IOException e) {
			throw new GzipCorruptException(e.getMessage(), e);
		}
	}

	@Override
	public long consumedSourceBytes() {
		return session.compressedBytes();
	}

	@Override
	public long initialCommittedSourceOffset() {
		return initialCommittedSourceOffset;
	}

	@Override
	public boolean reachedLogicalEnd(long consumedBytes) {
		return false;
	}

	@Override
	public void validate(ReadResult result) throws GzipCorruptException {
		// raw JSONLの未終端末尾は次回追記まで保持できるが、gzipはimmutableなので
		// stream EOFに未終端行が残る場合は追記待ちにせずcorruptとする。
		if (result.logicalEof() && result.trailingBytes() != 0) {
			throw new GzipCorruptException("gzip ended with an unterminated JSON line");
		}
	}

	@Override
	public void finish(IngestState state) {
		if (state == IngestState.READY) sessions.close(runDir);
	}

	@Override
	public void fail() {
		sessions.close(runDir);
	}

	@Override
	public boolean treatsBlockIOExceptionAsCorrupt() {
		return true;
	}

	@Override
	public boolean marksDatabaseFailureAsSourceReadError() {
		return false;
	}

	@Override
	public void close() {
		// converting中の展開済みbufferはGzipInputSessionsが次blockまで所有する。
	}
}
