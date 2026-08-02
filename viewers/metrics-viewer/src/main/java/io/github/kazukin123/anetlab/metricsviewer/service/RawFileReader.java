package io.github.kazukin123.anetlab.metricsviewer.service;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.SQLException;

import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsCacheDatabase;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsCacheDatabase.CachePreparation;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsCacheDatabase.IngestState;
import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsSource;

final class RawFileReader implements SourceReader {

	private final Path runDir;
	private final MetricsSource source;
	private InputStream input;
	private long consumedSourceBytes;
	private long initialCommittedSourceOffset;

	RawFileReader(Path runDir, MetricsSource source) {
		this.runDir = runDir;
		this.source = source;
	}

	@Override
	public Preparation prepare(MetricsCacheDatabase database) throws IOException, SQLException {
		final CachePreparation preparation = database.prepare(runDir, source, false);
		initialCommittedSourceOffset = preparation.committedOffset();
		consumedSourceBytes = initialCommittedSourceOffset;
		if (preparation.sourceUnchangedError()) {
			return new Preparation(preparation, false);
		}

		// source snapshot取得後の追記は次blockへ送り、この時点の末尾だけを読む。
		input = new BufferedInputStream(Files.newInputStream(source.path()), 64 * 1024);
		input.skipNBytes(initialCommittedSourceOffset);
		return new Preparation(preparation, true);
	}

	@Override
	public int readByte() throws IOException {
		final int value = input.read();
		if (value >= 0) consumedSourceBytes++;
		return value;
	}

	@Override
	public long consumedSourceBytes() {
		return consumedSourceBytes;
	}

	@Override
	public long initialCommittedSourceOffset() {
		return initialCommittedSourceOffset;
	}

	@Override
	public boolean reachedLogicalEnd(long consumedBytes) {
		return consumedBytes >= source.size();
	}

	@Override
	public void finish(IngestState state) {
	}

	@Override
	public void fail() {
	}

	@Override
	public boolean treatsBlockIOExceptionAsCorrupt() {
		return false;
	}

	@Override
	public boolean marksDatabaseFailureAsSourceReadError() {
		return true;
	}

	@Override
	public void close() throws IOException {
		if (input != null) input.close();
	}
}
