package io.github.kazukin123.anetlab.metricsviewer.service;

import java.io.BufferedInputStream;
import java.io.FilterInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import java.util.zip.GZIPInputStream;

import jakarta.annotation.PreDestroy;

import org.springframework.stereotype.Component;

import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsSource;

@Component
public class GzipInputSessions {

	private final Map<Path, Session> sessions = new HashMap<>();

	synchronized boolean hasSession(Path runDir, MetricsSource source) {
		final Session current = sessions.get(runDir.toAbsolutePath().normalize());
		return current != null && current.matchesSource(source);
	}

	synchronized Session acquire(Path runDir, MetricsSource source, String generation)
			throws IOException {
		final Path key = runDir.toAbsolutePath().normalize();
		final Session current = sessions.get(key);
		if (current != null && current.matches(source, generation)) return current;
		if (current != null) current.close();

		final CountingInputStream counter = new CountingInputStream(Files.newInputStream(source.path()));
		try {
			final GZIPInputStream gzip = new GZIPInputStream(
					new BufferedInputStream(counter, 64 * 1024),
					64 * 1024);
			final Session created = new Session(source, generation, counter, gzip);
			sessions.put(key, created);
			return created;
		} catch (IOException e) {
			counter.close();
			sessions.remove(key);
			throw e;
		}
	}

	synchronized void close(Path runDir) {
		final Session session = sessions.remove(runDir.toAbsolutePath().normalize());
		if (session == null) return;
		try {
			session.close();
		} catch (IOException ignored) {
			// 終了処理では元の取り込みerrorを優先する。
		}
	}

	synchronized void retainRuns(Set<Path> runDirs) {
		final Set<Path> normalized = runDirs.stream()
				.map(path -> path.toAbsolutePath().normalize())
				.collect(java.util.stream.Collectors.toUnmodifiableSet());
		final Iterator<Map.Entry<Path, Session>> iterator = sessions.entrySet().iterator();
		while (iterator.hasNext()) {
			final Map.Entry<Path, Session> entry = iterator.next();
			if (normalized.contains(entry.getKey())) continue;
			try {
				entry.getValue().close();
			} catch (IOException ignored) {
				// 消失Runのhandle解放を継続する。
			}
			iterator.remove();
		}
	}

	@PreDestroy
	synchronized void closeAll() {
		for (Session session : sessions.values()) {
			try {
				session.close();
			} catch (IOException ignored) {
				// process終了時は残りのsessionも閉じる。
			}
		}
		sessions.clear();
	}

	static final class Session implements AutoCloseable {
		private final MetricsSource source;
		private final String generation;
		private final CountingInputStream counter;
		private final GZIPInputStream input;

		private Session(
				MetricsSource source,
				String generation,
				CountingInputStream counter,
				GZIPInputStream input) {
			this.source = source;
			this.generation = generation;
			this.counter = counter;
			this.input = input;
		}

		InputStream input() {
			return input;
		}

		long compressedBytes() {
			return counter.count();
		}

		private boolean matches(MetricsSource candidate, String candidateGeneration) {
			return matchesSource(candidate) && generation.equals(candidateGeneration);
		}

		private boolean matchesSource(MetricsSource candidate) {
			return source.path().toAbsolutePath().normalize()
							.equals(candidate.path().toAbsolutePath().normalize())
					&& source.size() == candidate.size()
					&& source.modifiedTime() == candidate.modifiedTime();
		}

		@Override
		public void close() throws IOException {
			input.close();
		}
	}

	private static final class CountingInputStream extends FilterInputStream {
		private long count;

		private CountingInputStream(InputStream input) {
			super(input);
		}

		@Override
		public int read() throws IOException {
			final int value = super.read();
			if (value >= 0) count++;
			return value;
		}

		@Override
		public int read(byte[] buffer, int offset, int length) throws IOException {
			final int read = super.read(buffer, offset, length);
			if (read > 0) count += read;
			return read;
		}

		private long count() {
			return count;
		}
	}
}
