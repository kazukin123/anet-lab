package io.github.kazukin123.anetlab.metricsviewer.service;

import java.util.stream.Collectors;
import java.util.Objects;
import java.io.BufferedInputStream;
import java.io.FilterInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.BasicFileAttributes;
import java.nio.file.attribute.FileTime;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import java.util.zip.GZIPInputStream;

import io.github.kazukin123.anetlab.metricsviewer.infra.MetricsSource;

public class GzipInputSessions {

	static final int MAX_STREAMS = 16;
	static final int MAX_BACKGROUND_STREAMS = 4;
	private Set<Path> priorityRuns = Set.of();

	synchronized void replacePriority(Set<Path> runs) {
		priorityRuns = Set.copyOf(runs);
	}

	synchronized String admissionReason(Path run) {
		if (sessions.containsKey(run))
			return null;
		if (sessions.size() >= MAX_STREAMS)
			return "gzip_stream_limit";
		if (!priorityRuns.contains(run)
				&& sessions.keySet().stream().filter(p -> !priorityRuns.contains(p)).count() >= MAX_BACKGROUND_STREAMS)
			return "background_gzip_limit";
		return null;
	}

	static final class ResourceUnavailableException extends IOException {
		private final String reason;
		ResourceUnavailableException(String reason) {
			super(reason);
			this.reason = reason;
		}
		String reason() {
			return reason;
		}
	}

	synchronized Set<Path> activeRuns() {
		return Set.copyOf(sessions.keySet());
	}

	private final Map<Path, Session> sessions = new HashMap<>();

	synchronized boolean hasSession(Path runDir, MetricsSource source) {
		final Session current = sessions.get(runDir.toAbsolutePath().normalize());
		return current != null && current.matchesSource(source);
	}

	synchronized boolean matchesGeneration(Path runDir, MetricsSource source, String generation) {
		final Session current = sessions.get(runDir.toAbsolutePath().normalize());
		return current != null && current.matches(source, generation);
	}

	synchronized Session acquire(Path runDir, MetricsSource source, String generation)
			throws IOException {
		final Path key = runDir.toAbsolutePath().normalize();
		final Session current = sessions.get(key);
		if (current != null && current.matches(source, generation)) return current;
		if (current != null) {
			sessions.remove(key);
			current.close();
		}
		final String reason = admissionReason(key);
		if (reason != null)
			throw new ResourceUnavailableException(reason);

		final BasicFileAttributes attributes = Files.readAttributes(source.path(), BasicFileAttributes.class);
		final CountingInputStream counter = new CountingInputStream(Files.newInputStream(source.path()));
		try {
			final GZIPInputStream gzip = new GZIPInputStream(
					new BufferedInputStream(counter, 64 * 1024),
					64 * 1024);
			final Session created = new Session(source, generation, counter, gzip, attributes);
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
				.collect(Collectors.toUnmodifiableSet());
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
		private final Object fileKey;
		private final FileTime fileModified;
		private final FileTime fileCreated;
		final SourceReader.InputBuffer buffer = new SourceReader.InputBuffer();

		private Session(
				MetricsSource source,
				String generation,
				CountingInputStream counter,
				GZIPInputStream input, BasicFileAttributes attributes) {
			this.source = source;
			this.generation = generation;
			this.counter = counter;
			this.input = input;
			fileKey = attributes.fileKey();
			fileModified = attributes.lastModifiedTime();
			fileCreated = attributes.creationTime();
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
					&& source.modifiedTime() == candidate.modifiedTime() && matchesFileIdentity(candidate.path());
		}

		private boolean matchesFileIdentity(Path file) {
			// 同じサイズ・mtimeでもファイル置換は旧handleと異なる入力として扱う。
			try {
				final BasicFileAttributes attributes = Files.readAttributes(file, BasicFileAttributes.class);
				return Objects.equals(fileKey, attributes.fileKey())
						&& fileModified.equals(attributes.lastModifiedTime())
						&& fileCreated.equals(attributes.creationTime());
			}
			catch (IOException e) {
				return false;
			}
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
