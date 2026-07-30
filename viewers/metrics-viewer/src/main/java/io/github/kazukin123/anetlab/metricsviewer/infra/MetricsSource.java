package io.github.kazukin123.anetlab.metricsviewer.infra;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.nio.file.attribute.BasicFileAttributes;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.Optional;

public record MetricsSource(Path path, String kind, long size, long modifiedTime) {

	private static final int FINGERPRINT_BYTES = 64 * 1024;

	public static Optional<MetricsSource> select(Path runDir) throws IOException {
		final Path jsonl = runDir.resolve("metrics.jsonl");
		final Path gzip = runDir.resolve("metrics.jsonl.gz");
		final Path selected;
		final String kind;
		if (Files.isRegularFile(jsonl)) {
			selected = jsonl;
			kind = "jsonl";
		} else if (Files.isRegularFile(gzip)) {
			selected = gzip;
			kind = "jsonl.gz";
		} else {
			return Optional.empty();
		}
		final BasicFileAttributes attributes =
				Files.readAttributes(selected, BasicFileAttributes.class);
		return Optional.of(new MetricsSource(
				selected,
				kind,
				attributes.size(),
				attributes.lastModifiedTime().toMillis()));
	}

	public String headSha256() throws IOException {
		return sha256(0L, Math.min(size, FINGERPRINT_BYTES));
	}

	public String headSha256(long sourceSizeAtFingerprint) throws IOException {
		return sha256(0L, Math.min(Math.min(size, sourceSizeAtFingerprint), FINGERPRINT_BYTES));
	}

	public String sha256Before(long offset) throws IOException {
		final long end = Math.max(0L, Math.min(offset, size));
		final long start = Math.max(0L, end - FINGERPRINT_BYTES);
		return sha256(start, end - start);
	}

	private String sha256(long offset, long length) throws IOException {
		final MessageDigest digest;
		try {
			digest = MessageDigest.getInstance("SHA-256");
		} catch (NoSuchAlgorithmException e) {
			throw new IllegalStateException("SHA-256 is unavailable", e);
		}

		try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
			channel.position(offset);
			final ByteBuffer buffer = ByteBuffer.allocate(8192);
			long remaining = length;
			while (remaining > 0L) {
				buffer.clear();
				buffer.limit((int) Math.min(buffer.capacity(), remaining));
				final int read = channel.read(buffer);
				if (read < 0) break;
				buffer.flip();
				digest.update(buffer);
				remaining -= read;
			}
		}
		return HexFormat.of().formatHex(digest.digest());
	}
}
