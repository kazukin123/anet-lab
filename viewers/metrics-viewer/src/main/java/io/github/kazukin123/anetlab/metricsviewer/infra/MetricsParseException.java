package io.github.kazukin123.anetlab.metricsviewer.infra;

import java.io.IOException;

/**
 * Exception thrown when a JSONL metric line cannot be parsed correctly.
 * Contains the byte offset and raw line text for debugging.
 */
public class MetricsParseException extends IOException {
	private final long offset;
	private final String rawLine;

	public MetricsParseException(String message, long offset, String rawLine, Throwable cause) {
		super(message, cause);
		this.offset = offset;
		this.rawLine = rawLine;
	}

	public long getOffset() {
		return offset;
	}

	public String getRawLine() {
		return rawLine;
	}
}
