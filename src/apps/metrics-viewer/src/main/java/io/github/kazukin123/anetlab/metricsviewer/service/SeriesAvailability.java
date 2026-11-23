package io.github.kazukin123.anetlab.metricsviewer.service;

/**
 * externalNameはHTTP JSONのavailabilityで公開する安定表現であり、互換性なく変更しない。
 */
enum SeriesAvailability {
	OK("ok"),
	PENDING("pending"),
	NOT_FOUND("not_found"),
	EMPTY("empty");

	private final String externalName;

	SeriesAvailability(String externalName) {
		this.externalName = externalName;
	}

	// tag の欠落と点のない範囲は、取り込みが完了して初めて確定する。
	static SeriesAvailability forRange(boolean stillIngesting, boolean hasTag, boolean hasPoints) {
		if (hasPoints) return OK;
		if (stillIngesting) return PENDING;
		return hasTag ? EMPTY : NOT_FOUND;
	}

	String externalName() {
		return externalName;
	}
}
