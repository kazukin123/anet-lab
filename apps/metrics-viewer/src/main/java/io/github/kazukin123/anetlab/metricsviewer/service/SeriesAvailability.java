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

	String externalName() {
		return externalName;
	}
}
