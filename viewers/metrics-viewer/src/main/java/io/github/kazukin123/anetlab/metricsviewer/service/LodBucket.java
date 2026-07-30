package io.github.kazukin123.anetlab.metricsviewer.service;

record LodBucket(
		long ordinalFrom,
		long ordinalTo,
		long count,
		long stepFirst,
		long stepLast,
		long minOrdinal,
		long minStep,
		double minValue,
		long maxOrdinal,
		long maxStep,
		double maxValue,
		double mean,
		double lastValue) {

	static LodBucket point(long ordinal, long step, double value) {
		return new LodBucket(
				ordinal,
				ordinal + 1L,
				1L,
				step,
				step,
				ordinal,
				step,
				value,
				ordinal,
				step,
				value,
				value,
				value);
	}
}
