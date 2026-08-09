package io.github.kazukin123.anetlab.metricsviewer.service;

import java.sql.ResultSet;
import java.sql.SQLException;

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

	static final int LOD_FACTOR = 16;
	static final int POINTS_PER_LOD_BUCKET = 3;

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

	static LodBucket fromLodRow(ResultSet result, int level) throws SQLException {
		final long bucket = result.getLong("bucket");
		final long ordinalFrom = Math.multiplyExact(bucket, widthForLevel(level));
		final long count = result.getLong("cnt");
		return new LodBucket(
				ordinalFrom,
				ordinalFrom + count,
				count,
				result.getLong("step_first"),
				result.getLong("step_last"),
				result.getLong("min_ordinal"),
				result.getLong("min_step"),
				result.getDouble("vmin"),
				result.getLong("max_ordinal"),
				result.getLong("max_step"),
				result.getDouble("vmax"),
				result.getDouble("vmean"),
				result.getDouble("vlast"));
	}

	static long widthForLevel(int level) {
		long width = 1L;
		for (int i = 0; i < level; i++) width = Math.multiplyExact(width, LOD_FACTOR);
		return width;
	}
}
