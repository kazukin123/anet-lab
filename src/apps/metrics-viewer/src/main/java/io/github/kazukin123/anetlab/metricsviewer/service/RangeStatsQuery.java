package io.github.kazukin123.anetlab.metricsviewer.service;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;

import io.github.kazukin123.anetlab.metricsviewer.view.model.TagStats;

final class RangeStatsQuery {

	private RangeStatsQuery() {
	}

	static TagStats summarize(Connection connection, MetricsQueryPlanner.TagInput tag,
			QueryExecution query) throws SQLException {
		long count = 0;
		long minStep = 0;
		long maxStep = 0;
		double min = Double.POSITIVE_INFINITY;
		double max = Double.NEGATIVE_INFINITY;
		double mean = 0;
		double m2 = 0;
		double last = 0;

		// 既存の範囲検索が求めた序数区間を順に読み、全点から母分散を逐次計算する。
		try (PreparedStatement statement = connection.prepareStatement("""
				SELECT step, value
				FROM scalars
				WHERE tag_id=? AND ordinal>=? AND ordinal<?
				ORDER BY ordinal
				""");
				StatementRegistration ignored = query.registerStatement(statement)) {
			statement.setLong(1, tag.tagId());
			statement.setLong(2, tag.ordinalFrom());
			statement.setLong(3, tag.ordinalTo());
			query.checkpoint();
			try (ResultSet rows = statement.executeQuery()) {
				while (rows.next()) {
					query.checkpoint();
					final long step = rows.getLong(1);
					final double value = rows.getDouble(2);
					if (count == 0) minStep = step;
					maxStep = step;
					min = Math.min(min, value);
					max = Math.max(max, value);
					final double delta = value - mean;
					mean += delta / ++count;
					m2 += delta * (value - mean);
					last = value;
				}
			}
		}

		// 同じ snapshot 内で求めた序数区間と実際に読んだ件数が一致することを確かめる。
		if (count != tag.ordinalTo() - tag.ordinalFrom()) {
			throw new SQLException("Unexpected L0 count for tag " + tag.tagId()
					+ ": count=" + count + " expected=" + (tag.ordinalTo() - tag.ordinalFrom()));
		}
		final double variance = Math.max(0.0, m2 / count);
		return TagStats.builder()
				.count(count).minStep(minStep).maxStep(maxStep)
				.minValue(min).maxValue(max).lastValue(last)
				.mean(mean).variance(variance).stdDev(Math.sqrt(variance))
				.build();
	}
}
