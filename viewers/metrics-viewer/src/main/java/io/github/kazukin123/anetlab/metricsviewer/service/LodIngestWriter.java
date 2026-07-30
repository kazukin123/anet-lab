package io.github.kazukin123.anetlab.metricsviewer.service;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.List;

import org.springframework.stereotype.Component;

@Component
public class LodIngestWriter {

	State restore(Connection connection, long tagId, long pointCount) throws SQLException {
		final State state = new State();
		long childCount = pointCount;
		int childLevel = 0;
		state.ensureLevel(0);

		// 親bucketになっていない末尾最大15子だけを各levelから復元する。
		while (childCount > 0L) {
			final int remainder = (int) (childCount % 16L);
			state.ensureLevel(childLevel);
			if (remainder > 0) {
				final long firstChild = childCount - remainder;
				if (childLevel == 0) {
					restoreL0(connection, tagId, firstChild, state.pending(childLevel));
				} else {
					restoreLod(
							connection,
							tagId,
							childLevel,
							firstChild,
							state.pending(childLevel));
				}
			}
			childCount /= 16L;
			childLevel++;
		}
		return state;
	}

	void append(
			InsertSession insertSession,
			long tagId,
			State state,
			LodBucket point) throws SQLException {
		appendChild(insertSession, tagId, state, 0, point);
	}

	InsertSession openInsertSession(Connection connection) throws SQLException {
		return new InsertSession(connection);
	}

	private static void restoreL0(
			Connection connection,
			long tagId,
			long firstOrdinal,
			List<LodBucket> pending) throws SQLException {
		try (PreparedStatement statement = connection.prepareStatement("""
				SELECT ordinal, step, value
				FROM scalars
				WHERE tag_id=? AND ordinal>=?
				ORDER BY ordinal
				""")) {
			statement.setLong(1, tagId);
			statement.setLong(2, firstOrdinal);
			try (ResultSet result = statement.executeQuery()) {
				while (result.next()) {
					pending.add(LodBucket.point(
							result.getLong(1),
							result.getLong(2),
							result.getDouble(3)));
				}
			}
		}
	}

	private static void restoreLod(
			Connection connection,
			long tagId,
			int level,
			long firstBucket,
			List<LodBucket> pending) throws SQLException {
		try (PreparedStatement statement = connection.prepareStatement("""
				SELECT bucket, cnt, step_first, step_last,
				       min_ordinal, min_step, vmin,
				       max_ordinal, max_step, vmax, vmean, vlast
				FROM scalars_lod
				WHERE tag_id=? AND level=? AND bucket>=?
				ORDER BY bucket
				""")) {
			statement.setLong(1, tagId);
			statement.setInt(2, level);
			statement.setLong(3, firstBucket);
			try (ResultSet result = statement.executeQuery()) {
				while (result.next()) pending.add(fromLodRow(result, level));
			}
		}
	}

	private static LodBucket fromLodRow(ResultSet result, int level) throws SQLException {
		final long bucket = result.getLong(1);
		final long ordinalFrom = Math.multiplyExact(bucket, widthForLevel(level));
		final long count = result.getLong(2);
		return new LodBucket(
				ordinalFrom,
				ordinalFrom + count,
				count,
				result.getLong(3),
				result.getLong(4),
				result.getLong(5),
				result.getLong(6),
				result.getDouble(7),
				result.getLong(8),
				result.getLong(9),
				result.getDouble(10),
				result.getDouble(11),
				result.getDouble(12));
	}

	private static void appendChild(
			InsertSession insertSession,
			long tagId,
			State state,
			int childLevel,
			LodBucket child) throws SQLException {
		state.ensureLevel(childLevel);
		final List<LodBucket> pending = state.pending(childLevel);
		pending.add(child);
		if (pending.size() < 16) return;

		final LodBucket parent = combine(pending);
		pending.clear();
		final int parentLevel = childLevel + 1;
		final long bucket = parent.ordinalFrom() / widthForLevel(parentLevel);
		insertSession.insert(tagId, parentLevel, bucket, parent);
		appendChild(insertSession, tagId, state, parentLevel, parent);
	}

	private static LodBucket combine(List<LodBucket> children) {
		if (children.size() != 16) {
			throw new IllegalArgumentException("LOD parent requires exactly 16 children");
		}
		final LodBucket first = children.get(0);
		final LodBucket last = children.get(children.size() - 1);
		long count = 0L;
		double weightedSum = 0.0;
		LodBucket minChild = first;
		LodBucket maxChild = first;
		for (LodBucket child : children) {
			count += child.count();
			weightedSum += child.mean() * child.count();
			if (child.minValue() < minChild.minValue()
					|| (child.minValue() == minChild.minValue()
							&& child.minOrdinal() < minChild.minOrdinal())) {
				minChild = child;
			}
			if (child.maxValue() > maxChild.maxValue()
					|| (child.maxValue() == maxChild.maxValue()
							&& child.maxOrdinal() < maxChild.maxOrdinal())) {
				maxChild = child;
			}
		}
		return new LodBucket(
				first.ordinalFrom(),
				last.ordinalTo(),
				count,
				first.stepFirst(),
				last.stepLast(),
				minChild.minOrdinal(),
				minChild.minStep(),
				minChild.minValue(),
				maxChild.maxOrdinal(),
				maxChild.maxStep(),
				maxChild.maxValue(),
				weightedSum / count,
				last.lastValue());
	}

	private static long widthForLevel(int level) {
		long width = 1L;
		for (int i = 0; i < level; i++) width = Math.multiplyExact(width, 16L);
		return width;
	}

	static final class State {
		private final List<List<LodBucket>> pendingByLevel = new ArrayList<>();

		private void ensureLevel(int level) {
			while (pendingByLevel.size() <= level) pendingByLevel.add(new ArrayList<>());
		}

		private List<LodBucket> pending(int level) {
			return pendingByLevel.get(level);
		}
	}

	static final class InsertSession implements AutoCloseable {
		private final PreparedStatement statement;

		private InsertSession(Connection connection) throws SQLException {
			statement = connection.prepareStatement("""
					INSERT INTO scalars_lod(
					  tag_id, level, bucket, cnt, step_first, step_last,
					  min_ordinal, min_step, vmin,
					  max_ordinal, max_step, vmax, vmean, vlast
					) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
					""");
		}

		private void insert(long tagId, int level, long bucket, LodBucket value)
				throws SQLException {
			statement.setLong(1, tagId);
			statement.setInt(2, level);
			statement.setLong(3, bucket);
			statement.setLong(4, value.count());
			statement.setLong(5, value.stepFirst());
			statement.setLong(6, value.stepLast());
			statement.setLong(7, value.minOrdinal());
			statement.setLong(8, value.minStep());
			statement.setDouble(9, value.minValue());
			statement.setLong(10, value.maxOrdinal());
			statement.setLong(11, value.maxStep());
			statement.setDouble(12, value.maxValue());
			statement.setDouble(13, value.mean());
			statement.setDouble(14, value.lastValue());
			statement.executeUpdate();
		}

		@Override
		public void close() throws SQLException {
			statement.close();
		}
	}
}
