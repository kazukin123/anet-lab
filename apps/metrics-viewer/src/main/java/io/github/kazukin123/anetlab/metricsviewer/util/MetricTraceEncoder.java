package io.github.kazukin123.anetlab.metricsviewer.util;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Collections;
import java.util.List;

public final class MetricTraceEncoder {

	// float 250,000要素を1 MBのbinary chunkとして転送する。
	private static final int CHUNK_SIZE = 250_000;

	private MetricTraceEncoder() {
	}

	/** double配列をBase64文字列のリストに変換する。 */
	public static List<String> encodeDoubleArray(double[] values) {
		if (values == null || values.length == 0) return Collections.emptyList();

		final List<String> chunks = new ArrayList<>();
		for (int offset = 0; offset < values.length; offset += CHUNK_SIZE) {
			final int count = Math.min(CHUNK_SIZE, values.length - offset);
			final byte[] bytes = new byte[count * Double.BYTES];
			ByteBuffer.wrap(bytes)
					.order(ByteOrder.LITTLE_ENDIAN)
					.asDoubleBuffer()
					.put(values, offset, count);
			chunks.add(Base64.getEncoder().encodeToString(bytes));
		}
		return chunks;
	}

	/** float配列をBase64文字列のリストに変換する。 */
	public static List<String> encodeFloatArray(float[] values) {
		if (values == null || values.length == 0) return Collections.emptyList();

		final List<String> chunks = new ArrayList<>();
		for (int offset = 0; offset < values.length; offset += CHUNK_SIZE) {
			final int count = Math.min(CHUNK_SIZE, values.length - offset);
			final byte[] bytes = new byte[count * Float.BYTES];
			ByteBuffer.wrap(bytes)
					.order(ByteOrder.LITTLE_ENDIAN)
					.asFloatBuffer()
					.put(values, offset, count);
			chunks.add(Base64.getEncoder().encodeToString(bytes));
		}
		return chunks;
	}
}
