package io.github.kazukin123.anetlab.metricsviewer.util;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Base64;

public class MetricTraceEncoder {

	/** float配列をBase64文字列に変換 (Float32Array互換: LittleEndian) */
	public static String encodeFloatArray(float[] values) {
		if (values == null || values.length == 0) return "";
		final byte[] bytes = new byte[values.length * 4];
		ByteBuffer.wrap(bytes)
				.order(ByteOrder.LITTLE_ENDIAN) // ← 追加
				.asFloatBuffer()
				.put(values);
		return Base64.getEncoder().encodeToString(bytes);
	}

	/** int配列をBase64文字列に変換 (Int32Array互換: LittleEndian) */
	public static String encodeIntArray(int[] values) {
		if (values == null || values.length == 0) return "";
		final byte[] bytes = new byte[values.length * 4];
		ByteBuffer.wrap(bytes)
				.order(ByteOrder.LITTLE_ENDIAN) // ← 追加
				.asIntBuffer()
				.put(values);
		return Base64.getEncoder().encodeToString(bytes);
	}
}
