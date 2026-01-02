package io.github.kazukin123.anetlab.metricsviewer.util;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Base64;
import java.util.Collections;
import java.util.List;

public class MetricTraceEncoder {

    // 分割サイズ設定 (floatは4byteなので、250,000要素 = 1MBバイナリ = 約1.33MB Base64文字列)
    private static final int CHUNK_SIZE = 250_000;

    /** float配列をBase64文字列のリストに変換 */
    public static List<String> encodeFloatArray(float[] values) {
        if (values == null || values.length == 0) return Collections.emptyList();

        List<String> chunks = new ArrayList<>();
        int length = values.length;

        for (int i = 0; i < length; i += CHUNK_SIZE) {
            int end = Math.min(length, i + CHUNK_SIZE);
            int count = end - i;
            
            // 部分配列を切り出し
            byte[] bytes = new byte[count * 4];
            float[] part = new float[count];
            System.arraycopy(values, i, part, 0, count);

            ByteBuffer.wrap(bytes)
                    .order(ByteOrder.LITTLE_ENDIAN)
                    .asFloatBuffer()
                    .put(part);

            chunks.add(Base64.getEncoder().encodeToString(bytes));
        }
        return chunks;
    }

    /** int配列をBase64文字列のリストに変換 */
    public static List<String> encodeIntArray(int[] values) {
        if (values == null || values.length == 0) return Collections.emptyList();

        List<String> chunks = new ArrayList<>();
        int length = values.length;

        for (int i = 0; i < length; i += CHUNK_SIZE) {
            int end = Math.min(length, i + CHUNK_SIZE);
            int count = end - i;

            byte[] bytes = new byte[count * 4];
            int[] part = new int[count];
            System.arraycopy(values, i, part, 0, count);

            ByteBuffer.wrap(bytes)
                    .order(ByteOrder.LITTLE_ENDIAN)
                    .asIntBuffer()
                    .put(part);

            chunks.add(Base64.getEncoder().encodeToString(bytes));
        }
        return chunks;
    }
}