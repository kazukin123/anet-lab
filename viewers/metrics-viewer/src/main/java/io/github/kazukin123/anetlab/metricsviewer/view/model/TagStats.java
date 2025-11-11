package io.github.kazukin123.anetlab.metricsviewer.view.model;

public class TagStats {
    private int minStep = Integer.MAX_VALUE;
    private int maxStep = 0;
    private int count = 0;
    private double lastValue = Double.NaN;
    private double sum = 0.0;
    private double sumSq = 0.0;
    private long updatedAt = 0L;

    public synchronized void record(int step, float value) {
        if (step < minStep) minStep = step;
        if (step > maxStep) maxStep = step;
        count++;
        lastValue = value;
        sum += value;
        sumSq += value * value;
        updatedAt = System.currentTimeMillis();
    }

    public long getMaxStep() { return maxStep; }
    public long getMinStep() { return minStep == Long.MAX_VALUE ? 0L : minStep; }
    public long getCount() { return count; }
    public double getLastValue() { return lastValue; }
    public double getMean() { return count == 0 ? Double.NaN : sum / count; }
    public double getVariance() { return count <= 1 ? Double.NaN : (sumSq / count) - Math.pow(getMean(), 2); }
    public long getUpdatedAt() { return updatedAt; }
}
