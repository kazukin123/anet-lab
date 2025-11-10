package io.github.kazukin123.anetlab.metricsviewer.view.model;

public class TagStats {
    private long minStep = Long.MAX_VALUE;
    private long maxStep = 0L;
    private long count = 0L;
    private double lastValue = Double.NaN;
    private double sum = 0.0;
    private double sumSq = 0.0;
    private long updatedAt = 0L;

    public synchronized void record(long step, double value) {
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
