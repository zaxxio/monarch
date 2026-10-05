package org.monarch.nn.learningrate;

public class OneCycleLR implements LRSchedule {

    private final double maxLR, minLR;
    private final int totalEpochs;

    public OneCycleLR(double maxLR, double minLR, int totalEpochs) {
        this.maxLR = maxLR;
        this.minLR = minLR;
        this.totalEpochs = Math.max(1, totalEpochs);
    }

    @Override
    public double getLR(int epoch, int step) {
        int half = totalEpochs / 2;
        if (epoch < half) {
            // Up phase
            double t = (double) epoch / half;
            return minLR + (maxLR - minLR) * t;
        } else {
            // Down phase
            double t = (double)(epoch - half) / (totalEpochs - half);
            return maxLR - (maxLR - minLR) * t;
        }
    }

    @Override public String name() { return "OneCycle"; }
}