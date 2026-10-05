package org.monarch.nn.learningrate;

public class LinearWarmupLR implements LRSchedule {

    private final double baseLR;
    private final int warmupEpochs;

    public LinearWarmupLR(double baseLR, int warmupEpochs) {
        this.baseLR = baseLR;
        this.warmupEpochs = Math.max(1, warmupEpochs);
    }

    @Override
    public double getLR(int epoch, int step) {
        if (epoch >= warmupEpochs) return baseLR;
        return baseLR * (epoch + 1.0) / warmupEpochs;
    }

    @Override public String name() { return "LinearWarmup"; }
}