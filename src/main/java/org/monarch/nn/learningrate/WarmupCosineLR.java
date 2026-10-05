package org.monarch.nn.learningrate;

public class WarmupCosineLR implements LRSchedule {

    private final double baseLR, minLR;
    private final int warmupEpochs, totalEpochs;

    public WarmupCosineLR(double baseLR, double minLR, int warmupEpochs, int totalEpochs) {
        this.baseLR = baseLR;
        this.minLR  = minLR;
        this.warmupEpochs = Math.max(1, warmupEpochs);
        this.totalEpochs  = Math.max(warmupEpochs + 1, totalEpochs);
    }

    @Override
    public double getLR(int epoch, int step) {
        if (epoch < warmupEpochs) {
            return baseLR * (epoch + 1.0) / warmupEpochs;
        }
        double progress = (double)(epoch - warmupEpochs)
                        / (totalEpochs - warmupEpochs);
        progress = Math.min(1.0, progress);
        return minLR + 0.5 * (baseLR - minLR)
                     * (1 + Math.cos(Math.PI * progress));
    }

    @Override public String name() { return "WarmupCosine"; }
}