package org.monarch.nn.learningrate;

public class CosineAnnealingLR implements LRSchedule {

    private final double baseLR, minLR;
    private final int maxEpochs;

    public CosineAnnealingLR(double baseLR, double minLR, int maxEpochs) {
        this.baseLR = baseLR;
        this.minLR  = minLR;
        this.maxEpochs = Math.max(1, maxEpochs);
    }

    @Override
    public double getLR(int epoch, int step) {
        int e = Math.min(epoch, maxEpochs);
        return minLR + 0.5 * (baseLR - minLR)
                     * (1 + Math.cos(Math.PI * e / maxEpochs));
    }

    @Override public String name() { return "CosineAnnealing"; }
}