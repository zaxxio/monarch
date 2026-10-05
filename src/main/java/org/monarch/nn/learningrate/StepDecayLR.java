package org.monarch.nn.learningrate;

public class StepDecayLR implements LRSchedule {

    private final double baseLR, gamma;
    private final int stepSize;

    public StepDecayLR(double baseLR, int stepSize, double gamma) {
        this.baseLR  = baseLR;
        this.stepSize = stepSize;
        this.gamma   = gamma;
    }

    @Override
    public double getLR(int epoch, int step) {
        return baseLR * Math.pow(gamma, (double) epoch / stepSize);
    }

    @Override public String name() { return "StepDecay"; }
}