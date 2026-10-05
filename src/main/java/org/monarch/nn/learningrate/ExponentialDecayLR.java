package org.monarch.nn.learningrate;

public class ExponentialDecayLR implements LRSchedule {

    private final double baseLR, decay;

    public ExponentialDecayLR(double baseLR, double decay) {
        this.baseLR = baseLR;
        this.decay  = decay;
    }

    @Override
    public double getLR(int epoch, int step) {
        return baseLR * Math.exp(-decay * epoch);
    }

    @Override public String name() { return "ExponentialDecay"; }
}