package org.monarch.nn.data;

public class MLData {
    private final double[] inputs;
    private final double[] targets;

    public MLData(double[] inputs, double[] targets) {
        this.inputs = inputs;
        this.targets = targets;
    }
    public double[] getInputs() { return inputs; }
    public double[] getTargets() { return targets; }
}