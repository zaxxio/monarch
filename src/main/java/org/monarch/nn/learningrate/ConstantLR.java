package org.monarch.nn.learningrate;

public class ConstantLR implements LRSchedule {
    private final double lr;
    public ConstantLR(double lr) { this.lr = lr; }
    public double getLR(int e, int s) { return lr; }
    public String name() { return "Constant(" + lr + ")"; }
}