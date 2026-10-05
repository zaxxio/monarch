package org.monarch.nn.optimizer;

public interface Optimizer {
    void update(double[] params, double[] grads, double lr, int step);
    void reset();
    String name();
}