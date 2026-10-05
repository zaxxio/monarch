package org.monarch.nn.optimizer;

public class SGDOptimizer implements Optimizer {
    @Override public void update(double[] p, double[] g, double lr, int step) {
        for (int i = 0; i < p.length; i++) p[i] -= lr * g[i];
    }
    @Override public void reset() { }
    @Override public String name() { return "SGD"; }
}