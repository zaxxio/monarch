package org.monarch.nn.optimizer;

public class AdaGradOptimizer implements Optimizer {

    private final double eps;
    private double[] cache;

    public AdaGradOptimizer() { this(1e-8); }
    public AdaGradOptimizer(double eps) { this.eps = eps; }

    @Override
    public void update(double[] params, double[] grads, double lr, int step) {
        if (cache == null || cache.length != params.length)
            cache = new double[params.length];

        for (int i = 0; i < params.length; i++) {
            cache[i] += grads[i] * grads[i];
            params[i] -= lr * grads[i] / (Math.sqrt(cache[i]) + eps);
        }
    }

    @Override public void reset() { cache = null; }
    @Override public String name() { return "AdaGrad"; }
}