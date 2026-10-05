package org.monarch.nn.optimizer;

public class RMSPropOptimizer implements Optimizer {

    private final double decay, eps;
    private double[] cache;

    public RMSPropOptimizer() { this(0.9, 1e-8); }
    public RMSPropOptimizer(double decay, double eps) {
        this.decay = decay;
        this.eps   = eps;
    }

    @Override
    public void update(double[] params, double[] grads, double lr, int step) {
        if (cache == null || cache.length != params.length)
            cache = new double[params.length];

        for (int i = 0; i < params.length; i++) {
            cache[i] = decay * cache[i] + (1 - decay) * grads[i] * grads[i];
            params[i] -= lr * grads[i] / (Math.sqrt(cache[i]) + eps);
        }
    }

    @Override public void reset() { cache = null; }
    @Override public String name() { return "RMSProp"; }
}