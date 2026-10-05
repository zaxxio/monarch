package org.monarch.nn.optimizer;

public class AdaDeltaOptimizer implements Optimizer {

    private final double rho, eps;
    private double[] eg2, edx2;

    public AdaDeltaOptimizer() { this(0.95, 1e-6); }
    public AdaDeltaOptimizer(double rho, double eps) {
        this.rho = rho;
        this.eps = eps;
    }

    @Override
    public void update(double[] params, double[] grads, double lr, int step) {
        if (eg2 == null || eg2.length != params.length) {
            eg2  = new double[params.length];
            edx2 = new double[params.length];
        }
        // AdaDelta ignores the caller-supplied lr; it self-scales.

        for (int i = 0; i < params.length; i++) {
            eg2[i] = rho * eg2[i] + (1 - rho) * grads[i] * grads[i];
            double dx = -Math.sqrt((edx2[i] + eps) / (eg2[i] + eps)) * grads[i];
            params[i] += dx;
            edx2[i] = rho * edx2[i] + (1 - rho) * dx * dx;
        }
    }

    @Override public void reset() { eg2 = null; edx2 = null; }
    @Override public String name() { return "AdaDelta"; }
}