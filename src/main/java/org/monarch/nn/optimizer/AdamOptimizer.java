package org.monarch.nn.optimizer;

public class AdamOptimizer implements Optimizer {

    private final double beta1, beta2, eps;
    private double[] m, v;

    public AdamOptimizer() { this(0.9, 0.999, 1e-8); }
    public AdamOptimizer(double beta1, double beta2, double eps) {
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.eps   = eps;
    }

    @Override
    public void update(double[] params, double[] grads, double lr, int step) {
        if (m == null || m.length != params.length) {
            m = new double[params.length];
            v = new double[params.length];
        }
        double bc1 = 1.0 - Math.pow(beta1, step + 1);
        double bc2 = 1.0 - Math.pow(beta2, step + 1);

        for (int i = 0; i < params.length; i++) {
            m[i] = beta1 * m[i] + (1 - beta1) * grads[i];
            v[i] = beta2 * v[i] + (1 - beta2) * grads[i] * grads[i];

            double mHat = m[i] / bc1;
            double vHat = v[i] / bc2;
            params[i] -= lr * mHat / (Math.sqrt(vHat) + eps);
        }
    }

    @Override public void reset() { m = null; v = null; }
    @Override public String name() { return "Adam"; }
}