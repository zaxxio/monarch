package org.monarch.nn.optimizer;

public class NadamOptimizer implements Optimizer {

    private final double beta1, beta2, eps;
    private double[] m, v;

    public NadamOptimizer() { this(0.9, 0.999, 1e-8); }
    public NadamOptimizer(double beta1, double beta2, double eps) {
        this.beta1 = beta1; this.beta2 = beta2; this.eps = eps;
    }

    @Override
    public void update(double[] params, double[] grads, double lr, int step) {
        if (m == null || m.length != params.length) {
            m = new double[params.length];
            v = new double[params.length];
        }
        int t = step + 1;
        double bc1 = 1.0 - Math.pow(beta1, t);
        double bc2 = 1.0 - Math.pow(beta2, t);
        double mu  = beta1 * (1.0 - 0.5 * Math.pow(0.5, t - 1));
        double muNext = beta1 * (1.0 - 0.5 * Math.pow(0.5, t));

        for (int i = 0; i < params.length; i++) {
            double g = grads[i];
            m[i] = beta1 * m[i] + (1 - beta1) * g;
            v[i] = beta2 * v[i] + (1 - beta2) * g * g;

            double mHat = m[i] / bc1;
            double vHat = v[i] / bc2;
            double mNest = mu * mHat + (1 - mu) * g / bc1;
            double mNextNest = muNext * mHat + (1 - muNext) * g / bc1;

            params[i] -= lr * mNextNest / (Math.sqrt(vHat) + eps);
        }
    }

    @Override public void reset() { m = null; v = null; }
    @Override public String name() { return "Nadam"; }
}