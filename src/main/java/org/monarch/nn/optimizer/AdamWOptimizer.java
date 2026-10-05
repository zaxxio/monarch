package org.monarch.nn.optimizer;

public class AdamWOptimizer implements Optimizer {

    private final double beta1, beta2, eps, weightDecay;
    private double[] m, v;

    public AdamWOptimizer() { this(0.9, 0.999, 1e-8, 0.01); }
    public AdamWOptimizer(double beta1, double beta2, double eps, double weightDecay) {
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.eps   = eps;
        this.weightDecay = weightDecay;
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

            // Decoupled weight decay: shrink params before the Adam step
            params[i] -= lr * weightDecay * params[i];
            params[i] -= lr * mHat / (Math.sqrt(vHat) + eps);
        }
    }

    @Override public void reset() { m = null; v = null; }
    @Override public String name() { return "AdamW(wd=" + weightDecay + ")"; }
}