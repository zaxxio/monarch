package org.monarch.nn.optimizer;

public class MomentumOptimizer implements Optimizer {

    private final double momentum;
    private double[] velocity;

    public MomentumOptimizer() { this(0.9); }
    public MomentumOptimizer(double momentum) { this.momentum = momentum; }

    @Override
    public void update(double[] params, double[] grads, double lr, int step) {
        if (velocity == null || velocity.length != params.length)
            velocity = new double[params.length];

        for (int i = 0; i < params.length; i++) {
            velocity[i] = momentum * velocity[i] - lr * grads[i];
            params[i] += velocity[i];
        }
    }

    @Override public void reset() { velocity = null; }
    @Override public String name() { return "Momentum(" + momentum + ")"; }
}