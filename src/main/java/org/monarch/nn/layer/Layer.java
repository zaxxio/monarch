package org.monarch.nn.layer;

import org.monarch.nn.activation.ActivationFunction;
import org.monarch.nn.backend.CpuTensorOps;
import org.monarch.nn.backend.TensorOps;
import org.monarch.nn.optimizer.Optimizer;
import org.monarch.nn.tensor.Tensor;

import java.io.Serializable;
import java.util.UUID;

public abstract class Layer implements Serializable {

    protected UUID layerId;
    protected int nIn;
    protected int nOut;
    protected ActivationFunction activationFunction = ActivationFunction.IDENTITY;

    /** Hardware ops — CPU by default, swapped to Metal by MultiLayerNetwork. */
    protected transient TensorOps ops = CpuTensorOps.INSTANCE;

    /**
     * Set by MultiLayerNetwork when the incoming gradient already
     * incorporates the activation derivative (last-layer combined-loss form).
     * When true, the layer skips its own activation derivative.
     */
    protected transient boolean skipActivationDerivative = false;

    protected Layer() { this.layerId = UUID.randomUUID(); }

    public int getNIn() { return nIn; }
    public int getNOut() { return nOut; }
    public UUID getLayerId() { return layerId; }
    public ActivationFunction getActivationFunction() { return activationFunction; }

    public abstract void init(int[] inputShape);
    public abstract int[] outputShape(int[] inputShape);
    public abstract Tensor forward(Tensor input, boolean training);
    public abstract Tensor backward(Tensor gradOutput);
    public abstract void update(Optimizer opt, double lr, int step, int batchSize);
    public abstract LayerType getType();

    public void setOps(TensorOps ops) {
        this.ops = ops == null ? CpuTensorOps.INSTANCE : ops;
    }

    public TensorOps getOps() { return ops; }

    public boolean isSkipActivationDerivative() {
        return skipActivationDerivative;
    }

    public void setSkipActivationDerivative(boolean skip) {
        this.skipActivationDerivative = skip;
    }

    // Parameter hooks — default to "no parameters"
    public double[] getParams() { return null; }
    public int parameterCount() { return 0; }
    public double[] getGrads()  { return null; }
    public void setParams(double[] params) { /* no-op */ }

    // ------------------------------------------------------------------
    // Builder
    // ------------------------------------------------------------------
    public abstract static class Builder<T extends Layer, B extends Builder<T, B>> {
        protected int nIn = -1;
        protected int nOut = -1;
        protected ActivationFunction activationFunction = ActivationFunction.IDENTITY;

        public B nIn(int v) { this.nIn = v; return self(); }
        public B nOut(int v) { this.nOut = v; return self(); }
        public B activation(ActivationFunction a) { this.activationFunction = a; return self(); }

        protected abstract B self();
        public abstract T build();
    }
}