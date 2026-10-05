package org.monarch.nn.layer;

import org.monarch.nn.optimizer.Optimizer;
import org.monarch.nn.tensor.Tensor;

public class InputLayer extends Layer {

    protected InputLayer(Builder b) {
        this.nIn = b.nIn;
        this.nOut = b.nIn;
    }

    @Override public void init(int[] inputShape) { }
    @Override public int[] outputShape(int[] inputShape) { return inputShape.clone(); }
    @Override public Tensor forward(Tensor input, boolean training) { return input; }
    @Override public Tensor backward(Tensor gradOutput) { return gradOutput; }
    @Override public void update(Optimizer opt, double lr, int step, int batchSize) { }
    @Override public LayerType getType() { return LayerType.INPUT; }

    public static class Builder extends Layer.Builder<InputLayer, Builder> {
        @Override protected Builder self() { return this; }
        @Override public InputLayer build() { return new InputLayer(this); }
    }
}