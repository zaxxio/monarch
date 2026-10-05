package org.monarch.nn.layer;

import org.monarch.nn.optimizer.Optimizer;
import org.monarch.nn.tensor.Tensor;

/**
 * Flattens all non-batch dimensions into a single feature axis.
 *
 *   Input : [B, C, H, W]   (or any [B, d1, d2, ..., dn])
 *   Output: [B, C*H*W]     (or [B, d1*d2*...*dn])
 *
 * Used to bridge CNN/Pooling output to Dense layers.
 * No parameters, no activation — pure reshape.
 */
public class FlattenLayer extends Layer {

    private int[] lastInputShape;

    protected FlattenLayer(Builder b) { /* nothing to configure */ }

    @Override
    public LayerType getType() { return LayerType.FLATTEN; }

    @Override
    public void init(int[] inputShape) {
        if (inputShape.length < 2)
            throw new IllegalArgumentException(
                    "Flatten expects at least 2 dimensions, got "
                    + java.util.Arrays.toString(inputShape));
    }

    @Override
    public int[] outputShape(int[] inputShape) {
        int size = 1;
        for (int s : inputShape) size *= s;
        return new int[]{ size };
    }

    @Override
    public Tensor forward(Tensor input, boolean training) {
        lastInputShape = input.getShape();
        int B = input.getShape()[0];
        int flat = input.size() / B;

        // Data is unchanged — only the shape metadata changes.
        return new Tensor(input.getData().clone(), B, flat);
    }

    @Override
    public Tensor backward(Tensor gradOutput) {
        // Restore the original multi-dimensional shape.
        return new Tensor(gradOutput.getData().clone(), lastInputShape);
    }

    @Override
    public void update(Optimizer opt, double lr, int step, int batchSize) { }

    public static class Builder extends Layer.Builder<FlattenLayer, Builder> {
        @Override protected Builder self() { return this; }
        @Override public FlattenLayer build() { return new FlattenLayer(this); }
    }
}