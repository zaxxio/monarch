package org.monarch.nn.layer;

import org.monarch.nn.optimizer.Optimizer;
import org.monarch.nn.tensor.Tensor;

import java.util.Arrays;

/**
 * Reshapes a tensor without changing its data.
 *
 * Per-sample shape changes from whatever comes in to whatever
 * the builder declared. The batch dimension is preserved.
 *
 *   input  : [B, 784]        (per-sample [784])
 *   output : [B, 1, 28, 28]  (per-sample [1, 28, 28])
 *
 * Data order is row-major in both forms, so the flat array is
 * copied verbatim and only the shape metadata changes.
 *
 * No parameters. No gradients. Only the shape is cached.
 */
public class ReshapeLayer extends Layer {

    private final int[] targetShape;    // per-sample shape (excludes batch)
    private int[] lastInputShape;       // cached at forward() for backward()

    protected ReshapeLayer(Builder b) {
        this.targetShape = b.targetShape.clone();
    }

    @Override
    public LayerType getType() {
        return LayerType.RESHAPE;
    }

    @Override
    public void init(int[] inputShape) {
        // Nothing to allocate. But we validate that the total
        // element count matches so a broken config fails early.
        int inSize = 1;
        for (int s : inputShape) inSize *= s;

        int outSize = 1;
        for (int s : targetShape) outSize *= s;

        if (inSize != outSize) {
            throw new IllegalStateException(
                    "ReshapeLayer size mismatch: input "
                    + Arrays.toString(inputShape)
                    + " (" + inSize + " elements) → target "
                    + Arrays.toString(targetShape)
                    + " (" + outSize + " elements)");
        }
    }

    @Override
    public int[] outputShape(int[] inputShape) {
        return targetShape.clone();
    }

    @Override
    public Tensor forward(Tensor input, boolean training) {
        lastInputShape = input.getShape();
        int B = input.getShape()[0];

        int[] outShape = new int[targetShape.length + 1];
        outShape[0] = B;
        System.arraycopy(targetShape, 0, outShape, 1, targetShape.length);

        // Clone the flat data — the shape changes, the values don't.
        return new Tensor(input.getData().clone(), outShape);
    }

    @Override
    public Tensor backward(Tensor gradOutput) {
        // Restore the original multi-dimensional shape.
        return new Tensor(gradOutput.getData().clone(), lastInputShape);
    }

    @Override
    public void update(Optimizer opt, double lr, int step, int batchSize) {
        // No parameters to update.
    }

    // No getParams / getGrads override — the defaults return null,
    // which MultiLayerNetwork treats as "no trainable parameters".

    // ==================================================================
    // Builder
    // ==================================================================
    public static class Builder extends Layer.Builder<ReshapeLayer, Builder> {

        private int[] targetShape = null;

        /** Declare the per-sample target shape. E.g. {@code .shape(1, 28, 28)} */
        public Builder shape(int... s) {
            if (s == null || s.length == 0) {
                throw new IllegalArgumentException(
                        "ReshapeLayer.shape(...) requires at least one dimension");
            }
            this.targetShape = s.clone();
            return this;
        }

        @Override
        protected Builder self() { return this; }

        @Override
        public ReshapeLayer build() {
            if (targetShape == null) {
                throw new IllegalStateException(
                        "ReshapeLayer requires .shape(d1, d2, ...)");
            }
            return new ReshapeLayer(this);
        }
    }
}