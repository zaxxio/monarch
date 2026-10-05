package org.monarch.nn.backend;

public interface TensorOps {

    /**
     * Dense forward: Z = X · Wᵀ + b
     *   X : [B, in]   (row-major)
     *   W : [out, in] (row-major)
     *   b : [out]
     *   Z : [B, out]
     */
    void denseForward(double[] X, double[] W, double[] b, double[] Z,
                      int B, int in, int out);

    /**
     * Dense backward for one batch.
     *   dX : [B, in]   (output)
     *   dW : [out, in] (output)
     *   db : [out]     (output)
     *   dZ : [B, out]  (input)
     *   X  : [B, in]   (cached input)
     *   W  : [out, in] (current weights)
     */
    void denseBackward(double[] dX, double[] dW, double[] db,
                       double[] dZ, double[] X, double[] W,
                       int B, int in, int out);

    /**
     * Called by the training loop before forward() to let the backend
     * invalidate any device-side caches that may be stale after the
     * previous optimizer step.
     *
     * Default: no-op (CPU backend doesn't cache anything).
     */
    default void beginBatch() { }

    /**
     * Called by the training loop after optimizer update to mark any
     * device-side parameter caches as stale. On the next beginBatch()
     * they'll be re-uploaded.
     *
     * Default: no-op.
     */
    default void endBatch() { }

    String name();
    void sync();
}