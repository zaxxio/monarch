package org.monarch.nn.backend;

import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public final class MetalTensorOps implements TensorOps {

    // ---- Device-side scratch buffers (shared across all dense layers) ----
    private float[] X_f, W_f, b_f, Z_f;
    private float[] dX_f, dW_f, db_f, dZ_f;

    // ---- Cached execution plans per (B, in, out) shape ----
    private final Map<Long, TornadoExecutionPlan> fwdPlans = new HashMap<>();
    private final Map<Long, TornadoExecutionPlan> bwdPlans = new HashMap<>();

    // ---- Parameter cache for the forward pass ----
    private double[] cachedW = null;
    private double[] cachedB = null;
    private int cachedWLen = 0;
    private int cachedBLen = 0;
    private boolean weightsDirty = true;

    public MetalTensorOps() {
        TornadoBootstrap.init();

        try {
            Class<?> rt = Class.forName(
                    "uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider");
            rt.getMethod("getTornadoRuntime").invoke(null);
        } catch (Throwable t) {
            Throwable cause = t.getCause() != null ? t.getCause() : t;
            throw new IllegalStateException(
                    "TornadoVM runtime not available: " + cause.getMessage(), cause);
        }
    }

    private void ensureCapacity(int B, int in, int out) {
        int xSize = B * in;
        int wSize = out * in;
        int zSize = B * out;

        if (X_f == null || X_f.length < xSize) { X_f = new float[xSize]; dX_f = new float[xSize]; }
        if (W_f == null || W_f.length < wSize) { W_f = new float[wSize]; dW_f = new float[wSize]; }
        if (b_f == null || b_f.length < out)   { b_f = new float[out];   db_f = new float[out];   }
        if (Z_f == null || Z_f.length < zSize) { Z_f = new float[zSize]; dZ_f = new float[zSize]; }
    }

    private static long key(int B, int in, int out) {
        return ((long) B << 40) | ((long) in << 20) | out;
    }

    // ==================================================================
    // Forward plan
    // ==================================================================
    private TornadoExecutionPlan fwdPlan(int B, int in, int out) {
        return fwdPlans.computeIfAbsent(key(B, in, out), k -> {
            TaskGraph g = new TaskGraph("denseFwd" + B + "_" + in + "_" + out)
                    .transferToDevice(DataTransferMode.EVERY_EXECUTION, X_f, W_f, b_f)
                    .task("forward", TornadoKernels::denseForward,
                            X_f, W_f, b_f, Z_f, B, in, out)
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, Z_f);
            return new TornadoExecutionPlan(g.snapshot());
        });
    }

    // ==================================================================
    // Backward plan
    //
    // IMPORTANT: dX_f, dW_f, db_f are included in transferToDevice so that
    // whatever the host-side zeroing wrote (see denseBackward) actually
    // reaches the device before the kernels run. Without this, the kernels
    // see stale residue from previous layers / batches, and any "+=" in
    // the kernel overcounts.
    // ==================================================================
    private TornadoExecutionPlan bwdPlan(int B, int in, int out) {
        return bwdPlans.computeIfAbsent(key(B, in, out), k -> {
            TaskGraph g = new TaskGraph("denseBwd" + B + "_" + in + "_" + out)
                    .transferToDevice(DataTransferMode.EVERY_EXECUTION,
                            dZ_f, X_f, W_f,
                            dX_f, dW_f, db_f)
                    .task("dx", TornadoKernels::denseBackwardDX, dZ_f, W_f, dX_f, B, in, out)
                    .task("dw", TornadoKernels::denseBackwardDW, dZ_f, X_f, dW_f, B, in, out)
                    .task("db", TornadoKernels::denseBackwardDB, dZ_f, db_f, B, out)
                    .transferToHost(DataTransferMode.EVERY_EXECUTION, dX_f, dW_f, db_f);
            return new TornadoExecutionPlan(g.snapshot());
        });
    }

    // ==================================================================
    // Batch lifecycle
    // ==================================================================

    @Override
    public void beginBatch() {
        // Nothing to do — buffers are managed lazily.
    }

    @Override
    public void endBatch() {
        // The optimizer updated W and b in place on the host.
        // Force the next forward to re-upload.
        weightsDirty = true;
    }

    // ==================================================================
    // Forward
    // ==================================================================

    @Override
    public void denseForward(double[] X, double[] W, double[] b, double[] Z,
                             int B, int in, int out) {
        ensureCapacity(B, in, out);

        toFloat(X, X_f, B * in);

        // Upload W and b only when they actually changed since the last
        // forward call. The cache key is the host array identity plus length.
        boolean needUpload = weightsDirty
                || cachedW != W || cachedB != b
                || cachedWLen != out * in || cachedBLen != out;

        if (needUpload) {
            toFloat(W, W_f, out * in);
            toFloat(b, b_f, out);
            cachedW = W;
            cachedB = b;
            cachedWLen = out * in;
            cachedBLen = out;
            weightsDirty = false;
        }

        fwdPlan(B, in, out).execute();

        toDouble(Z_f, Z, B * out);
    }

    // ==================================================================
    // Backward
    // ==================================================================

    @Override
    public void denseBackward(double[] dX, double[] dW, double[] db,
                              double[] dZ, double[] X, double[] W,
                              int B, int in, int out) {
        ensureCapacity(B, in, out);

        // ---- Upload this call's inputs to the shared float buffers ----
        // W_f is shared across all layers, so we must re-upload W every
        // backward call. The forward-pass cache does not help here because
        // between layer N's forward and layer N's backward, other layers'
        // forwards have overwritten W_f.
        toFloat(dZ, dZ_f, B * out);
        toFloat(X,  X_f,  B * in);
        toFloat(W,  W_f,  out * in);

        // ---- Zero the shared gradient accumulators ----
        // dX_f, dW_f, db_f are shared across every dense layer and every
        // batch. The kernels accumulate with "+=", so without zeroing, the
        // gradient carries residue from previous layers and previous
        // batches — which is the source of the ~1.87x overcounting.
        Arrays.fill(dX_f, 0, B * in,   0.0f);
        Arrays.fill(dW_f, 0, out * in, 0.0f);
        Arrays.fill(db_f, 0, out,      0.0f);

        // ---- Run the kernels ----
        // transferToDevice inside bwdPlan uploads the zeroed accumulators
        // and the fresh inputs; the kernels then add into them.
        bwdPlan(B, in, out).execute();

        // ---- Copy results back to host ----
        toDouble(dX_f, dX, B * in);
        toDouble(dW_f, dW, out * in);
        toDouble(db_f, db, out);
    }

    // ==================================================================
    @Override
    public String name() { return "Metal (TornadoVM)"; }

    @Override
    public void sync() { }

    private static void toFloat(double[] src, float[] dst, int n) {
        for (int i = 0; i < n; i++) dst[i] = (float) src[i];
    }

    private static void toDouble(float[] src, double[] dst, int n) {
        for (int i = 0; i < n; i++) dst[i] = src[i];
    }
}