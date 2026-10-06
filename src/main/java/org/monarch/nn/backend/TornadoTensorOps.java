package org.monarch.nn.backend;

import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * Dense-layer ops on a TornadoVM device (CUDA/PTX, Metal, ...).
 * Replaces MetalTensorOps: nothing in here is vendor specific, the device
 * is whatever TornadoVM runs the plan on (default device, or the one pinned
 * with -Dmonarch.tornado.device=b:d).
 *
 * This is the per-layer fallback path. The fast path is {@link GpuTrainer},
 * which keeps the whole network on the device and builds its plan through
 * {@link #newPlan(ImmutableTaskGraph)} so it lands on the same device.
 */
public final class TornadoTensorOps implements TensorOps {

    private final Backend requested;
    private final Object pinnedDevice;      // null => Tornado's default device
    private final String description;       // may be null

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

    public TornadoTensorOps(Backend requested, Object pinnedDevice, String description) {
        this.requested = requested;
        this.pinnedDevice = pinnedDevice;
        this.description = description;

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

    /** Creates an execution plan on this backend's device. */
    TornadoExecutionPlan newPlan(ImmutableTaskGraph graph) {
        return TornadoDevices.pin(new TornadoExecutionPlan(graph), pinnedDevice);
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
            return newPlan(g.snapshot());
        });
    }

    // ==================================================================
    // Backward plan
    //
    // dX_f, dW_f, db_f are included in transferToDevice so the host-side
    // zeroing in denseBackward reaches the device before the kernels run.
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
            return newPlan(g.snapshot());
        });
    }

    // ==================================================================
    // Batch lifecycle
    // ==================================================================

    @Override
    public void beginBatch() { }

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

        toFloat(dZ, dZ_f, B * out);
        toFloat(X,  X_f,  B * in);
        toFloat(W,  W_f,  out * in);

        Arrays.fill(dX_f, 0, B * in,   0.0f);
        Arrays.fill(dW_f, 0, out * in, 0.0f);
        Arrays.fill(db_f, 0, out,      0.0f);

        bwdPlan(B, in, out).execute();

        toDouble(dX_f, dX, B * in);
        toDouble(dW_f, dW, out * in);
        toDouble(db_f, db, out);
    }

    // ==================================================================
    @Override
    public String name() {
        String label;
        switch (requested) {
            case CUDA:  label = "CUDA (TornadoVM)";  break;
            case METAL: label = "Metal (TornadoVM)"; break;
            default:    label = "GPU (TornadoVM)";   break;
        }
        return description == null ? label : label + " " + description;
    }

    @Override
    public void sync() { }

    private static void toFloat(double[] src, float[] dst, int n) {
        for (int i = 0; i < n; i++) dst[i] = (float) src[i];
    }

    private static void toDouble(float[] src, double[] dst, int n) {
        for (int i = 0; i < n; i++) dst[i] = src[i];
    }
}