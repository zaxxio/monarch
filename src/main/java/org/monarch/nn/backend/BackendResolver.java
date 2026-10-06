package org.monarch.nn.backend;

import org.monarch.nn.layer.Layer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

public final class BackendResolver {

    private static final Logger log = LoggerFactory.getLogger(BackendResolver.class);

    /**
     * Below this parameter count, GPU transfer overhead dominates
     * the compute savings. Auto mode falls back to CPU.
     */
    private static final int GPU_PARAM_THRESHOLD = 100_000;

    /**
     * -Dmonarch.backend.strict=true makes an explicit METAL/CUDA request throw
     * instead of silently falling back to the CPU. Recommended while
     * benchmarking: a silent CPU fallback looks exactly like a slow GPU.
     */
    private static final String STRICT_PROPERTY = "monarch.backend.strict";

    private BackendResolver() {}

    /**
     * Auto-resolving variant. Picks CPU if the model is too small.
     */
    public static TensorOps resolve(Backend requested, List<Layer> layers) {
        if (requested == null) requested = Backend.CPU;

        if (requested == Backend.AUTO) {
            int params = countParams(layers);
            if (params < GPU_PARAM_THRESHOLD) {
                log.info("Model has {} params (< {} threshold) - using CPU",
                        params, GPU_PARAM_THRESHOLD);
                return CpuTensorOps.INSTANCE;
            }
            log.info("Model has {} params - considering GPU", params);
            return tryTornadoOrFallback(Backend.AUTO);
        }

        return resolve(requested);
    }

    /** Backward-compatible variant for callers that don't pass layers. */
    public static TensorOps resolve(Backend requested) {
        if (requested == null) requested = Backend.CPU;
        switch (requested) {
            case CPU:
                log.info("Selected backend: CPU");
                return CpuTensorOps.INSTANCE;

            case METAL:
            case CUDA:
            case AUTO:
                return tryTornadoOrFallback(requested);

            case OPENCL:
            default:
                return fallback(requested, requested + " is not implemented");
        }
    }

    private static int countParams(List<Layer> layers) {
        if (layers == null) return 0;
        int n = 0;
        for (Layer l : layers) {
            n += l.parameterCount();
        }
        return n;
    }

    // ------------------------------------------------------------------
    // TornadoVM (CUDA / Metal / AUTO)
    // ------------------------------------------------------------------
    private static TensorOps tryTornadoOrFallback(Backend requested) {
        try {
            TornadoBootstrap.init();
        } catch (Throwable t) {
            return fallback(requested, "TornadoBootstrap init failed: " + t);
        }

        if (!isClassAvailable("uk.ac.manchester.tornado.api.TornadoExecutionPlan")) {
            return fallback(requested, "TornadoVM API not on classpath");
        }

        try {
            Class.forName("uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider")
                    .getMethod("getTornadoRuntime")
                    .invoke(null);
        } catch (Throwable t) {
            return fallback(requested, "TornadoVM runtime init failed: " + summarize(unwrap(t))
                    + " (is the app running through the `tornado` launcher with a backend installed?)");
        }

        // ---- Which device will actually run the kernels? ----
        Object device;
        try {
            device = TornadoDevices.configuredDevice();   // -Dmonarch.tornado.device=b:d
        } catch (RuntimeException e) {
            return fallback(requested, e.getMessage());
        }

        Object probe = device != null ? device : TornadoDevices.defaultDevice();
        String description = TornadoDevices.describe(probe);
        Boolean fits = TornadoDevices.matches(requested, description);

        if (Boolean.FALSE.equals(fits)) {
            return fallback(requested, requested + " requested, but the "
                    + (device != null ? "pinned" : "default") + " TornadoVM device is '"
                    + description + "'. Install a TornadoVM build with the matching backend "
                    + "(CUDA = PTX), or pick the device with -D"
                    + TornadoDevices.DEVICE_PROPERTY + "=<backend>:<device> "
                    + "(indices from `tornado --devices`).");
        }
        if (description == null && requested != Backend.AUTO) {
            log.warn("Could not read the TornadoVM device name - assuming the default device "
                    + "is the requested {} one. Check `tornado --devices`.", requested);
        }

        try {
            TornadoTensorOps ops = new TornadoTensorOps(requested, device, description);
            log.info("Selected backend: {}", ops.name());
            return ops;
        } catch (Throwable t) {
            return fallback(requested, "TornadoVM backend init failed: " + summarize(unwrap(t)));
        }
    }

    /** CPU fallback, or an exception when -Dmonarch.backend.strict=true. */
    private static TensorOps fallback(Backend requested, String reason) {
        if (Boolean.getBoolean(STRICT_PROPERTY) && requested != Backend.AUTO) {
            throw new IllegalStateException(requested + " backend unavailable: " + reason);
        }
        if (requested == Backend.METAL || requested == Backend.CUDA) {
            log.warn("{} explicitly requested but unavailable ({}) - using CPU",
                    requested, reason);
        } else {
            log.warn("{} - using CPU", reason);
        }
        return CpuTensorOps.INSTANCE;
    }

    private static boolean isClassAvailable(String name) {
        try { Class.forName(name); return true; }
        catch (Throwable t) { return false; }
    }

    private static Throwable unwrap(Throwable t) {
        if (t.getCause() != null && t.getCause() != t) return t.getCause();
        return t;
    }

    private static String summarize(Throwable t) {
        String msg = t.getMessage();
        return msg == null
                ? t.getClass().getSimpleName()
                : t.getClass().getSimpleName() + ": " + msg;
    }
}