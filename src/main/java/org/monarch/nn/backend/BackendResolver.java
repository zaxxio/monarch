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

    private BackendResolver() {}

    /**
     * Auto-resolving variant. Picks CPU if the model is too small.
     */
    public static TensorOps resolve(Backend requested, List<Layer> layers) {
        if (requested == null) requested = Backend.CPU;

        // Auto picks based on model size.
        if (requested == Backend.AUTO) {
            int params = countParams(layers);
            if (params < GPU_PARAM_THRESHOLD) {
                log.info("Model has {} params (< {} threshold) — using CPU",
                        params, GPU_PARAM_THRESHOLD);
                return CpuTensorOps.INSTANCE;
            }
            log.info("Model has {} params — considering GPU", params);
            return tryMetalOrFallback(Backend.AUTO);
        }

        // Explicit backend choice — honor it.
        switch (requested) {
            case CPU:
                log.info("Selected backend: CPU");
                return CpuTensorOps.INSTANCE;

            case METAL:
                return tryMetalOrFallback(Backend.METAL);

            case CUDA:
            case OPENCL:
            default:
                log.warn("{} is not implemented — falling back to CPU", requested);
                return CpuTensorOps.INSTANCE;
        }
    }

    /** Backward-compatible variant for callers that don't pass layers. */
    public static TensorOps resolve(Backend requested) {
        if (requested == null) requested = Backend.CPU;
        switch (requested) {
            case CPU:
                log.info("Selected backend: CPU");
                return CpuTensorOps.INSTANCE;
            case METAL:
            case AUTO:
                return tryMetalOrFallback(requested);
            case CUDA:
            case OPENCL:
            default:
                log.warn("{} is not implemented — falling back to CPU", requested);
                return CpuTensorOps.INSTANCE;
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

    private static TensorOps tryMetalOrFallback(Backend requested) {
        try {
            TornadoBootstrap.init();
        } catch (Throwable t) {
            log.warn("TornadoBootstrap init failed: {}", t.toString());
            if (requested == Backend.METAL) {
                log.warn("METAL explicitly requested but bootstrap failed — using CPU");
            }
            return CpuTensorOps.INSTANCE;
        }

        if (!isClassAvailable("uk.ac.manchester.tornado.api.TornadoExecutionPlan")) {
            log.info("TornadoVM API not on classpath — using CPU");
            if (requested == Backend.METAL) {
                log.warn("METAL explicitly requested but the TornadoVM API is missing");
            }
            return CpuTensorOps.INSTANCE;
        }

        try {
            Class.forName("uk.ac.manchester.tornado.api.runtime.TornadoRuntimeProvider")
                    .getMethod("getTornadoRuntime")
                    .invoke(null);
        } catch (Throwable t) {
            Throwable cause = unwrap(t);
            log.warn("TornadoVM runtime init failed: {}", summarize(cause));
            if (requested == Backend.METAL) {
                log.warn("METAL explicitly requested but the runtime failed — using CPU");
            }
            return CpuTensorOps.INSTANCE;
        }

        try {
            MetalTensorOps ops = new MetalTensorOps();
            log.info("Selected backend: {}", ops.name());
            return ops;
        } catch (Throwable t) {
            Throwable cause = unwrap(t);
            log.warn("Metal backend init failed: {}", summarize(cause));
            return CpuTensorOps.INSTANCE;
        }
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