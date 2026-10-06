package org.monarch.nn.backend;

import uk.ac.manchester.tornado.api.TornadoExecutionPlan;

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * Device selection helpers for the TornadoVM-backed GPU path (CUDA / Metal).
 *
 * TornadoVM executes whatever backends the installed SDK was built with
 * (PTX = CUDA, OpenCL, SPIR-V, Metal). Without help, a plan runs on Tornado's
 * default device (0:0), which is the right device when only one backend is
 * installed but may be the wrong one on a machine with several.
 *
 * This class provides:
 *  - an optional explicit pin:  -Dmonarch.tornado.device=<backend>:<device>
 *    (indices as printed by `tornado --devices`, e.g. "0:1")
 *  - a best-effort description of the device for logging and for checking
 *    that a CUDA request didn't land on, say, an Apple GPU.
 *
 * Device objects are handled reflectively on purpose: the TornadoDevice type
 * moved packages between TornadoVM releases and this keeps the project
 * compiling across them. If reflection fails, everything degrades to
 * "use Tornado's default device" and the device description is unknown.
 */
final class TornadoDevices {

    static final String DEVICE_PROPERTY = "monarch.tornado.device";

    private TornadoDevices() {}

    // ------------------------------------------------------------------
    // Selection
    // ------------------------------------------------------------------

    /**
     * The device named by -Dmonarch.tornado.device=b:d, or null if the
     * property is not set.
     *
     * @throws IllegalStateException if the property is malformed or names a
     *                               device that doesn't exist.
     */
    static Object configuredDevice() {
        String spec = System.getProperty(DEVICE_PROPERTY);
        if (spec == null || spec.trim().isEmpty()) return null;

        String[] p = spec.trim().split(":");
        if (p.length != 2)
            throw new IllegalStateException(
                    "-D" + DEVICE_PROPERTY + " must look like 0:1 (backend:device), got '" + spec + "'");
        try {
            int b = Integer.parseInt(p[0].trim());
            int d = Integer.parseInt(p[1].trim());
            Object dev = getDevice(b, d);
            if (dev == null)
                throw new IllegalStateException("No TornadoVM device " + spec
                        + " (run `tornado --devices` to list them)");
            return dev;
        } catch (NumberFormatException e) {
            throw new IllegalStateException(
                    "-D" + DEVICE_PROPERTY + " must look like 0:1, got '" + spec + "'", e);
        }
    }

    /** Tornado's default device (0:0), or null if it can't be queried. */
    static Object defaultDevice() {
        return getDevice(0, 0);
    }

    private static Object getDevice(int backend, int device) {
        try {
            Method m = TornadoExecutionPlan.class.getMethod("getDevice", int.class, int.class);
            return m.invoke(null, backend, device);
        } catch (Throwable t) {
            return null;
        }
    }

    /** Applies the device to the plan (reflective withDevice). Returns the plan to use. */
    static TornadoExecutionPlan pin(TornadoExecutionPlan plan, Object device) {
        if (device == null) return plan;
        for (Method m : TornadoExecutionPlan.class.getMethods()) {
            if (!m.getName().equals("withDevice") || m.getParameterCount() != 1) continue;
            if (!m.getParameterTypes()[0].isInstance(device)) continue;
            try {
                Object r = m.invoke(plan, device);
                return r instanceof TornadoExecutionPlan ? (TornadoExecutionPlan) r : plan;
            } catch (ReflectiveOperationException e) {
                throw new IllegalStateException("TornadoExecutionPlan.withDevice failed: " + e, e);
            }
        }
        throw new IllegalStateException(
                "This TornadoVM version has no TornadoExecutionPlan.withDevice(device); "
                + "remove -D" + DEVICE_PROPERTY + " and rely on the default device.");
    }

    // ------------------------------------------------------------------
    // Description / verification
    // ------------------------------------------------------------------

    /** "[platform] device name", or null if it can't be read. */
    static String describe(Object device) {
        if (device == null) return null;
        String platform = callString(device, "getPlatformName");
        String name = callString(device, "getDeviceName");
        if (platform == null && name == null) return null;
        StringBuilder sb = new StringBuilder();
        if (platform != null) sb.append('[').append(platform).append("] ");
        if (name != null) sb.append(name);
        return sb.toString().trim();
    }

    /**
     * Does the device description fit the requested backend?
     *
     * @return TRUE / FALSE when we can tell, null when the description is
     *         unknown (don't block in that case).
     *
     * Note: NVIDIA's OpenCL platform is also called "NVIDIA CUDA", so this
     * can't tell PTX from OpenCL on an NVIDIA GPU. Both are fine for us
     * (same kernels); `tornado --devices` shows the actual driver.
     */
    static Boolean matches(Backend requested, String description) {
        if (description == null) return null;
        String s = description.toLowerCase();
        switch (requested) {
            case CUDA:
                return s.contains("nvidia") || s.contains("cuda") || s.contains("ptx");
            case METAL:
                return s.contains("metal") || s.contains("apple");
            default:
                return Boolean.TRUE;
        }
    }

    // ------------------------------------------------------------------
    // Reflection helpers
    // ------------------------------------------------------------------

    /** Calls a public no-arg method via any exported type in the hierarchy. */
    private static String callString(Object target, String method) {
        for (Class<?> c : typesOf(target.getClass())) {
            try {
                Method m = c.getMethod(method);
                Object r = m.invoke(target);
                return r == null ? null : r.toString();
            } catch (Throwable ignored) {
                // not on this type, or not accessible - try the next one
            }
        }
        return null;
    }

    /** The class, its superclasses and all interfaces (public types first by hierarchy order). */
    private static List<Class<?>> typesOf(Class<?> start) {
        Set<Class<?>> seen = new LinkedHashSet<>();
        List<Class<?>> work = new ArrayList<>();
        work.add(start);
        while (!work.isEmpty()) {
            Class<?> c = work.remove(0);
            if (c == null || !seen.add(c)) continue;
            work.add(c.getSuperclass());
            for (Class<?> i : c.getInterfaces()) work.add(i);
        }
        return new ArrayList<>(seen);
    }
}