package org.monarch.nn.weight;

/**
 * High-level batch strategy. Controls how many samples are grouped
 * into one gradient update.
 *
 *   GRADIENT_DESCENT             — full batch (one update per epoch)
 *   STOCHASTIC_GRADIENT_DESCENT  — one sample per update
 *   MINI_BATCH_GRADIENT_DESCENT  — batch size from config
 *   ADAPTIVE                     — mini-batch with automatic tuning
 */
public enum OptimizationAlgo {
    GRADIENT_DESCENT,
    STOCHASTIC_GRADIENT_DESCENT,
    MINI_BATCH_GRADIENT_DESCENT,
    ADAPTIVE;

    public static OptimizationAlgo fromString(String s) {
        if (s == null) return STOCHASTIC_GRADIENT_DESCENT;
        try {
            return valueOf(s.trim().toUpperCase());
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(
                    "Unknown optimizationAlgo: '" + s + "'. Valid: "
                    + java.util.Arrays.toString(values()));
        }
    }
}