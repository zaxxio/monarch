package org.monarch.nn.optimizer;

/** Type-safe selection of optimizer algorithm. */
public enum OptimizerType {
    SGD,
    MOMENTUM,
    ADAM,
    ADAMW,
    NADAM,
    RMSPROP,
    ADAGRAD,
    ADADELTA;

    /** Case-insensitive parse; useful for YAML/CLI strings. */
    public static OptimizerType fromString(String s) {
        if (s == null) return SGD;
        try {
            return valueOf(s.trim().toUpperCase());
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(
                    "Unknown optimizer: '" + s + "'. Valid: " + java.util.Arrays.toString(values()));
        }
    }
}