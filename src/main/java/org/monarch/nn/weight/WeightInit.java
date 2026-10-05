package org.monarch.nn.weight;

/** Weight initialization strategy. */
public enum WeightInit {
    XAVIER,     // Glorot: std = sqrt(2 / (fanIn + fanOut))
    HE,         // Kaiming: std = sqrt(2 / fanIn)
    LECUN,      // std = sqrt(1 / fanIn)
    NORMAL,     // std = 0.01
    UNIFORM,    // [-0.05, 0.05]
    ZERO;

    public static WeightInit fromString(String s) {
        if (s == null) return XAVIER;
        try {
            return valueOf(s.trim().toUpperCase());
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(
                    "Unknown weightInit: '" + s + "'. Valid: "
                    + java.util.Arrays.toString(values()));
        }
    }
}