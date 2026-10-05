package org.monarch.nn.activation;

public enum ActivationFunction {
    SIGMOID,
    TANH,
    RELU,
    LEAKY_RELU,
    SWISH,
    IDENTITY,
    SOFTMAX;

    public static ActivationFunction fromString(String s) {
        if (s == null) return IDENTITY;
        try {
            return valueOf(s.trim().toUpperCase());
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(
                    "Unknown activation: '" + s + "'. Valid: " + java.util.Arrays.toString(values()));
        }
    }
}