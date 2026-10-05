package org.monarch.nn.learningrate;

/** Type-safe selection of learning-rate schedule. */
public enum LRScheduleType {
    CONSTANT,
    STEP,
    EXPONENTIAL,
    COSINE;

    public static LRScheduleType fromString(String s) {
        if (s == null) return CONSTANT;
        try {
            return valueOf(s.trim().toUpperCase());
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(
                    "Unknown lrSchedule: '" + s + "'. Valid: " + java.util.Arrays.toString(values()));
        }
    }
}