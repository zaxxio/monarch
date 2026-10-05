package org.monarch.nn.layer;

public enum LossType {
    MSE,                    // regression — identity/tanh output
    CROSS_ENTROPY,          // multi-class — softmax output
    BINARY_CROSS_ENTROPY    // binary — sigmoid output
}