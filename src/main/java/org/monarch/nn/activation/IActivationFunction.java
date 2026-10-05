package org.monarch.nn.activation;

public interface IActivationFunction {
    double output(double z);          // z = pre-activation
    double outputDerivative(double z);
}