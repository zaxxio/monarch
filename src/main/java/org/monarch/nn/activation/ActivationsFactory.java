package org.monarch.nn.activation;

public final class ActivationsFactory {

    private ActivationsFactory() {}

    public static IActivationFunction get(ActivationFunction a) {
        switch (a) {
            case SIGMOID:    return SIGMOID;
            case TANH:       return TANH;
            case RELU:       return RELU;
            case LEAKY_RELU: return LEAKY_RELU;
            case SWISH:      return SWISH;
            case IDENTITY:   return IDENTITY;
            default: throw new IllegalArgumentException("No element-wise impl for " + a);
        }
    }

    public static final IActivationFunction SIGMOID = new IActivationFunction() {
        public double output(double z) { return 1.0 / (1.0 + Math.exp(-z)); }
        public double outputDerivative(double z) { double s = output(z); return s * (1 - s); }
    };

    public static final IActivationFunction TANH = new IActivationFunction() {
        public double output(double z) { return Math.tanh(z); }
        public double outputDerivative(double z) { double t = Math.tanh(z); return 1 - t * t; }
    };

    public static final IActivationFunction RELU = new IActivationFunction() {
        public double output(double z) { return z > 0 ? z : 0; }
        public double outputDerivative(double z) { return z > 0 ? 1 : 0; }
    };

    public static final IActivationFunction LEAKY_RELU = new IActivationFunction() {
        public double output(double z) { return z > 0 ? z : 0.01 * z; }
        public double outputDerivative(double z) { return z > 0 ? 1 : 0.01; }
    };

    public static final IActivationFunction SWISH = new IActivationFunction() {
        public double output(double z) { double s = 1.0 / (1.0 + Math.exp(-z)); return z * s; }
        public double outputDerivative(double z) {
            double s = 1.0 / (1.0 + Math.exp(-z));
            return s + z * s * (1 - s);
        }
    };

    public static final IActivationFunction IDENTITY = new IActivationFunction() {
        public double output(double z) { return z; }
        public double outputDerivative(double z) { return 1; }
    };
}