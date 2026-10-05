package org.monarch.nn.layer;

import org.monarch.nn.activation.ActivationFunction;

public class OutputLayer extends DenseLayer {

    private final LossType lossType;

    protected OutputLayer(Builder b) {
        super(b, b.initStd, b.seed);
        this.lossType = b.lossType;
    }

    @Override
    public LayerType getType() {
        return LayerType.OUTPUT;
    }

    public LossType getLossType() {
        return lossType;
    }

    public static class Builder extends Layer.Builder<OutputLayer, Builder> {

        // Package-private so OutputLayer's constructor can read them.
        double initStd = -1;
        long seed = 42;
        LossType lossType = LossType.CROSS_ENTROPY;

        public Builder() {
            this.activationFunction = ActivationFunction.SOFTMAX;
        }

        public Builder initStd(double s) {
            this.initStd = s;
            return this;
        }

        public Builder seed(long s) {
            this.seed = s;
            return this;
        }

        public Builder lossType(LossType t) {
            this.lossType = t;
            return this;
        }

        public Builder inferLoss() {
            switch (activationFunction) {
                case SOFTMAX:
                    this.lossType = LossType.CROSS_ENTROPY;
                    break;
                case SIGMOID:
                    this.lossType = LossType.BINARY_CROSS_ENTROPY;
                    break;
                default:
                    this.lossType = LossType.MSE;
                    break;
            }
            return this;
        }

        @Override
        protected Builder self() {
            return this;
        }

        @Override
        public OutputLayer build() {
            return new OutputLayer(this);
        }
    }
}