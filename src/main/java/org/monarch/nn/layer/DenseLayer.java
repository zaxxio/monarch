package org.monarch.nn.layer;

import org.monarch.nn.activation.ActivationFunction;
import org.monarch.nn.activation.ActivationsFactory;
import org.monarch.nn.activation.IActivationFunction;
import org.monarch.nn.optimizer.Optimizer;
import org.monarch.nn.tensor.Tensor;

import java.util.Random;

public class DenseLayer extends Layer {

    protected final double initStd;
    protected final long seed;

    protected double[] W;    // [nOut, nIn]
    protected double[] bias; // [nOut]
    protected double[] dW;
    protected double[] db;

    protected Tensor lastInput, lastZ, lastOutput;

    protected DenseLayer(Builder b) {
        this.nIn = b.nIn;
        this.nOut = b.nOut;
        this.activationFunction = b.activationFunction;
        this.seed = b.seed;
        this.initStd = b.initStd > 0 ? b.initStd
                : Math.sqrt(2.0 / Math.max(1, nIn));
    }

    protected DenseLayer(Layer.Builder<?, ?> b, double initStd, long seed) {
        this.nIn = b.nIn;
        this.nOut = b.nOut;
        this.activationFunction = b.activationFunction;
        this.seed = seed;
        this.initStd = initStd > 0 ? initStd
                : Math.sqrt(2.0 / Math.max(1, nIn));
    }

    @Override
    public void init(int[] inputShape) {
        if (nIn <= 0) nIn = inputShape[inputShape.length - 1];

        W = new double[nOut * nIn];
        bias = new double[nOut];
        dW = new double[nOut * nIn];
        db = new double[nOut];

        Random rng = new Random(seed);
        for (int i = 0; i < W.length; i++) W[i] = rng.nextGaussian() * initStd;
    }

    @Override
    public int[] outputShape(int[] inputShape) {
        return new int[]{nOut};
    }

    @Override
    public Tensor forward(Tensor input, boolean training) {
        int B = input.getShape()[0];
        int flat = input.size() / B;
        if (flat != nIn)
            throw new IllegalStateException(
                    "Dense expects " + nIn + " features, got " + flat);

        lastInput = input;
        Tensor z = new Tensor(B, nOut);
        Tensor y = new Tensor(B, nOut);

        double[] x = input.getData();
        double[] zD = z.getData();

        ops.denseForward(x, W, bias, zD, B, nIn, nOut);

        if (activationFunction == ActivationFunction.SOFTMAX) {
            applySoftmax(zD, y.getData(), B, nOut);
        } else {
            IActivationFunction fn = ActivationsFactory.get(activationFunction);
            for (int i = 0; i < zD.length; i++)
                y.getData()[i] = fn.output(zD[i]);
        }

        lastZ = z;
        lastOutput = y;
        return y;
    }

    @Override
    public Tensor backward(Tensor gradOutput) {
        int B = lastInput.getShape()[0];
        int flat = lastInput.size() / B;
        Tensor dInput = new Tensor(B, flat);

        double[] dY = gradOutput.getData();
        double[] dX = dInput.getData();
        double[] zD = lastZ.getData();
        double[] yD = lastOutput.getData();

        double[] dZ = new double[B * nOut];

        if (skipActivationDerivative) {
            // Caller already folded the activation derivative into gradOutput
            // (combined softmax+CE or sigmoid+BCE gradient). Pass through.
            System.arraycopy(dY, 0, dZ, 0, dZ.length);
        } else if (activationFunction == ActivationFunction.SOFTMAX) {
            for (int bi = 0; bi < B; bi++) {
                int off = bi * nOut;
                double sumDY = 0;
                for (int o = 0; o < nOut; o++) sumDY += dY[off + o] * yD[off + o];
                for (int o = 0; o < nOut; o++)
                    dZ[off + o] = yD[off + o] * (dY[off + o] - sumDY);
            }
        } else {
            IActivationFunction fn = ActivationsFactory.get(activationFunction);
            for (int i = 0; i < dZ.length; i++)
                dZ[i] = dY[i] * fn.outputDerivative(zD[i]);
        }

        ops.denseBackward(dX, dW, db, dZ, lastInput.getData(), W, B, nIn, nOut);

        return dInput;
    }

    @Override
    public void update(Optimizer opt, double lr, int step, int batchSize) {
        double inv = 1.0 / batchSize;
        for (int i = 0; i < dW.length; i++) dW[i] *= inv;
        for (int i = 0; i < db.length; i++) db[i] *= inv;
        opt.update(W, dW, lr, step);
        opt.update(bias, db, lr, step);
    }

    @Override
    public LayerType getType() {
        return LayerType.DENSE;
    }

    @Override
    public double[] getParams() {
        double[] p = new double[W.length + bias.length];
        System.arraycopy(W, 0, p, 0, W.length);
        System.arraycopy(bias, 0, p, W.length, bias.length);
        return p;
    }

    @Override
    public void setParams(double[] params) {
        int wSize = W.length;
        if (params.length != wSize + bias.length)
            throw new IllegalArgumentException(
                    "Dense setParams size mismatch: expected "
                            + (wSize + bias.length) + ", got " + params.length);
        System.arraycopy(params, 0, W, 0, wSize);
        System.arraycopy(params, wSize, bias, 0, bias.length);
    }

    @Override
    public int parameterCount() {
        // After init — precise count from the allocated arrays.
        if (W != null) return W.length + bias.length;
        // Before init — estimate from declared nIn / nOut.
        int in = Math.max(0, nIn);
        int out = Math.max(0, nOut);
        return in * out + out;
    }

    @Override
    public double[] getGrads() {
        double[] g = new double[dW.length + db.length];
        System.arraycopy(dW, 0, g, 0, dW.length);
        System.arraycopy(db, 0, g, dW.length, db.length);
        return g;
    }

    protected static void applySoftmax(double[] z, double[] y, int B, int D) {
        for (int bi = 0; bi < B; bi++) {
            int off = bi * D;
            double max = z[off];
            for (int i = 1; i < D; i++) if (z[off + i] > max) max = z[off + i];
            double sum = 0;
            for (int i = 0; i < D; i++) {
                double e = Math.exp(z[off + i] - max);
                y[off + i] = e;
                sum += e;
            }
            for (int i = 0; i < D; i++) y[off + i] /= sum;
        }
    }

    public static class Builder extends Layer.Builder<DenseLayer, Builder> {
        private double initStd = -1;
        private long seed = 42;

        public Builder initStd(double s) {
            this.initStd = s;
            return this;
        }

        public Builder seed(long s) {
            this.seed = s;
            return this;
        }

        @Override
        protected Builder self() {
            return this;
        }

        @Override
        public DenseLayer build() {
            return new DenseLayer(this);
        }
    }
}