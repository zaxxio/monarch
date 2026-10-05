package org.monarch.nn.layer;

import org.monarch.nn.activation.ActivationsFactory;
import org.monarch.nn.activation.IActivationFunction;
import org.monarch.nn.optimizer.Optimizer;
import org.monarch.nn.optimizer.Parallel;
import org.monarch.nn.tensor.Tensor;

import java.util.Arrays;
import java.util.Random;

/**
 * 2D convolution.
 *
 * Input : [B, C_in, H, W]
 * Filters : [C_out, C_in, K_h, K_w]
 * Output: [B, C_out, H_out, W_out]
 *
 *   H_out = (H + 2*padding - K_h) / stride + 1
 *   W_out = (W + 2*padding - K_w) / stride + 1
 *
 * Optimized path: caches all strides, accesses the raw data arrays
 * directly (no Tensor.get()/set() overhead), and runs the batch
 * dimension across all CPU cores.
 */
public class Conv2DLayer extends Layer {

    private final int inChannels, outChannels;
    private final int kernelH, kernelW;
    private final int stride, padding;
    private final double initStd;
    private final long seed;

    private int inH, inW, outH, outW;

    private Tensor filters;        // [C_out, C_in, K_h, K_w]
    private double[] biases;       // [C_out]

    private Tensor filterGrad;
    private double[] biasGrad;

    private Tensor lastInput;
    private Tensor lastZ;

    protected Conv2DLayer(Builder b) {
        this.inChannels   = b.inChannels;
        this.outChannels  = b.outChannels;
        this.kernelH      = b.kernelH;
        this.kernelW      = b.kernelW;
        this.stride       = b.stride;
        this.padding      = b.padding;
        this.activationFunction = b.activationFunction;
        this.nIn          = inChannels;
        this.nOut         = outChannels;
        this.seed         = b.seed;
        this.initStd      = b.initStd > 0 ? b.initStd
                : Math.sqrt(2.0 / (inChannels * kernelH * kernelW));
    }

    @Override
    public LayerType getType() { return LayerType.CONV2D; }

    @Override
    public void init(int[] inputShape) {
        if (inputShape.length != 3)
            throw new IllegalArgumentException(
                    "Conv2D expects [C,H,W] per-sample shape, got "
                    + Arrays.toString(inputShape));
        if (inputShape[0] != inChannels)
            throw new IllegalArgumentException(
                    "Conv2D expects " + inChannels + " input channels, got "
                    + inputShape[0]);

        this.inH = inputShape[1];
        this.inW = inputShape[2];
        this.outH = (inH + 2 * padding - kernelH) / stride + 1;
        this.outW = (inW + 2 * padding - kernelW) / stride + 1;

        if (outH <= 0 || outW <= 0)
            throw new IllegalStateException(
                    "Conv2D produces invalid output size: "
                    + outH + "x" + outW);

        Random rng = new Random(seed);
        filters    = new Tensor(outChannels, inChannels, kernelH, kernelW);
        filterGrad = new Tensor(outChannels, inChannels, kernelH, kernelW);
        biases     = new double[outChannels];
        biasGrad   = new double[outChannels];

        double[] f = filters.getData();
        for (int i = 0; i < f.length; i++) f[i] = rng.nextGaussian() * initStd;
    }

    @Override
    public int[] outputShape(int[] inputShape) {
        int h = inputShape[1], w = inputShape[2];
        return new int[]{
                outChannels,
                (h + 2 * padding - kernelH) / stride + 1,
                (w + 2 * padding - kernelW) / stride + 1
        };
    }

    // ==================================================================
    // Forward
    // ==================================================================
    @Override
    public Tensor forward(Tensor input, boolean training) {
        int[] s = input.getShape();
        if (s.length != 4 || s[1] != inChannels || s[2] != inH || s[3] != inW)
            throw new IllegalArgumentException(
                    "Conv2D input mismatch: " + Arrays.toString(s)
                    + " (expected [B, " + inChannels + ", " + inH + ", " + inW + "])");

        lastInput = input;
        final int B = s[0];

        final Tensor z = new Tensor(B, outChannels, outH, outW);
        final Tensor y = new Tensor(B, outChannels, outH, outW);

        final double[] xD = input.getData();
        final double[] fD = filters.getData();
        final double[] zD = z.getData();
        final double[] yD = y.getData();
        final IActivationFunction fn = ActivationsFactory.get(activationFunction);

        final int xSampleStride = inChannels * inH * inW;
        final int zSampleStride = outChannels * outH * outW;
        final int fOutStride    = inChannels * kernelH * kernelW;
        final int xChanStride   = inH * inW;
        final int fChanStride   = kernelH * kernelW;
        final int zOutStride    = outH * outW;

        Parallel.forRange(B, b -> {
            int xSampleBase = b * xSampleStride;
            int zSampleBase = b * zSampleStride;

            for (int oc = 0; oc < outChannels; oc++) {
                double bias = biases[oc];
                int fOutBase = oc * fOutStride;
                int zOutBase = zSampleBase + oc * zOutStride;

                for (int oh = 0; oh < outH; oh++) {
                    int zRowBase = zOutBase + oh * outW;
                    for (int ow = 0; ow < outW; ow++) {
                        double sum = bias;
                        int ohS = oh * stride - padding;

                        for (int ic = 0; ic < inChannels; ic++) {
                            int xChannelBase = xSampleBase + ic * xChanStride;
                            int fChannelBase = fOutBase + ic * fChanStride;

                            for (int kh = 0; kh < kernelH; kh++) {
                                int ih = ohS + kh;
                                if (ih < 0 || ih >= inH) continue;
                                int xRowBase = xChannelBase + ih * inW;
                                int fRowBase = fChannelBase + kh * kernelW;

                                for (int kw = 0; kw < kernelW; kw++) {
                                    int iw = ow * stride - padding + kw;
                                    if (iw < 0 || iw >= inW) continue;
                                    sum += xD[xRowBase + iw] * fD[fRowBase + kw];
                                }
                            }
                        }
                        int idx = zRowBase + ow;
                        zD[idx] = sum;
                        yD[idx] = fn.output(sum);
                    }
                }
            }
        });

        lastZ = z;
        return y;
    }

    // ==================================================================
    // Backward
    // ==================================================================
    @Override
    public Tensor backward(Tensor gradOutput) {
        final int B = lastInput.getShape()[0];
        Tensor dInput = new Tensor(B, inChannels, inH, inW);

        final double[] xD    = lastInput.getData();
        final double[] fD    = filters.getData();
        final double[] gD    = gradOutput.getData();
        final double[] zD    = lastZ.getData();
        final double[] dXD   = dInput.getData();

        final IActivationFunction fn = ActivationsFactory.get(activationFunction);

        final int xSampleStride = inChannels * inH * inW;
        final int gSampleStride = outChannels * outH * outW;
        final int fOutStride    = inChannels * kernelH * kernelW;
        final int xChanStride   = inH * inW;
        final int fChanStride   = kernelH * kernelW;
        final int gOutStride    = outH * outW;

        // Per-thread gradient accumulators. Avoids atomic contention
        // on filterGrad / biasGrad, which are shared across all samples.
        final int threads = Parallel.numThreads();
        final double[][] localWG = new double[threads][filters.size()];
        final double[][] localBG = new double[threads][outChannels];

        Parallel.forRange(B, b -> {
            // Bucket by thread ID. Two threads may share a bucket if
            // the thread count exceeds `threads`, but each bucket is
            // still a valid partial sum — the final reduce combines them.
            int tid = (int) (Thread.currentThread().threadId() % threads);
            double[] lwg = localWG[tid];
            double[] lbg = localBG[tid];

            int xSampleBase = b * xSampleStride;
            int gSampleBase = b * gSampleStride;

            for (int oc = 0; oc < outChannels; oc++) {
                int fOutBase = oc * fOutStride;
                int gOutBase = gSampleBase + oc * gOutStride;

                for (int oh = 0; oh < outH; oh++) {
                    int gRowBase = gOutBase + oh * outW;
                    int ohS = oh * stride - padding;

                    for (int ow = 0; ow < outW; ow++) {
                        int gIdx = gRowBase + ow;
                        double dZ = gD[gIdx] * fn.outputDerivative(zD[gIdx]);
                        lbg[oc] += dZ;

                        for (int ic = 0; ic < inChannels; ic++) {
                            int xChannelBase = xSampleBase + ic * xChanStride;
                            int fChannelBase = fOutBase + ic * fChanStride;

                            for (int kh = 0; kh < kernelH; kh++) {
                                int ih = ohS + kh;
                                if (ih < 0 || ih >= inH) continue;
                                int xRowBase = xChannelBase + ih * inW;
                                int dXRowBase = xChannelBase + ih * inW;
                                int fRowBase = fChannelBase + kh * kernelW;

                                for (int kw = 0; kw < kernelW; kw++) {
                                    int iw = ow * stride - padding + kw;
                                    if (iw < 0 || iw >= inW) continue;
                                    int fIdx = fRowBase + kw;
                                    double xVal = xD[xRowBase + iw];
                                    lwg[fIdx] += dZ * xVal;
                                    dXD[dXRowBase + iw] += dZ * fD[fIdx];
                                }
                            }
                        }
                    }
                }
            }
        });

        // Reduce thread-local gradients into the layer's buffers.
        filterGrad.zero();
        Arrays.fill(biasGrad, 0.0);
        double[] fgD = filterGrad.getData();
        for (int t = 0; t < threads; t++) {
            double[] lwg = localWG[t];
            double[] lbg = localBG[t];
            for (int i = 0; i < lwg.length; i++) fgD[i] += lwg[i];
            for (int i = 0; i < lbg.length; i++) biasGrad[i] += lbg[i];
        }

        return dInput;
    }

    // ==================================================================
    // Update
    // ==================================================================
    @Override
    public void update(Optimizer opt, double lr, int step, int batchSize) {
        double inv = 1.0 / batchSize;

        int wSize = filters.size();
        double[] p = new double[wSize + outChannels];
        double[] g = new double[wSize + outChannels];

        System.arraycopy(filters.getData(), 0, p, 0, wSize);
        System.arraycopy(biases, 0, p, wSize, outChannels);

        System.arraycopy(filterGrad.getData(), 0, g, 0, wSize);
        for (int i = 0; i < wSize; i++) g[i] *= inv;
        for (int i = 0; i < outChannels; i++) g[wSize + i] = biasGrad[i] * inv;

        opt.update(p, g, lr, step);

        System.arraycopy(p, 0, filters.getData(), 0, wSize);
        System.arraycopy(p, wSize, biases, 0, outChannels);
    }

    // ==================================================================
    @Override
    public double[] getParams() {
        int wSize = filters.size();
        double[] p = new double[wSize + outChannels];
        System.arraycopy(filters.getData(), 0, p, 0, wSize);
        System.arraycopy(biases, 0, p, wSize, outChannels);
        return p;
    }

    @Override
    public void setParams(double[] params) {
        int wSize = filters.size();
        if (params.length != wSize + outChannels)
            throw new IllegalArgumentException(
                    "Conv2D setParams size mismatch: expected "
                            + (wSize + outChannels) + ", got " + params.length);
        System.arraycopy(params, 0, filters.getData(), 0, wSize);
        System.arraycopy(params, wSize, biases, 0, outChannels);
    }

    @Override
    public int parameterCount() {
        if (filters != null) return filters.size() + outChannels;
        return inChannels * outChannels * kernelH * kernelW + outChannels;
    }

    @Override
    public double[] getGrads() {
        int wSize = filters.size();
        double[] g = new double[wSize + outChannels];
        System.arraycopy(filterGrad.getData(), 0, g, 0, wSize);
        System.arraycopy(biasGrad, 0, g, wSize, outChannels);
        return g;
    }

    // ==================================================================
    public static class Builder extends Layer.Builder<Conv2DLayer, Builder> {

        private int inChannels, outChannels;
        private int kernelH = 3, kernelW = 3;
        private int stride = 1, padding = 0;
        private double initStd = -1;
        private long seed = 42;

        public Builder inChannels(int v)        { this.inChannels = v;  return this; }
        public Builder outChannels(int v)       { this.outChannels = v; return this; }
        public Builder kernelSize(int h, int w) { kernelH = h; kernelW = w; return this; }
        public Builder kernelSize(int k)        { return kernelSize(k, k); }
        public Builder stride(int v)            { this.stride = v;      return this; }
        public Builder padding(int v)           { this.padding = v;     return this; }
        public Builder initStd(double v)        { this.initStd = v;     return this; }
        public Builder seed(long s)             { this.seed = s;        return this; }

        @Override protected Builder self() { return this; }
        @Override public Conv2DLayer build() { return new Conv2DLayer(this); }
    }
}