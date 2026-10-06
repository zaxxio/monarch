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
 * Loop order is (oc, ic, kh, kw) outside and a contiguous run over
 * output columns inside, with the horizontal padding bounds hoisted out
 * of the inner loop. The JIT can auto-vectorize that axpy loop.
 *
 * Backward has no shared accumulators:
 *   dW / db : parallel over (oc, ic) pairs, each owns its filter slice
 *   dX      : parallel over batch, each sample owns its dX slice
 */
public class Conv2DLayer extends Layer {

    private final int inChannels, outChannels;
    private final int kernelH, kernelW;
    private final int stride, padding;
    private final double initStd;
    private final long seed;
    private final boolean computeInputGrad;

    private int inH, inW, outH, outW;

    // Valid output-column range per kernel column kw (hoisted bounds check)
    private int[] owLo, owHi;

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
        this.computeInputGrad = b.computeInputGrad;
        this.initStd      = b.initStd > 0 ? b.initStd
                : Math.sqrt(2.0 / (inChannels * kernelH * kernelW));
    }

    @Override
    public LayerType getType() { return LayerType.CONV2D; }

    public int getKernelH() { return kernelH; }
    public int getKernelW() { return kernelW; }
    public int getStride()  { return stride; }
    public int getPadding() { return padding; }

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

        // For each kw, the output columns whose input column is in range:
        //   iw = ow*stride - padding + kw  in [0, inW)
        owLo = new int[kernelW];
        owHi = new int[kernelW];
        for (int kw = 0; kw < kernelW; kw++) {
            int lo = (padding - kw) > 0 ? (padding - kw + stride - 1) / stride : 0;
            int num = inW - 1 + padding - kw;
            int hi = num >= 0 ? Math.min(outW, num / stride + 1) : 0;
            owLo[kw] = lo;
            owHi[kw] = hi;
        }

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
        final double[] bias = biases;
        final int[] lo = owLo, hi = owHi;
        final IActivationFunction fn = ActivationsFactory.get(activationFunction);

        final int xSampleStride = inChannels * inH * inW;
        final int zSampleStride = outChannels * outH * outW;
        final int fOutStride    = inChannels * kernelH * kernelW;
        final int xChanStride   = inH * inW;
        final int fChanStride   = kernelH * kernelW;
        final int zOutStride    = outH * outW;

        Parallel.forRange(B, b -> {
            final int xBase = b * xSampleStride;
            final int zBase = b * zSampleStride;

            for (int oc = 0; oc < outChannels; oc++) {
                final int zOc = zBase + oc * zOutStride;
                Arrays.fill(zD, zOc, zOc + zOutStride, bias[oc]);
                final int fOc = oc * fOutStride;

                for (int ic = 0; ic < inChannels; ic++) {
                    final int xIc = xBase + ic * xChanStride;
                    final int fIc = fOc + ic * fChanStride;

                    for (int kh = 0; kh < kernelH; kh++) {
                        for (int kw = 0; kw < kernelW; kw++) {
                            final double w = fD[fIc + kh * kernelW + kw];
                            final int cLo = lo[kw], cHi = hi[kw];
                            if (cLo >= cHi) continue;

                            for (int oh = 0; oh < outH; oh++) {
                                int ih = oh * stride - padding + kh;
                                if (ih < 0 || ih >= inH) continue;
                                int zRow = zOc + oh * outW;
                                int xRow = xIc + ih * inW - padding + kw;

                                if (stride == 1) {
                                    for (int ow = cLo; ow < cHi; ow++)
                                        zD[zRow + ow] += w * xD[xRow + ow];
                                } else {
                                    for (int ow = cLo; ow < cHi; ow++)
                                        zD[zRow + ow] += w * xD[xRow + ow * stride];
                                }
                            }
                        }
                    }
                }
            }

            // Activation pass
            for (int i = zBase, end = zBase + zSampleStride; i < end; i++)
                yD[i] = fn.output(zD[i]);
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
        final Tensor dInput = new Tensor(B, inChannels, inH, inW);

        final double[] xD  = lastInput.getData();
        final double[] fD  = filters.getData();
        final double[] gD  = gradOutput.getData();
        final double[] zD  = lastZ.getData();
        final double[] dXD = dInput.getData();
        final double[] fgD = filterGrad.getData();
        final double[] bg  = biasGrad;
        final int[] lo = owLo, hi = owHi;

        final IActivationFunction fn = ActivationsFactory.get(activationFunction);

        final int xSampleStride = inChannels * inH * inW;
        final int gSampleStride = outChannels * outH * outW;
        final int fOutStride    = inChannels * kernelH * kernelW;
        final int xChanStride   = inH * inW;
        final int fChanStride   = kernelH * kernelW;
        final int gOutStride    = outH * outW;

        // ---- dZ = dY * f'(z) ----
        final double[] dZ = new double[B * gSampleStride];
        Parallel.forRange(B, b -> {
            for (int i = b * gSampleStride, end = i + gSampleStride; i < end; i++)
                dZ[i] = gD[i] * fn.outputDerivative(zD[i]);
        });

        // ---- db: parallel over oc, each owns bg[oc] ----
        Parallel.forRange(outChannels, oc -> {
            double sum = 0;
            for (int b = 0; b < B; b++) {
                int base = b * gSampleStride + oc * gOutStride;
                for (int i = 0; i < gOutStride; i++) sum += dZ[base + i];
            }
            bg[oc] = sum;
        });

        // ---- dW: one task per filter element (oc, ic, kh, kw) ----
        // Fine-grained so it parallelizes even for small channel counts,
        // and 4 independent accumulators break the add-latency chain.
        final int nFilter = outChannels * fOutStride;
        Parallel.forRange(nFilter, f -> {
            final int oc   = f / fOutStride;
            final int rem  = f - oc * fOutStride;
            final int ic   = rem / fChanStride;
            final int rem2 = rem - ic * fChanStride;
            final int kh   = rem2 / kernelW;
            final int kw   = rem2 - kh * kernelW;
            final int cLo = lo[kw], cHi = hi[kw];

            double s0 = 0, s1 = 0, s2 = 0, s3 = 0;

            if (cLo < cHi) {
                for (int b = 0; b < B; b++) {
                    final int gOc = b * gSampleStride + oc * gOutStride;
                    final int xIc = b * xSampleStride + ic * xChanStride;
                    for (int oh = 0; oh < outH; oh++) {
                        int ih = oh * stride - padding + kh;
                        if (ih < 0 || ih >= inH) continue;
                        final int gRow = gOc + oh * outW;
                        final int xRow = xIc + ih * inW - padding + kw;

                        int ow = cLo;
                        for (; ow + 3 < cHi; ow += 4) {
                            s0 += dZ[gRow + ow]     * xD[xRow + ow * stride];
                            s1 += dZ[gRow + ow + 1] * xD[xRow + (ow + 1) * stride];
                            s2 += dZ[gRow + ow + 2] * xD[xRow + (ow + 2) * stride];
                            s3 += dZ[gRow + ow + 3] * xD[xRow + (ow + 3) * stride];
                        }
                        for (; ow < cHi; ow++)
                            s0 += dZ[gRow + ow] * xD[xRow + ow * stride];
                    }
                }
            }
            fgD[f] = (s0 + s1) + (s2 + s3);
        });

        // ---- dX: parallel over batch, each sample owns its slice ----
        // Skipped for the first conv (nothing consumes the gradient).
        if (computeInputGrad) {
            Parallel.forRange(B, b -> {
                final int dxBase = b * xSampleStride;
                final int gBase  = b * gSampleStride;

                for (int oc = 0; oc < outChannels; oc++) {
                    final int gOc = gBase + oc * gOutStride;
                    final int fOc = oc * fOutStride;

                    for (int ic = 0; ic < inChannels; ic++) {
                        final int dxIc = dxBase + ic * xChanStride;
                        final int fIc  = fOc + ic * fChanStride;

                        for (int kh = 0; kh < kernelH; kh++) {
                            for (int kw = 0; kw < kernelW; kw++) {
                                final double w = fD[fIc + kh * kernelW + kw];
                                final int cLo = lo[kw], cHi = hi[kw];
                                if (cLo >= cHi) continue;

                                for (int oh = 0; oh < outH; oh++) {
                                    int ih = oh * stride - padding + kh;
                                    if (ih < 0 || ih >= inH) continue;
                                    int gRow  = gOc + oh * outW;
                                    int dxRow = dxIc + ih * inW - padding + kw;

                                    if (stride == 1) {
                                        for (int ow = cLo; ow < cHi; ow++)
                                            dXD[dxRow + ow] += w * dZ[gRow + ow];
                                    } else {
                                        for (int ow = cLo; ow < cHi; ow++)
                                            dXD[dxRow + ow * stride] += w * dZ[gRow + ow];
                                    }
                                }
                            }
                        }
                    }
                }
            });
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
        private boolean computeInputGrad = true;

        public Builder inChannels(int v)        { this.inChannels = v;  return this; }
        public Builder outChannels(int v)       { this.outChannels = v; return this; }
        public Builder kernelSize(int h, int w) { kernelH = h; kernelW = w; return this; }
        public Builder kernelSize(int k)        { return kernelSize(k, k); }
        public Builder stride(int v)            { this.stride = v;      return this; }
        public Builder padding(int v)           { this.padding = v;     return this; }
        public Builder initStd(double v)        { this.initStd = v;     return this; }
        public Builder seed(long s)             { this.seed = s;        return this; }

        /** Set false for the first conv: its input gradient is never used. */
        public Builder computeInputGrad(boolean v) { this.computeInputGrad = v; return this; }

        @Override protected Builder self() { return this; }
        @Override public Conv2DLayer build() { return new Conv2DLayer(this); }
    }
}