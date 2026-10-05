package org.monarch.nn.layer;

import org.monarch.nn.optimizer.Optimizer;
import org.monarch.nn.optimizer.Parallel;
import org.monarch.nn.tensor.Tensor;

public class MaxPoolingLayer extends Layer {

    private final int poolH, poolW, stride;
    private int inH, inW, outH, outW;
    private Tensor lastInput;

    /**
     * Flat argmax buffer: [B][C][outH][outW] flattened row-major.
     * Encoding: argmax index within the pool window (0..poolH*poolW-1).
     * int[] is faster than int[][][][] and lets the backward pass
     * process with a simple linear scan.
     */
    private int[] argmax;

    protected MaxPoolingLayer(Builder b) {
        this.poolH = b.poolH;
        this.poolW = b.poolW;
        this.stride = b.stride;
    }

    @Override
    public void init(int[] inputShape) {
        if (inputShape.length != 3)
            throw new IllegalArgumentException("MaxPool expects [C,H,W]");
        this.inH = inputShape[1];
        this.inW = inputShape[2];
        this.outH = (inH - poolH) / stride + 1;
        this.outW = (inW - poolW) / stride + 1;

        if (outH <= 0 || outW <= 0)
            throw new IllegalStateException(
                    "MaxPool produces invalid output size: "
                    + outH + "x" + outW);
    }

    @Override
    public int[] outputShape(int[] inputShape) {
        int h = inputShape[1], w = inputShape[2];
        return new int[]{
                inputShape[0],
                (h - poolH) / stride + 1,
                (w - poolW) / stride + 1
        };
    }

    // ==================================================================
    // Forward
    // ==================================================================
    @Override
    public Tensor forward(Tensor input, boolean training) {
        final int B = input.getShape()[0];
        final int C = input.getShape()[1];
        lastInput = input;

        final Tensor out = new Tensor(B, C, outH, outW);
        final double[] inD  = input.getData();
        final double[] outD = out.getData();

        final int inSampleStride  = C * inH * inW;
        final int outSampleStride = C * outH * outW;
        final int inChanStride    = inH * inW;
        final int outChanStride   = outH * outW;

        final int[] am = new int[B * C * outH * outW];
        this.argmax = am;

        Parallel.forRange(B, b -> {
            int inSampleBase  = b * inSampleStride;
            int outSampleBase = b * outSampleStride;
            int amSampleBase  = b * C * outH * outW;

            for (int c = 0; c < C; c++) {
                int inChanBase  = inSampleBase + c * inChanStride;
                int outChanBase = outSampleBase + c * outChanStride;
                int amChanBase  = amSampleBase + c * outH * outW;

                for (int oh = 0; oh < outH; oh++) {
                    int outRowBase = outChanBase + oh * outW;
                    int amRowBase  = amChanBase + oh * outW;
                    int ohS = oh * stride;

                    for (int ow = 0; ow < outW; ow++) {
                        int owS = ow * stride;
                        double max = -Double.MAX_VALUE;
                        int arg = 0;

                        for (int kh = 0; kh < poolH; kh++) {
                            int inRowBase = inChanBase + (ohS + kh) * inW;
                            for (int kw = 0; kw < poolW; kw++) {
                                double v = inD[inRowBase + owS + kw];
                                if (v > max) {
                                    max = v;
                                    arg = kh * poolW + kw;
                                }
                            }
                        }
                        outD[outRowBase + ow] = max;
                        am[amRowBase + ow] = arg;
                    }
                }
            }
        });

        return out;
    }

    // ==================================================================
    // Backward
    // ==================================================================
    @Override
    public Tensor backward(Tensor gradOutput) {
        final int B = lastInput.getShape()[0];
        final int C = lastInput.getShape()[1];
        final Tensor dInput = new Tensor(B, C, inH, inW);

        final double[] gD  = gradOutput.getData();
        final double[] dXD = dInput.getData();
        final int[] am = this.argmax;

        final int gSampleStride  = C * outH * outW;
        final int dXSampleStride = C * inH * inW;
        final int gChanStride    = outH * outW;
        final int dXChanStride   = inH * inW;

        Parallel.forRange(B, b -> {
            int gSampleBase  = b * gSampleStride;
            int dXSampleBase = b * dXSampleStride;
            int amSampleBase = b * C * outH * outW;

            for (int c = 0; c < C; c++) {
                int gChanBase  = gSampleBase  + c * gChanStride;
                int dXChanBase = dXSampleBase + c * dXChanStride;
                int amChanBase = amSampleBase + c * outH * outW;

                for (int oh = 0; oh < outH; oh++) {
                    int gRowBase  = gChanBase + oh * outW;
                    int amRowBase = amChanBase + oh * outW;
                    int ohS = oh * stride;

                    for (int ow = 0; ow < outW; ow++) {
                        int arg = am[amRowBase + ow];
                        int kh = arg / poolW;
                        int kw = arg % poolW;
                        int ih = ohS + kh;
                        int iw = ow * stride + kw;
                        dXD[dXChanBase + ih * inW + iw] += gD[gRowBase + ow];
                    }
                }
            }
        });

        return dInput;
    }

    // ==================================================================
    @Override
    public void update(Optimizer opt, double lr, int step, int batchSize) { }

    @Override public LayerType getType() { return LayerType.MAXPOOL; }

    public static class Builder extends Layer.Builder<MaxPoolingLayer, Builder> {
        private int poolH = 2, poolW = 2, stride = 2;
        public Builder poolSize(int h, int w) { poolH = h; poolW = w; return this; }
        public Builder stride(int s) { this.stride = s; return this; }
        @Override protected Builder self() { return this; }
        @Override public MaxPoolingLayer build() { return new MaxPoolingLayer(this); }
    }
}