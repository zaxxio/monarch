package org.monarch.nn.backend;

import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.math.TornadoMath;

/**
 * Float kernels for a device-resident CNN training step.
 *
 * Rules these kernels follow (learned from Tornado's Graal compiler):
 *  - NO integer '/' or '%' anywhere: chained divisions by loop-invariant
 *    values crash Graal's ReassociationPhase. Indices come from nested
 *    @Parallel loops (max 3 dims) instead of decoding a flat index.
 *  - loop bounds of @Parallel loops are scalar int args
 *  - extra geometry lives in an int[] cfg array
 *  - no double (Apple GPUs have no fp64)
 *  - convolution is stride 1 only
 */
public final class GpuNetKernels {

    private GpuNetKernels() {}

    // ==================================================================
    // Conv2D   cfg = {B, inC, inH, inW, outC, outH, outW, kH, kW, stride, pad, relu}
    // ==================================================================

    public static void convForward(float[] x, float[] w, float[] bias, float[] y,
                                   int[] c, int B, int outC, int outH) {
        int inC = c[1], inH = c[2], inW = c[3];
        int outW = c[6], kH = c[7], kW = c[8], pad = c[10], relu = c[11];

        for (@Parallel int b = 0; b < B; b++) {
            for (@Parallel int oc = 0; oc < outC; oc++) {
                for (@Parallel int oh = 0; oh < outH; oh++) {
                    for (int ow = 0; ow < outW; ow++) {
                        float sum = bias[oc];
                        for (int ic = 0; ic < inC; ic++) {
                            int xBase = (b * inC + ic) * inH * inW;
                            int wBase = (oc * inC + ic) * kH * kW;
                            for (int kh = 0; kh < kH; kh++) {
                                int ih = oh - pad + kh;
                                if (ih >= 0 && ih < inH) {
                                    for (int kw = 0; kw < kW; kw++) {
                                        int iw = ow - pad + kw;
                                        if (iw >= 0 && iw < inW) {
                                            sum += x[xBase + ih * inW + iw]
                                                 * w[wBase + kh * kW + kw];
                                        }
                                    }
                                }
                            }
                        }
                        if (relu == 1 && sum < 0f) sum = 0f;
                        y[((b * outC + oc) * outH + oh) * outW + ow] = sum;
                    }
                }
            }
        }
    }

    /** dW[oc, ic, kh, kw]: parallel over (oc, ic, kh), serial over kw. */
    public static void convBackwardW(float[] dz, float[] x, float[] dw,
                                     int[] c, int outC, int inC, int kH) {
        int B = c[0], inH = c[2], inW = c[3];
        int outH = c[5], outW = c[6], kW = c[8], pad = c[10];

        for (@Parallel int oc = 0; oc < outC; oc++) {
            for (@Parallel int ic = 0; ic < inC; ic++) {
                for (@Parallel int kh = 0; kh < kH; kh++) {
                    for (int kw = 0; kw < kW; kw++) {
                        float sum = 0f;
                        for (int b = 0; b < B; b++) {
                            for (int oh = 0; oh < outH; oh++) {
                                int ih = oh - pad + kh;
                                if (ih >= 0 && ih < inH) {
                                    int dzRow = ((b * outC + oc) * outH + oh) * outW;
                                    int xRow = ((b * inC + ic) * inH + ih) * inW;
                                    for (int ow = 0; ow < outW; ow++) {
                                        int iw = ow - pad + kw;
                                        if (iw >= 0 && iw < inW) {
                                            sum += dz[dzRow + ow] * x[xRow + iw];
                                        }
                                    }
                                }
                            }
                        }
                        dw[((oc * inC + ic) * kH + kh) * kW + kw] = sum;
                    }
                }
            }
        }
    }

    public static void convBackwardB(float[] dz, float[] db, int[] c, int outC) {
        int B = c[0], outH = c[5], outW = c[6];
        int plane = outH * outW;
        for (@Parallel int oc = 0; oc < outC; oc++) {
            float sum = 0f;
            for (int b = 0; b < B; b++) {
                int base = (b * outC + oc) * plane;
                for (int p = 0; p < plane; p++) sum += dz[base + p];
            }
            db[oc] = sum;
        }
    }

    /** dX: parallel over (b, ic, ih), serial over iw. Stride 1 only. */
    public static void convBackwardX(float[] dz, float[] w, float[] dx,
                                     int[] c, int B, int inC, int inH) {
        int inW = c[3], outC = c[4], outH = c[5], outW = c[6];
        int kH = c[7], kW = c[8], pad = c[10];

        for (@Parallel int b = 0; b < B; b++) {
            for (@Parallel int ic = 0; ic < inC; ic++) {
                for (@Parallel int ih = 0; ih < inH; ih++) {
                    for (int iw = 0; iw < inW; iw++) {
                        float sum = 0f;
                        for (int oc = 0; oc < outC; oc++) {
                            for (int kh = 0; kh < kH; kh++) {
                                int oh = ih + pad - kh;
                                if (oh >= 0 && oh < outH) {
                                    for (int kw = 0; kw < kW; kw++) {
                                        int ow = iw + pad - kw;
                                        if (ow >= 0 && ow < outW) {
                                            sum += dz[((b * outC + oc) * outH + oh) * outW + ow]
                                                 * w[((oc * inC + ic) * kH + kh) * kW + kw];
                                        }
                                    }
                                }
                            }
                        }
                        dx[((b * inC + ic) * inH + ih) * inW + iw] = sum;
                    }
                }
            }
        }
    }

    // ==================================================================
    // Activation backward (ReLU): dz = g where y > 0 else 0
    // ==================================================================
    public static void actBackward(float[] g, float[] y, float[] dz, int relu, int n) {
        for (@Parallel int i = 0; i < n; i++) {
            float gi = g[i];
            if (relu == 1 && y[i] <= 0f) gi = 0f;
            dz[i] = gi;
        }
    }

    public static void zero(float[] a, int n) {
        for (@Parallel int i = 0; i < n; i++) a[i] = 0f;
    }

    // ==================================================================
    // MaxPool (square window, stride == pool)
    // cfg = {B, C, inH, inW, outH, outW, pool, stride}
    // ==================================================================

    /** bc = B * C. arg[] stores the flat input index of each window's max. */
    public static void poolForward(float[] x, float[] y, int[] arg, int[] c,
                                   int bc, int outH, int outW) {
        int inH = c[2], inW = c[3], pool = c[6], stride = c[7];

        for (@Parallel int plane = 0; plane < bc; plane++) {
            for (@Parallel int oh = 0; oh < outH; oh++) {
                for (@Parallel int ow = 0; ow < outW; ow++) {
                    int base = plane * inH * inW;
                    float best = -3.0e38f;
                    int bestIdx = base;
                    for (int kh = 0; kh < pool; kh++) {
                        for (int kw = 0; kw < pool; kw++) {
                            int p = base + (oh * stride + kh) * inW + (ow * stride + kw);
                            float v = x[p];
                            if (v > best) {
                                best = v;
                                bestIdx = p;
                            }
                        }
                    }
                    int o = (plane * outH + oh) * outW + ow;
                    y[o] = best;
                    arg[o] = bestIdx;
                }
            }
        }
    }

    /** Windows don't overlap (stride == pool), so scatter is race-free. dx must be zeroed first. */
    public static void poolScatter(float[] g, int[] arg, float[] dx, int n) {
        for (@Parallel int o = 0; o < n; o++) {
            dx[arg[o]] = g[o];
        }
    }

    // ==================================================================
    // Dense   cfg = {B, in, out, relu}
    // ==================================================================

    public static void denseForward(float[] x, float[] w, float[] bias, float[] y,
                                    int[] c, int B, int out) {
        int in = c[1], relu = c[3];
        for (@Parallel int b = 0; b < B; b++) {
            for (@Parallel int o = 0; o < out; o++) {
                float sum = bias[o];
                int xOff = b * in;
                int wOff = o * in;
                for (int i = 0; i < in; i++) sum += x[xOff + i] * w[wOff + i];
                if (relu == 1 && sum < 0f) sum = 0f;
                y[b * out + o] = sum;
            }
        }
    }

    public static void denseBackwardW(float[] dz, float[] x, float[] dw,
                                      int[] c, int out, int in) {
        int B = c[0];
        for (@Parallel int o = 0; o < out; o++) {
            for (@Parallel int i = 0; i < in; i++) {
                float sum = 0f;
                for (int b = 0; b < B; b++) sum += dz[b * out + o] * x[b * in + i];
                dw[o * in + i] = sum;
            }
        }
    }

    public static void denseBackwardB(float[] dz, float[] db, int[] c, int out) {
        int B = c[0];
        for (@Parallel int o = 0; o < out; o++) {
            float sum = 0f;
            for (int b = 0; b < B; b++) sum += dz[b * out + o];
            db[o] = sum;
        }
    }

    public static void denseBackwardX(float[] dz, float[] w, float[] dx,
                                      int[] c, int B, int in) {
        int out = c[2];
        for (@Parallel int b = 0; b < B; b++) {
            for (@Parallel int i = 0; i < in; i++) {
                float sum = 0f;
                for (int o = 0; o < out; o++) sum += dz[b * out + o] * w[o * in + i];
                dx[b * in + i] = sum;
            }
        }
    }

    // ==================================================================
    // Softmax + cross-entropy (log-sum-exp form, no division).
    // dlogits = (softmax - onehot) * wgt[b]; loss[b] unweighted.
    // ==================================================================
    public static void softmaxCE(float[] logits, int[] labels, float[] wgt,
                                 float[] dlogits, float[] loss, int B, int C) {
        for (@Parallel int b = 0; b < B; b++) {
            int off = b * C;
            float mx = logits[off];
            for (int k = 1; k < C; k++) {
                if (logits[off + k] > mx) mx = logits[off + k];
            }
            float sum = 0f;
            for (int k = 0; k < C; k++) sum += TornadoMath.exp(logits[off + k] - mx);
            float lse = mx + TornadoMath.log(sum);

            int lab = labels[b];
            loss[b] = lse - logits[off + lab];

            float wb = wgt[b];
            for (int k = 0; k < C; k++) {
                float p = TornadoMath.exp(logits[off + k] - lse);
                float t = 0f;
                if (k == lab) t = 1f;
                dlogits[off + k] = (p - t) * wb;
            }
        }
    }

    // ==================================================================
    // Adam.  hyper = {lrEff, beta1, beta2, eps}; lrEff includes bias correction.
    // ==================================================================
    public static void adam(float[] p, float[] g, float[] m, float[] v,
                            float[] hyper, int n) {
        for (@Parallel int i = 0; i < n; i++) {
            float lrEff = hyper[0];
            float b1 = hyper[1];
            float b2 = hyper[2];
            float eps = hyper[3];
            float gi = g[i];
            float mi = b1 * m[i] + (1f - b1) * gi;
            float vi = b2 * v[i] + (1f - b2) * gi * gi;
            m[i] = mi;
            v[i] = vi;
            p[i] = p[i] - lrEff * mi / (TornadoMath.sqrt(vi) + eps);
        }
    }
}