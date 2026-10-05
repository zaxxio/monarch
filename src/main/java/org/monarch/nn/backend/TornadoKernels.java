package org.monarch.nn.backend;

import uk.ac.manchester.tornado.api.annotations.Parallel;

public final class TornadoKernels {

    private TornadoKernels() {}

    /** Z[b, o] = bias[o] + Σ_i X[b, i] · W[o, i] */
    public static void denseForward(float[] X, float[] W, float[] bias, float[] Z,
                                    int B, int in, int out) {
        for (@Parallel int b = 0; b < B; b++) {
            for (@Parallel int o = 0; o < out; o++) {
                float sum = bias[o];
                int xOff = b * in;
                int wOff = o * in;
                for (int i = 0; i < in; i++) sum += X[xOff + i] * W[wOff + i];
                Z[b * out + o] = sum;
            }
        }
    }

    /** dX[b, i] = Σ_o dZ[b, o] · W[o, i] */
    public static void denseBackwardDX(float[] dZ, float[] W, float[] dX,
                                       int B, int in, int out) {
        for (@Parallel int b = 0; b < B; b++) {
            for (@Parallel int i = 0; i < in; i++) {
                float sum = 0f;
                for (int o = 0; o < out; o++)
                    sum += dZ[b * out + o] * W[o * in + i];
                dX[b * in + i] = sum;
            }
        }
    }

    /** dW[o, i] = Σ_b dZ[b, o] · X[b, i] */
    public static void denseBackwardDW(float[] dZ, float[] X, float[] dW,
                                       int B, int in, int out) {
        for (@Parallel int o = 0; o < out; o++) {
            for (@Parallel int i = 0; i < in; i++) {
                float sum = 0f;
                for (int b = 0; b < B; b++)
                    sum += dZ[b * out + o] * X[b * in + i];
                dW[o * in + i] = sum;
            }
        }
    }

    /** db[o] = Σ_b dZ[b, o] */
    public static void denseBackwardDB(float[] dZ, float[] db, int B, int out) {
        for (@Parallel int o = 0; o < out; o++) {
            float sum = 0f;
            for (int b = 0; b < B; b++) sum += dZ[b * out + o];
            db[o] = sum;
        }
    }
}