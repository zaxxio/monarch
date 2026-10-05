package org.monarch.nn.backend;

import org.monarch.nn.optimizer.Parallel;

import java.util.Arrays;

public final class CpuTensorOps implements TensorOps {

    public static final CpuTensorOps INSTANCE = new CpuTensorOps();

    private CpuTensorOps() {}

    // ------------------------------------------------------------------
    // Per-thread gradient scratch for dense backward.
    //
    // Allocated lazily on first use and reused across calls so the hot
    // path doesn't allocate (and garbage-collect) hundreds of megabytes
    // every batch. Cleared to zero at the start of each reuse.
    // ------------------------------------------------------------------
    private double[][] dwCache;
    private double[][] dbCache;
    private int dwCacheSize;
    private int dbCacheSize;
    private int cacheThreads;

    private double[][] localDW(int size) {
        int t = Parallel.numThreads();
        if (dwCache == null || dwCacheSize != size || cacheThreads != t) {
            dwCache = new double[t][size];
            dwCacheSize = size;
            cacheThreads = t;
        } else {
            for (double[] row : dwCache) Arrays.fill(row, 0.0);
        }
        return dwCache;
    }

    private double[][] localDB(int size) {
        int t = Parallel.numThreads();
        if (dbCache == null || dbCacheSize != size || cacheThreads != t) {
            dbCache = new double[t][size];
            dbCacheSize = size;
            cacheThreads = t;
        } else {
            for (double[] row : dbCache) Arrays.fill(row, 0.0);
        }
        return dbCache;
    }

    // ------------------------------------------------------------------
    // Forward: Z = X @ W^T + b
    // ------------------------------------------------------------------
    @Override
    public void denseForward(double[] X, double[] W, double[] b, double[] Z,
                             int B, int in, int out) {
        Parallel.forRange(B, bi -> {
            int xOff = bi * in;
            int zOff = bi * out;
            for (int o = 0; o < out; o++) {
                double sum = b[o];
                int wOff = o * in;
                for (int i = 0; i < in; i++) sum += W[wOff + i] * X[xOff + i];
                Z[zOff + o] = sum;
            }
        });
    }

    // ------------------------------------------------------------------
    // Backward: dX, dW, db from dZ
    //
    // Parallelized over the batch dimension.
    //
    //   dX  — each row is written by exactly one batch item, so it can be
    //         filled directly with no cross-thread reduction.
    //   dW  — reduced across all batches, so each worker accumulates into
    //   db    its own scratch row and the results are summed at the end.
    // ------------------------------------------------------------------
    @Override
    public void denseBackward(double[] dX, double[] dW, double[] db,
                              double[] dZ, double[] X, double[] W,
                              int B, int in, int out) {
        final int threads = Parallel.numThreads();
        final double[][] localDW = localDW(out * in);
        final double[][] localDB = localDB(out);

        // dX has no cross-batch reduction, so a single zero pass suffices.
        Arrays.fill(dX, 0, B * in, 0.0);

        Parallel.forRange(B, bi -> {
            int tid = (int) (Thread.currentThread().threadId() % threads);
            double[] lwdw = localDW[tid];
            double[] lbdb = localDB[tid];

            int xOff = bi * in;
            int zOff = bi * out;

            for (int o = 0; o < out; o++) {
                double g = dZ[zOff + o];
                lbdb[o] += g;
                int wOff = o * in;
                for (int i = 0; i < in; i++) {
                    lwdw[wOff + i] += g * X[xOff + i];
                    dX[xOff + i]   += g * W[wOff + i];
                }
            }
        });

        // Reduce thread-local weight / bias gradients into the destination
        // arrays. Size here is out * in, up to ~8.6M for the first Dense.
        Arrays.fill(dW, 0, out * in, 0.0);
        Arrays.fill(db, 0, out, 0.0);
        for (int t = 0; t < threads; t++) {
            double[] lwdw = localDW[t];
            double[] lbdb = localDB[t];
            for (int i = 0; i < out * in; i++) dW[i] += lwdw[i];
            for (int o = 0; o < out; o++)      db[o] += lbdb[o];
        }
    }

    @Override public String name() { return "CPU"; }
    @Override public void sync() { }
}