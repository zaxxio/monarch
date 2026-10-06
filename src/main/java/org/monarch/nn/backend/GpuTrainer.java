package org.monarch.nn.backend;

import org.monarch.nn.activation.ActivationFunction;
import org.monarch.nn.config.MultiLayerConfiguration;
import org.monarch.nn.data.MLData;
import org.monarch.nn.layer.*;
import org.monarch.nn.optimizer.OptimizerType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.TornadoExecutionPlan;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

/**
 * Runs the whole training step (forward, softmax-CE loss, backward, Adam)
 * as ONE TornadoVM TaskGraph. Activations, gradients and optimizer state
 * stay on the device. Per step only the input batch / labels / weights go up
 * and the loss + updated parameters come down (about 0.4 MB for this model).
 *
 * Supported: Input, Reshape, Flatten, Conv2D (stride 1, ReLU/identity),
 * MaxPool (square, stride == pool), Dense (ReLU/identity), Output
 * (softmax + CE), Adam, no gradient clipping. Anything else -> tryCreate()
 * returns null and the CPU path is used.
 */
public final class GpuTrainer {

    private static final Logger log = LoggerFactory.getLogger(GpuTrainer.class);

    // NOTE: AdamOptimizer.java wasn't visible; these are the standard defaults.
    private static final float BETA1 = 0.9f, BETA2 = 0.999f, EPS = 1e-8f;

    private static final int CONV = 0, POOL = 1, DENSE = 2, OUT = 3;

    private static final class Rec {
        int kind;
        float[] in, out, w, b, dw, db;
        int[] arg;
        int[] cfg;
        int relu;
        Layer layer;
    }

    private final int B;
    private final int inFeatures;
    private final int nClasses;

    private final float[] xBatch, wgt, lossArr, hyper;
    private final int[] labels;

    private final List<Rec> paramRecs = new ArrayList<>();
    private final TornadoExecutionPlan plan;

    // ------------------------------------------------------------------
    // Factory: returns null (with a log line) if the GPU path can't be used.
    // ------------------------------------------------------------------
    public static GpuTrainer tryCreate(MultiLayerConfiguration cfg,
                                       List<Layer> layers, TensorOps ops) {
        if (!(ops instanceof TornadoTensorOps)) return null;
        final TornadoTensorOps tops = (TornadoTensorOps) ops;
        if (cfg.getOptimizer() != OptimizerType.ADAM) {
            log.warn("GPU trainer supports ADAM only (got {}) - using per-layer ops",
                    cfg.getOptimizer());
            return null;
        }
        if (cfg.getGradientClipNorm() > 0) {
            log.warn("GPU trainer doesn't support gradient clipping - using per-layer ops");
            return null;
        }
        try {
            GpuTrainer t = new GpuTrainer(layers, cfg.getBatchSize(), tops);
            log.info("GPU trainer active on {}: whole training step runs as one TaskGraph (batch={})",
                    tops.name(), cfg.getBatchSize());
            return t;
        } catch (IllegalArgumentException e) {
            log.warn("GPU trainer not used: {}", e.getMessage());
            return null;
        }
    }

    // ------------------------------------------------------------------
    private GpuTrainer(List<Layer> layers, int batch, TornadoTensorOps tops) {
        this.B = batch;
        final int cB = batch;

        final List<Consumer<TaskGraph>> tasks = new ArrayList<>();
        final List<Object> once = new ArrayList<>();
        final List<Rec> recs = new ArrayList<>();

        int[] shape = { layers.get(0).getNIn() };
        this.inFeatures = shape[0];
        this.xBatch = new float[B * inFeatures];
        float[] cur = xBatch;

        // ---------------- forward construction ----------------
        for (int li = 0; li < layers.size(); li++) {
            final int id = li;
            final Layer l = layers.get(li);

            if (l instanceof InputLayer || l instanceof ReshapeLayer
                    || l instanceof FlattenLayer) {
                shape = l.outputShape(shape);          // pure reinterpretation
                continue;
            }

            if (l instanceof Conv2DLayer) {
                Conv2DLayer c = (Conv2DLayer) l;
                if (shape.length != 3) throw new IllegalArgumentException("Conv2D needs [C,H,W]");
                if (c.getStride() != 1)
                    throw new IllegalArgumentException("GPU Conv2D supports stride 1 only");
                int relu = reluFlag(l);
                int inC = shape[0], inH = shape[1], inW = shape[2];
                int[] os = c.outputShape(shape);
                int outC = os[0], outH = os[1], outW = os[2];

                Rec r = new Rec();
                r.kind = CONV; r.layer = l; r.relu = relu; r.in = cur;
                r.cfg = new int[]{ B, inC, inH, inW, outC, outH, outW,
                        c.getKernelH(), c.getKernelW(), c.getStride(), c.getPadding(), relu };
                splitParams(r, l.getParams(), outC * inC * c.getKernelH() * c.getKernelW());
                r.out = new float[B * outC * outH * outW];

                final Rec fr = r;
                final int fOutC = outC, fOutH = outH;
                tasks.add(tg -> tg.task("convF" + id, GpuNetKernels::convForward,
                        fr.in, fr.w, fr.b, fr.out, fr.cfg, cB, fOutC, fOutH));
                recs.add(r); paramRecs.add(r);
                cur = r.out; shape = os;
                continue;
            }

            if (l instanceof MaxPoolingLayer) {
                MaxPoolingLayer p = (MaxPoolingLayer) l;
                if (shape.length != 3) throw new IllegalArgumentException("MaxPool needs [C,H,W]");
                if (p.getPoolH() != p.getPoolW() || p.getStride() != p.getPoolH())
                    throw new IllegalArgumentException("MaxPool must be square with stride == pool");
                int C = shape[0], inH = shape[1], inW = shape[2];
                int[] os = p.outputShape(shape);

                Rec r = new Rec();
                r.kind = POOL; r.layer = l; r.in = cur;
                r.cfg = new int[]{ B, C, inH, inW, os[1], os[2], p.getPoolH(), p.getStride() };
                r.out = new float[B * C * os[1] * os[2]];
                r.arg = new int[r.out.length];

                final Rec fr = r;
                final int bc = B * C, fOutH = os[1], fOutW = os[2];
                tasks.add(tg -> tg.task("poolF" + id, GpuNetKernels::poolForward,
                        fr.in, fr.out, fr.arg, fr.cfg, bc, fOutH, fOutW));
                recs.add(r);
                cur = r.out; shape = os;
                continue;
            }

            if (l instanceof DenseLayer) {            // includes OutputLayer
                boolean isOut = l instanceof OutputLayer;
                if (shape.length != 1) throw new IllegalArgumentException("Dense needs flat input");
                int in = shape[0], out = l.getNOut();
                if (in != l.getNIn())
                    throw new IllegalArgumentException("Dense nIn mismatch: " + in + " vs " + l.getNIn());

                int relu;
                if (isOut) {
                    OutputLayer o = (OutputLayer) l;
                    if (o.getActivationFunction() != ActivationFunction.SOFTMAX
                            || o.getLossType() != LossType.CROSS_ENTROPY)
                        throw new IllegalArgumentException("Output must be SOFTMAX + CROSS_ENTROPY");
                    relu = 0;
                } else {
                    relu = reluFlag(l);
                }

                Rec r = new Rec();
                r.kind = isOut ? OUT : DENSE; r.layer = l; r.relu = relu; r.in = cur;
                r.cfg = new int[]{ B, in, out, relu };
                splitParams(r, l.getParams(), in * out);
                r.out = new float[B * out];

                final Rec fr = r;
                final int fOut = out;
                tasks.add(tg -> tg.task("denseF" + id, GpuNetKernels::denseForward,
                        fr.in, fr.w, fr.b, fr.out, fr.cfg, cB, fOut));
                recs.add(r); paramRecs.add(r);
                cur = r.out; shape = new int[]{ out };
                continue;
            }

            throw new IllegalArgumentException("Unsupported layer: " + l.getClass().getSimpleName());
        }

        if (recs.isEmpty() || recs.get(recs.size() - 1).kind != OUT)
            throw new IllegalArgumentException("Last layer must be an OutputLayer");

        final Rec last = recs.get(recs.size() - 1);
        this.nClasses = last.cfg[2];
        this.labels = new int[B];
        this.wgt = new float[B];
        this.lossArr = new float[B];
        this.hyper = new float[]{ 0f, BETA1, BETA2, EPS };
        final float[] dlogits = new float[B * nClasses];
        final int cC = nClasses;
        final int[] fLabels = labels;
        final float[] fWgt = wgt, fLoss = lossArr;

        tasks.add(tg -> tg.task("softmaxCE", GpuNetKernels::softmaxCE,
                last.out, fLabels, fWgt, dlogits, fLoss, cB, cC));

        // ---------------- backward construction ----------------
        float[] gOut = dlogits;
        for (int r = recs.size() - 1; r >= 0; r--) {
            final int id = r;
            final Rec rc = recs.get(r);
            final float[] g = gOut;

            if (rc.kind == POOL) {
                final float[] dx = new float[rc.in.length];
                final int nIn = dx.length, nOut = rc.out.length;
                tasks.add(tg -> tg.task("poolZ" + id, GpuNetKernels::zero, dx, nIn));
                tasks.add(tg -> tg.task("poolB" + id, GpuNetKernels::poolScatter,
                        g, rc.arg, dx, nOut));
                once.add(dx);
                gOut = dx;
                continue;
            }

            // CONV / DENSE / OUT: get dz
            final float[] dz;
            if (rc.kind == OUT || rc.relu == 0) {
                dz = g;
            } else {
                dz = new float[rc.out.length];
                final int n = dz.length, relu = rc.relu;
                tasks.add(tg -> tg.task("act" + id, GpuNetKernels::actBackward,
                        g, rc.out, dz, relu, n));
                once.add(dz);
            }

            rc.dw = new float[rc.w.length];
            rc.db = new float[rc.b.length];
            once.add(rc.dw); once.add(rc.db);

            final boolean needDx = rc.in != xBatch;   // skip dX for the first layer
            final float[] dx = needDx ? new float[rc.in.length] : null;

            if (rc.kind == CONV) {
                final int cOutC = rc.cfg[4], cInC = rc.cfg[1], cKH = rc.cfg[7], cInH = rc.cfg[2];
                tasks.add(tg -> tg.task("convBW" + id, GpuNetKernels::convBackwardW,
                        dz, rc.in, rc.dw, rc.cfg, cOutC, cInC, cKH));
                tasks.add(tg -> tg.task("convBB" + id, GpuNetKernels::convBackwardB,
                        dz, rc.db, rc.cfg, cOutC));
                if (needDx) {
                    tasks.add(tg -> tg.task("convBX" + id, GpuNetKernels::convBackwardX,
                            dz, rc.w, dx, rc.cfg, cB, cInC, cInH));
                }
            } else {
                final int cOut = rc.cfg[2], cIn = rc.cfg[1];
                tasks.add(tg -> tg.task("denseBW" + id, GpuNetKernels::denseBackwardW,
                        dz, rc.in, rc.dw, rc.cfg, cOut, cIn));
                tasks.add(tg -> tg.task("denseBB" + id, GpuNetKernels::denseBackwardB,
                        dz, rc.db, rc.cfg, cOut));
                if (needDx) {
                    tasks.add(tg -> tg.task("denseBX" + id, GpuNetKernels::denseBackwardX,
                            dz, rc.w, dx, rc.cfg, cB, cIn));
                }
            }

            if (needDx) {
                once.add(dx);
                gOut = dx;
            }
        }

        // ---------------- Adam on device ----------------
        final List<Object> host = new ArrayList<>();
        host.add(lossArr);
        final float[] fHyper = hyper;
        for (int i = 0; i < paramRecs.size(); i++) {
            final int id = i;
            final Rec rc = paramRecs.get(i);
            final float[] mW = new float[rc.w.length], vW = new float[rc.w.length];
            final float[] mB = new float[rc.b.length], vB = new float[rc.b.length];
            final int nW = rc.w.length, nB = rc.b.length;
            tasks.add(tg -> tg.task("adamW" + id, GpuNetKernels::adam,
                    rc.w, rc.dw, mW, vW, fHyper, nW));
            tasks.add(tg -> tg.task("adamB" + id, GpuNetKernels::adam,
                    rc.b, rc.db, mB, vB, fHyper, nB));
            once.add(rc.w); once.add(rc.b);
            once.add(mW); once.add(vW); once.add(mB); once.add(vB);
            host.add(rc.w); host.add(rc.b);
        }

        // Everything else that must exist on the device
        once.add(dlogits);
        for (Rec rc : recs) {
            once.add(rc.out); once.add(rc.cfg);
            if (rc.arg != null) once.add(rc.arg);
        }

        // ---------------- assemble graph ----------------
        TaskGraph g = new TaskGraph("gpuTrain")
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, once.toArray())
                .transferToDevice(DataTransferMode.EVERY_EXECUTION,
                        xBatch, labels, wgt, hyper);
        for (Consumer<TaskGraph> t : tasks) t.accept(g);
        g.transferToHost(DataTransferMode.EVERY_EXECUTION, host.toArray());

        this.plan = tops.newPlan(g.snapshot());   // pinned to the selected device
    }

    // ------------------------------------------------------------------
    /** One full training step on the GPU. Returns the mean CE loss of the batch. */
    public double trainBatch(List<MLData> batch, double lr, int step) {
        int nValid = batch.size();
        float wv = 1f / nValid;

        for (int b = 0; b < B; b++) {
            if (b < nValid) {
                MLData d = batch.get(b);
                double[] in = d.getInputs();
                int off = b * inFeatures;
                for (int i = 0; i < inFeatures; i++) xBatch[off + i] = (float) in[i];
                double[] t = d.getTargets();
                int best = 0;
                for (int k = 1; k < t.length; k++) if (t[k] > t[best]) best = k;
                labels[b] = best;
                wgt[b] = wv;
            } else {                       // padding row of a short last batch
                labels[b] = 0;
                wgt[b] = 0f;
            }
        }

        double c1 = 1.0 - Math.pow(BETA1, step);
        double c2 = 1.0 - Math.pow(BETA2, step);
        hyper[0] = (float) (lr * Math.sqrt(c2) / c1);

        plan.execute();

        double loss = 0;
        for (int b = 0; b < nValid; b++) loss += lossArr[b];
        return loss / nValid;
    }

    /** Copy trained weights back into the layers and put them back on CPU ops. */
    public void finish(List<Layer> layers) {
        for (Rec rc : paramRecs) {
            double[] p = new double[rc.w.length + rc.b.length];
            for (int i = 0; i < rc.w.length; i++) p[i] = rc.w[i];
            for (int i = 0; i < rc.b.length; i++) p[rc.w.length + i] = rc.b[i];
            rc.layer.setParams(p);
        }
        // Inference (B=1) is faster on CPU than a GPU round trip per sample.
        for (Layer l : layers) l.setOps(CpuTensorOps.INSTANCE);
        log.info("GPU trainer: weights copied back to layers, inference uses CPU ops");
    }

    // ------------------------------------------------------------------
    private static int reluFlag(Layer l) {
        ActivationFunction a = l.getActivationFunction();
        if (a == ActivationFunction.RELU) return 1;
        if (a == ActivationFunction.IDENTITY) return 0;
        throw new IllegalArgumentException("Unsupported activation on GPU: " + a);
    }

    private static void splitParams(Rec r, double[] p, int wLen) {
        r.w = new float[wLen];
        r.b = new float[p.length - wLen];
        for (int i = 0; i < wLen; i++) r.w[i] = (float) p[i];
        for (int i = 0; i < r.b.length; i++) r.b[i] = (float) p[wLen + i];
    }
}