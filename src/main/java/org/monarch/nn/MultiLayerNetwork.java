package org.monarch.nn;

import org.monarch.nn.backend.BackendResolver;
import org.monarch.nn.backend.TensorOps;
import org.monarch.nn.config.MultiLayerConfiguration;
import org.monarch.nn.data.MLData;
import org.monarch.nn.data.MLDataSet;
import org.monarch.nn.history.TrainingHistory;
import org.monarch.nn.layer.Layer;
import org.monarch.nn.layer.LossType;
import org.monarch.nn.layer.OutputLayer;
import org.monarch.nn.learningrate.*;
import org.monarch.nn.logging.EpochCallback;
import org.monarch.nn.logging.TrainingLogger;
import org.monarch.nn.optimizer.*;
import org.monarch.nn.tensor.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class MultiLayerNetwork {

    private static final Logger log = LoggerFactory.getLogger(MultiLayerNetwork.class);

    private final MultiLayerConfiguration config;
    private final List<Layer> layers;
    private final List<Optimizer> layerOptimizers = new ArrayList<>();
    private final LRSchedule lrSchedule;
    private final TrainingHistory history = new TrainingHistory();
    private final TensorOps ops;

    private int globalStep = 0;
    private double finalError = Double.MAX_VALUE;

    public MultiLayerNetwork(MultiLayerConfiguration config) {
        this.config = config;
        this.layers = new ArrayList<>(config.getLayers());
        for (int i = 0; i < layers.size(); i++)
            layerOptimizers.add(buildOptimizer(config));
        this.lrSchedule = buildLRSchedule(config);
        this.ops = BackendResolver.resolve(config.getBackend(), this.layers);
    }

    public void init() {
        int[] shape = null;
        for (Layer layer : layers) {
            layer.setOps(this.ops);
            if (shape == null) shape = new int[]{ layer.getNIn() };
            layer.init(shape);
            shape = layer.outputShape(shape);
        }
        log.info("Initialized {} layers, backend = {}", layers.size(), ops.name());
    }

    public Tensor forward(Tensor input) {
        Tensor x = input;
        for (Layer l : layers) x = l.forward(x, true);
        return x;
    }

    public Tensor backward(Tensor gradOutput) {
        Tensor g = gradOutput;
        for (int i = layers.size() - 1; i >= 0; i--) g = layers.get(i).backward(g);
        return g;
    }

    // ==================================================================
    // fit()
    // ==================================================================
    public void fit(MLDataSet dataset) {
        fit(dataset, null);
    }

    public void fit(MLDataSet dataset, EpochCallback callback) {
        List<MLData> data = new ArrayList<>(dataset.getDataList());
        int batchSize = config.getBatchSize();
        int maxEpochs = config.getMaxEpoch();
        Random rng = new Random(config.getSeed());

        int totalParams = countParams();
        TrainingLogger ui = new TrainingLogger(maxEpochs);
        ui.header(describeTopology(), totalParams, batchSize);

        LossType lossType = resolveLossType();
        log.info("Loss function: {}", lossType);

        int totalBatchesPerEpoch = (data.size() + batchSize - 1) / batchSize;
        log.debug("Training start: {} samples, {} batches/epoch, optimizer={}, schedule={}",
                data.size(), totalBatchesPerEpoch,
                config.getOptimizer(), config.getLrSchedule());

        long trainingStart = System.currentTimeMillis();
        double bestLoss = Double.MAX_VALUE;
        int epoch;

        for (epoch = 0; epoch < maxEpochs; epoch++) {
            long epochStart = System.currentTimeMillis();

            if (config.isShuffleEachEpoch()) Collections.shuffle(data, rng);
            double epochLoss = 0;
            int batches = 0;

            for (int i = 0; i < data.size(); i += batchSize) {
                int end = Math.min(i + batchSize, data.size());
                List<MLData> batch = data.subList(i, end);
                int B = batch.size();

                ops.beginBatch();

                Tensor x = stack(batch, true);
                Tensor y = stack(batch, false);
                Tensor pred = forward(x);

                LossResult lr = computeLossAndGradient(pred, y, B, lossType);
                epochLoss += lr.loss;
                batches++;

                Layer last = layers.get(layers.size() - 1);
                boolean wasSkipping = last.isSkipActivationDerivative();
                last.setSkipActivationDerivative(true);

                backward(lr.gradient);

                last.setSkipActivationDerivative(wasSkipping);

                double lrValue = lrSchedule.getLR(epoch, globalStep);
                for (int li = 0; li < layers.size(); li++)
                    layers.get(li).update(layerOptimizers.get(li), lrValue, globalStep, B);
                globalStep++;

                ops.endBatch();

                if (config.isVerbose() && (globalStep % 10 == 0 || batches == totalBatchesPerEpoch)) {
                    long elapsed = System.currentTimeMillis() - epochStart;
                    double running = epochLoss / batches;
                    double avgMs  = (double) elapsed / batches;
                    double etaSec = (totalBatchesPerEpoch - batches) * avgMs / 1000.0;
                    System.out.printf(
                            "    ep %d  batch %4d/%d   loss=%.4f   %d ms   eta %.1fs%n",
                            epoch + 1,
                            batches, totalBatchesPerEpoch,
                            running,
                            elapsed,
                            etaSec);
                }
            }

            double avgLoss = epochLoss / batches;
            finalError = avgLoss;
            history.trainLoss.add(avgLoss);
            history.lrHistory.add(lrSchedule.getLR(epoch, globalStep));
            if (avgLoss < bestLoss) bestLoss = avgLoss;

            long epochMs = System.currentTimeMillis() - epochStart;
            double epochsPerSec = epochMs > 0 ? 1000.0 / epochMs : 0;

            ui.epoch(epoch + 1, maxEpochs, avgLoss, bestLoss,
                    history.lrHistory.get(history.lrHistory.size() - 1),
                    epochMs, epochsPerSec);

            if (callback != null) {
                try {
                    callback.onEpochEnd(epoch + 1, this);
                } catch (Exception e) {
                    log.warn("Epoch callback failed at epoch {}: {}", epoch + 1, e.getMessage());
                }
            }

            if (avgLoss <= config.getMinError()) {
                ui.converged(epoch + 1, bestLoss);
                log.info("Training converged: epoch={}, loss={}, elapsed={}s",
                        epoch + 1,
                        String.format("%.6f", bestLoss),
                        String.format("%.2f",
                                (System.currentTimeMillis() - trainingStart) / 1000.0));
                return;
            }
        }

        ui.finished(epoch, bestLoss);
        log.info("Training finished: epochs={}, loss={}, elapsed={}s",
                epoch,
                String.format("%.6f", bestLoss),
                String.format("%.2f",
                        (System.currentTimeMillis() - trainingStart) / 1000.0));
    }

    // ==================================================================
    // Save / load
    // ==================================================================

    private static final int MAGIC = 0x4D4F4E41;   // "MONA"
    private static final int VERSION = 1;

    public void save(File f) throws IOException {
        try (DataOutputStream out = new DataOutputStream(
                new BufferedOutputStream(new FileOutputStream(f)))) {
            out.writeInt(MAGIC);
            out.writeInt(VERSION);
            out.writeInt(layers.size());
            for (Layer l : layers) {
                double[] p = l.getParams();
                if (p == null) {
                    out.writeInt(-1);
                } else {
                    out.writeInt(p.length);
                    for (double v : p) out.writeDouble(v);
                }
            }
        }
        log.info("Saved model to {}", f.getAbsolutePath());
    }

    public void load(File f) throws IOException {
        try (DataInputStream in = new DataInputStream(
                new BufferedInputStream(new FileInputStream(f)))) {
            int magic = in.readInt();
            if (magic != MAGIC)
                throw new IOException("Not a Monarch model file: " + f);
            int version = in.readInt();
            if (version != VERSION)
                throw new IOException("Unsupported model version: " + version);
            int n = in.readInt();
            if (n != layers.size())
                throw new IOException("Layer count mismatch: file=" + n
                        + " network=" + layers.size());
            for (int i = 0; i < n; i++) {
                int len = in.readInt();
                if (len < 0) continue;
                double[] p = new double[len];
                for (int j = 0; j < len; j++) p[j] = in.readDouble();
                layers.get(i).setParams(p);
            }
        }
        log.info("Loaded model from {}", f.getAbsolutePath());
    }

    // ==================================================================
    // Loss dispatch
    // ==================================================================

    private static final class LossResult {
        final double loss;
        final Tensor gradient;
        LossResult(double loss, Tensor gradient) {
            this.loss = loss;
            this.gradient = gradient;
        }
    }

    private LossType resolveLossType() {
        Layer last = layers.get(layers.size() - 1);
        if (last instanceof OutputLayer)
            return ((OutputLayer) last).getLossType();
        return LossType.MSE;
    }

    private LossResult computeLossAndGradient(Tensor pred, Tensor y,
                                              int B, LossType lossType) {
        Tensor grad = new Tensor(pred.getShape());
        double totalLoss = 0;
        double inv = 1.0 / B;

        int[] predShape = pred.getShape();
        int lastDim = predShape[predShape.length - 1];
        int rows = pred.size() / lastDim;

        switch (lossType) {

            case BINARY_CROSS_ENTROPY: {
                for (int k = 0; k < pred.size(); k++) {
                    double p = pred.getData()[k];
                    double t = y.getData()[k];
                    double pc = Math.min(Math.max(p, 1e-12), 1 - 1e-12);
                    totalLoss += -(t * Math.log(pc) + (1 - t) * Math.log(1 - pc));
                    grad.getData()[k] = (p - t) * inv;
                }
                break;
            }

            case CROSS_ENTROPY: {
                for (int r = 0; r < rows; r++) {
                    int off = r * lastDim;
                    int target = 0;
                    for (int c = 1; c < lastDim; c++)
                        if (y.getData()[off + c] > y.getData()[off + target])
                            target = c;

                    double p = pred.getData()[off + target];
                    totalLoss += -Math.log(Math.max(p, 1e-12));

                    for (int c = 0; c < lastDim; c++) {
                        double pc = pred.getData()[off + c];
                        double oneHot = (c == target) ? 1.0 : 0.0;
                        grad.getData()[off + c] = (pc - oneHot) * inv;
                    }
                }
                break;
            }

            case MSE:
            default: {
                for (int k = 0; k < pred.size(); k++) {
                    double diff = pred.getData()[k] - y.getData()[k];
                    totalLoss += diff * diff;
                    grad.getData()[k] = 2.0 * diff * inv;
                }
                break;
            }
        }

        double meanLoss = (lossType == LossType.CROSS_ENTROPY)
                ? totalLoss / B
                : totalLoss / (B * lastDim);

        return new LossResult(meanLoss, grad);
    }

    // ==================================================================
    // Helpers
    // ==================================================================

    private int countParams() {
        int n = 0;
        for (Layer l : layers) {
            double[] p = l.getParams();
            if (p != null) n += p.length;
        }
        return n;
    }

    private String describeTopology() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < layers.size(); i++) {
            if (i > 0) sb.append(" → ");
            sb.append(layers.get(i).getClass().getSimpleName().replace("Layer", ""));
        }
        return sb.toString();
    }

    private Tensor stack(List<MLData> batch, boolean useInputs) {
        int B = batch.size();
        int F = useInputs
                ? batch.get(0).getInputs().length
                : batch.get(0).getTargets().length;
        Tensor t = new Tensor(B, F);
        double[] d = t.getData();
        for (int b = 0; b < B; b++) {
            double[] src = useInputs
                    ? batch.get(b).getInputs()
                    : batch.get(b).getTargets();
            System.arraycopy(src, 0, d, b * F, F);
        }
        return t;
    }

    public double[] predict(double... inputs) {
        Tensor x = new Tensor(1, inputs.length);
        System.arraycopy(inputs, 0, x.getData(), 0, inputs.length);
        return forward(x).getData().clone();
    }

    public double[] predict(MLData data) {
        return predict(data.getInputs());
    }

    public int predictClass(double... inputs) {
        double[] p = predict(inputs);
        int best = 0;
        for (int i = 1; i < p.length; i++) if (p[i] > p[best]) best = i;
        return best;
    }

    private static Optimizer buildOptimizer(MultiLayerConfiguration c) {
        switch (c.getOptimizer()) {
            case SGD:       return new SGDOptimizer();
            case MOMENTUM:  return new MomentumOptimizer(c.getMomentum());
            case ADAM:      return new AdamOptimizer();
            case ADAMW:     return new AdamWOptimizer();
            case NADAM:     return new NadamOptimizer();
            case RMSPROP:   return new RMSPropOptimizer();
            case ADAGRAD:   return new AdaGradOptimizer();
            case ADADELTA:  return new AdaDeltaOptimizer();
            default:
                throw new IllegalStateException(
                        "Unhandled optimizer: " + c.getOptimizer());
        }
    }

    private static LRSchedule buildLRSchedule(MultiLayerConfiguration c) {
        double base = c.getBaseLR();
        switch (c.getLrSchedule()) {
            case CONSTANT:    return new ConstantLR(base);
            case STEP:        return new StepDecayLR(base, c.getLrStepSize(), c.getLrGamma());
            case EXPONENTIAL: return new ExponentialDecayLR(base, c.getLrDecay());
            case COSINE:      return new CosineAnnealingLR(base, c.getLrMin(), c.getMaxEpoch());
            default:
                throw new IllegalStateException(
                        "Unhandled lrSchedule: " + c.getLrSchedule());
        }
    }

    // ---- Accessors ----

    public TrainingHistory getHistory() { return history; }
    public double getFinalError()       { return finalError; }
    public TensorOps getOps()           { return ops; }

    /** Number of layers, including InputLayer and OutputLayer. */
    public int layerCount() { return layers.size(); }

    /** Direct access to the layer at the given 0-based index. */
    public Layer layer(int i) { return layers.get(i); }
}