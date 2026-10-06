package org.monarch.nn.demo;

import org.monarch.nn.MultiLayerNetwork;
import org.monarch.nn.activation.ActivationFunction;
import org.monarch.nn.backend.Backend;
import org.monarch.nn.config.MultiLayerConfiguration;
import org.monarch.nn.data.MLData;
import org.monarch.nn.data.MLDataSet;
import org.monarch.nn.data.MnistLoader;
import org.monarch.nn.layer.*;
import org.monarch.nn.learningrate.LRScheduleType;
import org.monarch.nn.optimizer.OptimizerType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class MnistDriver {

    private static final Logger log = LoggerFactory.getLogger(MnistDriver.class);

    private static final String DATA_DIR =
            System.getProperty("mnist.dir", "./data/mnist");

    // ---- Smoke-test knobs ----
    private static final int TRAIN_SAMPLES = 5000;
    private static final int TEST_SAMPLES = 1000;

    public static void main(String[] args) {
        if (Thread.currentThread().getContextClassLoader() == null) {
            Thread.currentThread().setContextClassLoader(
                    MnistDriver.class.getClassLoader());
        }

        log.info("=== MNIST CNN (5K smoke test) ===");
        log.info("Data directory: {}", DATA_DIR);

        // ---- Load + subsample ----
        MLDataSet fullTrain = MnistLoader.loadTrain(DATA_DIR);
        MLDataSet fullTest = MnistLoader.loadTest(DATA_DIR);

        MLDataSet train = subsample(fullTrain, TRAIN_SAMPLES, 42);
        MLDataSet test = subsample(fullTest, TEST_SAMPLES, 42);

        log.info("Train: {} samples, Test: {} samples",
                train.getDataList().size(), test.getDataList().size());

        // ---- Network ----
        MultiLayerConfiguration cfg = new MultiLayerConfiguration.Builder()
                .backendRuntime(Backend.CUDA)
                .updater(0.001)
                .optimizer(OptimizerType.ADAM)
                .lrSchedule(LRScheduleType.COSINE)
                .batchSize(256)
                .maxEpoch(10)
                .minError(0.005)
                .seed(42)
                .verbose(false)
                .shuffleEachEpoch(true)
                .layer(0, new InputLayer.Builder().nIn(784).build())

                .layer(1, new ReshapeLayer.Builder().shape(1, 28, 28).build())

                .layer(2, new Conv2DLayer.Builder()
                        .inChannels(1).outChannels(8)
                        .kernelSize(3, 3).stride(1).padding(1)
                        .activation(ActivationFunction.RELU)
                        .build())

                .layer(3, new MaxPoolingLayer.Builder()
                        .poolSize(2, 2).stride(2).build())

                .layer(4, new Conv2DLayer.Builder()
                        .inChannels(8).outChannels(16)
                        .kernelSize(3, 3).stride(1).padding(1)
                        .activation(ActivationFunction.RELU)
                        .build())

                .layer(5, new MaxPoolingLayer.Builder()
                        .poolSize(2, 2).stride(2).build())

                .layer(6, new FlattenLayer.Builder().build())

                .layer(7, new DenseLayer.Builder()
                        .nIn(16 * 7 * 7).nOut(128)
                        .activation(ActivationFunction.RELU)
                        .build())

                .layer(8, new OutputLayer.Builder()
                        .nIn(128).nOut(10)
                        .activation(ActivationFunction.SOFTMAX)
                        .lossType(LossType.CROSS_ENTROPY)
                        .build())

                .build();

        log.debug("Configuration:\n{}", cfg);

        // ---- Train ----
        MultiLayerNetwork net = new MultiLayerNetwork(cfg);
        net.init();
        net.fit(train);

        // ---- Evaluate ----
        log.info("Evaluating on test set...");
        int correct = 0;
        int total = test.getDataList().size();
        int[] perClassCorrect = new int[10];
        int[] perClassTotal = new int[10];

        long evalStart = System.currentTimeMillis();
        for (MLData d : test.getDataList()) {
            int pred = net.predictClass(d.getInputs());
            int truth = argmax(d.getTargets());
            perClassTotal[truth]++;
            if (pred == truth) {
                correct++;
                perClassCorrect[truth]++;
            }
        }
        long evalMs = System.currentTimeMillis() - evalStart;

        double acc = 100.0 * correct / total;
        log.info("Test accuracy: {}/{} = {}%  ({} ms, {} samples/sec)",
                correct, total,
                String.format("%.2f", acc),
                evalMs,
                String.format("%.0f", total * 1000.0 / Math.max(1, evalMs)));

        log.info("Per-class accuracy:");
        for (int c = 0; c < 10; c++) {
            double clsAcc = 100.0 * perClassCorrect[c] / Math.max(1, perClassTotal[c]);
            log.info("  digit {}: {}/{} = {}%",
                    c, perClassCorrect[c], perClassTotal[c],
                    String.format("%.1f", clsAcc));
        }

        // ---- Sanity ----
        log.info("Sample predictions:");
        for (int i = 0; i < 5; i++) {
            MLData d = test.getDataList().get(i);
            double[] p = net.predict(d.getInputs());
            int pred = argmax(p);
            int truth = argmax(d.getTargets());
            log.info("  sample {}: pred={} truth={} conf={} {}",
                    i, pred, truth,
                    String.format("%.3f", p[pred]),
                    pred == truth ? "✓" : "✗");
        }
    }

    // ==================================================================
    // Deterministic class-balanced subsample
    // ==================================================================

    /**
     * Picks {@code n} samples from {@code full}, balanced across classes.
     * If {@code full} has label counts [6000, 6000, 6000, ...], and n=5000
     * across 10 classes, picks ~500 per class.
     * <p>
     * Uses a fixed seed for reproducible subsets across runs.
     */
    private static MLDataSet subsample(MLDataSet full, int n, long seed) {
        List<MLData> list = full.getDataList();
        int classes = list.get(0).getTargets().length;

        // Bucket by class
        List<List<MLData>> buckets = new ArrayList<>();
        for (int c = 0; c < classes; c++) buckets.add(new ArrayList<>());
        for (MLData d : list) {
            int label = argmax(d.getTargets());
            buckets.get(label).add(d);
        }

        // Shuffle each bucket deterministically
        Random rng = new Random(seed);
        for (List<MLData> bucket : buckets) Collections.shuffle(bucket, rng);

        // Round-robin draw until we have n samples (or run out)
        int perClass = n / classes;
        List<MLData> out = new ArrayList<>(n);

        // First pass: perClass from each
        for (List<MLData> bucket : buckets) {
            int take = Math.min(perClass, bucket.size());
            for (int i = 0; i < take; i++) out.add(bucket.get(i));
        }

        // Second pass: top up from the largest buckets if we're short
        int idx = 0;
        while (out.size() < n) {
            List<MLData> bucket = buckets.get(idx % classes);
            int taken = Math.min(perClass, bucket.size());
            if (taken < bucket.size()) out.add(bucket.get(taken + (idx / classes)));
            idx++;
            if (idx > n * classes) break;   // safety
        }

        // Shuffle the final list so epochs see mixed classes
        Collections.shuffle(out, rng);

        double[][] X = new double[out.size()][];
        double[][] Y = new double[out.size()][];
        for (int i = 0; i < out.size(); i++) {
            X[i] = out.get(i).getInputs();
            Y[i] = out.get(i).getTargets();
        }
        return new MLDataSet(X, Y);
    }

    private static int argmax(double[] a) {
        int best = 0;
        for (int i = 1; i < a.length; i++) if (a[i] > a[best]) best = i;
        return best;
    }
}