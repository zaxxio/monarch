package org.monarch.nn.demo;

import org.monarch.nn.MultiLayerNetwork;
import org.monarch.nn.activation.ActivationFunction;
import org.monarch.nn.backend.Backend;
import org.monarch.nn.config.MultiLayerConfiguration;
import org.monarch.nn.data.MLDataSet;
import org.monarch.nn.layer.DenseLayer;
import org.monarch.nn.layer.InputLayer;
import org.monarch.nn.layer.LossType;
import org.monarch.nn.layer.OutputLayer;
import org.monarch.nn.learningrate.LRScheduleType;
import org.monarch.nn.optimizer.OptimizerType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Arrays;

/**
 * Solves XOR.
 * <p>
 * Truth table:
 * 1,1 -> 0
 * 1,0 -> 1
 * 0,1 -> 1
 * 0,0 -> 0
 * <p>
 * A two-layer MLP with a nonlinearity is required — XOR is not
 * linearly separable, so a single dense layer cannot learn it.
 */
public class Driver {

    private static final Logger log = LoggerFactory.getLogger(Driver.class);

    // ---- Dataset ----
    private static final double[][] XOR_INPUT = {
            {1, 1},
            {1, 0},
            {0, 1},
            {0, 0}
    };

    private static final double[][] XOR_TARGET = {
            {0},
            {1},
            {1},
            {0}
    };

    public static void main(String[] args) {

        runXor();
    }

    private static void runXor() {
        final MultiLayerConfiguration cfg = MultiLayerConfiguration.Builder()
                .backendRuntime(Backend.METAL)
                .updater(0.01)
                .optimizationAlgorithm(OptimizerType.ADAM)
                .lrSchedule(LRScheduleType.EXPONENTIAL)
                .batchSize(4)
                .maxEpoch(10000)
                .minError(0.0001)
                .seed(42)
                .verbose(false)
                .shuffleEachEpoch(false)
                .input(new InputLayer.Builder()
                        .nIn(2)
                        .build())
                .dense(new DenseLayer.Builder()
                        .nIn(2).nOut(8)
                        .activation(ActivationFunction.LEAKY_RELU)
                        .build())

                .dense(new DenseLayer.Builder()
                        .nIn(8).nOut(8)
                        .activation(ActivationFunction.LEAKY_RELU)
                        .build())

                .output(new OutputLayer.Builder()
                        .nIn(8).nOut(1)
                        .activation(ActivationFunction.SIGMOID)
                        .lossType(LossType.MSE)
                        .build())

                .build();

        log.debug("Configuration:\n{}", cfg);

        MultiLayerNetwork net = new MultiLayerNetwork(cfg);
        net.init();
        net.fit(new MLDataSet(XOR_INPUT, XOR_TARGET));

        log.info("Predictions:");
        int correct = 0;
        for (int i = 0; i < XOR_INPUT.length; i++) {
            double[] pred = net.predict(XOR_INPUT[i]);
            int rounded = pred[0] >= 0.5 ? 1 : 0;
            int truth = (int) XOR_TARGET[i][0];
            boolean ok = rounded == truth;
            if (ok) correct++;
            log.info("  {} -> {:.4f}  (rounded={}, expected={})  {}",
                    Arrays.toString(XOR_INPUT[i]),
                    pred[0], rounded, truth, ok ? "✓" : "✗");
        }
        log.info("Accuracy: {}/{}", correct, XOR_INPUT.length);
        if (correct == XOR_INPUT.length) {
            log.info("XOR solved.");
        } else {
            log.warn("XOR not fully solved — {} of {} correct.", correct, XOR_INPUT.length);
        }
    }
}