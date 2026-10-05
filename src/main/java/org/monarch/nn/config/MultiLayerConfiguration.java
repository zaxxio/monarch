package org.monarch.nn.config;

import lombok.Getter;
import lombok.Setter;
import org.monarch.nn.backend.Backend;
import org.monarch.nn.layer.DenseLayer;
import org.monarch.nn.layer.InputLayer;
import org.monarch.nn.layer.Layer;
import org.monarch.nn.layer.OutputLayer;
import org.monarch.nn.learningrate.LRScheduleType;
import org.monarch.nn.optimizer.OptimizerType;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

@Getter
@Setter
public final class MultiLayerConfiguration implements Serializable {

    private static final long serialVersionUID = 1L;

    private final List<Layer> layers;
    private final double baseLR;
    private final double momentum;

    private final OptimizerType optimizer;      // ← enum
    private final LRScheduleType lrSchedule;    // ← enum

    private final int lrStepSize;
    private final double lrGamma;
    private final double lrDecay;
    private final double lrMin;
    private final Backend backend;

    private final int batchSize;
    private final int maxEpoch;
    private final double minError;
    private final double validationSplit;
    private final int earlyStoppingPatience;
    private final double earlyStoppingDelta;
    private final double gradientClipNorm;
    private final boolean shuffleEachEpoch;

    private final long seed;
    private final boolean realtime;
    private final int logEveryNEpochs;
    @Getter
    private boolean verbose = false;

    private MultiLayerConfiguration(Builder b) {
        this.layers = Collections.unmodifiableList(new ArrayList<>(b.layers));
        this.baseLR = b.baseLR;
        this.momentum = b.momentum;
        this.optimizer = b.optimizer;
        this.lrSchedule = b.lrSchedule;
        this.lrStepSize = b.lrStepSize;
        this.lrGamma = b.lrGamma;
        this.lrDecay = b.lrDecay;
        this.lrMin = b.lrMin;
        this.batchSize = b.batchSize;
        this.maxEpoch = b.maxEpoch;
        this.minError = b.minError;
        this.validationSplit = b.validationSplit;
        this.earlyStoppingPatience = b.earlyStoppingPatience;
        this.earlyStoppingDelta = b.earlyStoppingDelta;
        this.gradientClipNorm = b.gradientClipNorm;
        this.shuffleEachEpoch = b.shuffleEachEpoch;
        this.seed = b.seed;
        this.verbose = b.verbose;
        this.realtime = b.realtime;
        this.logEveryNEpochs = b.logEveryNEpochs;
        this.backend = b.backend;
        validate();
    }

    private void validate() {
        if (layers.isEmpty())
            throw new IllegalStateException("Configuration has no layers");
        for (int i = 0; i < layers.size(); i++)
            if (layers.get(i) == null)
                throw new IllegalStateException("Layer " + i + " is null");
        if (layers.get(0).getNIn() <= 0)
            throw new IllegalStateException("Input layer must declare nIn > 0");
        if (batchSize <= 0) throw new IllegalStateException("batchSize must be > 0");
        if (maxEpoch <= 0) throw new IllegalStateException("maxEpoch must be > 0");
        if (baseLR <= 0) throw new IllegalStateException("baseLR must be > 0");
        if (validationSplit < 0 || validationSplit >= 1)
            throw new IllegalStateException("validationSplit must be in [0, 1)");
        if (gradientClipNorm < 0)
            throw new IllegalStateException("gradientClipNorm must be >= 0");
        if (earlyStoppingPatience < 0)
            throw new IllegalStateException("earlyStoppingPatience must be >= 0");
    }

    public static class Builder {
        private final List<Layer> layers = new ArrayList<>();
        public boolean verbose;
        private double baseLR = 0.01;
        private double momentum = 0.9;
        private OptimizerType optimizer = OptimizerType.SGD;         // enum
        private LRScheduleType lrSchedule = LRScheduleType.CONSTANT; // enum
        private int lrStepSize = 10;
        private double lrGamma = 0.5;
        private double lrDecay = 0.01;
        private double lrMin = 0.0;
        private Backend backend = Backend.CPU;

        private int batchSize = 1;
        private int maxEpoch = 100;
        private double minError = 1e-3;
        private double validationSplit = 0.0;
        private int earlyStoppingPatience = 0;
        private double earlyStoppingDelta = 1e-4;
        private double gradientClipNorm = 0.0;
        private boolean shuffleEachEpoch = true;

        private long seed = 42;
        private boolean realtime = false;
        private int logEveryNEpochs = 1;

        // ---- Layers ----
        public Builder layer(int idx, Layer l) {
            while (layers.size() <= idx) layers.add(null);
            layers.set(idx, Objects.requireNonNull(l, "layer"));
            return this;
        }

        public Builder addLayer(Layer l) {
            layers.add(Objects.requireNonNull(l, "layer"));
            return this;
        }

        public Builder list() {
            return this;
        }

        // ---- Optimizer — enum + string overload ----
        public Builder optimizer(OptimizerType o) {
            this.optimizer = o;
            return this;
        }

        public Builder optimizer(String o) {
            this.optimizer = OptimizerType.fromString(o);
            return this;
        }

        // ---- LR Schedule — enum + string overload ----
        public Builder lrSchedule(LRScheduleType s) {
            this.lrSchedule = s;
            return this;
        }

        public Builder lrSchedule(String s) {
            this.lrSchedule = LRScheduleType.fromString(s);
            return this;
        }

        // ---- Other ----
        public Builder updater(double lr) {
            this.baseLR = lr;
            return this;
        }

        public Builder baseLR(double lr) {
            this.baseLR = lr;
            return this;
        }

        public Builder momentum(double m) {
            this.momentum = m;
            return this;
        }

        public Builder lrStepSize(int s) {
            this.lrStepSize = s;
            return this;
        }

        public Builder lrGamma(double g) {
            this.lrGamma = g;
            return this;
        }

        public Builder lrDecay(double d) {
            this.lrDecay = d;
            return this;
        }

        public Builder lrMin(double m) {
            this.lrMin = m;
            return this;
        }

        public Builder backendRuntime(Backend b) {
            this.backend = b;
            return this;
        }

        public Builder batchSize(int b) {
            this.batchSize = b;
            return this;
        }

        public Builder maxEpoch(int e) {
            this.maxEpoch = e;
            return this;
        }

        public Builder minError(double e) {
            this.minError = e;
            return this;
        }

        public Builder validationSplit(double v) {
            this.validationSplit = v;
            return this;
        }

        public Builder earlyStoppingPatience(int p) {
            this.earlyStoppingPatience = p;
            return this;
        }

        public Builder earlyStoppingDelta(double d) {
            this.earlyStoppingDelta = d;
            return this;
        }

        public Builder gradientClipNorm(double n) {
            this.gradientClipNorm = n;
            return this;
        }

        public Builder shuffleEachEpoch(boolean s) {
            this.shuffleEachEpoch = s;
            return this;
        }

        public Builder seed(long s) {
            this.seed = s;
            return this;
        }

        public Builder realtime(boolean r) {
            this.realtime = r;
            return this;
        }

        public Builder logEveryNEpochs(int n) {
            this.logEveryNEpochs = n;
            return this;
        }

        public MultiLayerConfiguration build() {
            return new MultiLayerConfiguration(this);
        }

        public Builder verbose(boolean b) {
            this.verbose = b;
            return this;
        }
    }

    // ==================================================================
    // Step Builder (from previous step)
    // ==================================================================
    public static StartStep Builder() {
        return new StepBuilder();
    }

    public interface StartStep {
        StartStep backendRuntime(Backend b);

        StartStep optimizationAlgorithm(OptimizerType o);

        StartStep updater(double lr);

        StartStep baseLR(double lr);

        StartStep momentum(double m);

        StartStep lrSchedule(LRScheduleType s);

        StartStep lrSchedule(String s);       // string overload

        StartStep lrStepSize(int s);

        StartStep lrGamma(double g);

        StartStep lrDecay(double d);

        StartStep lrMin(double m);

        StartStep batchSize(int b);

        StartStep maxEpoch(int e);

        StartStep minError(double e);

        StartStep validationSplit(double v);

        StartStep earlyStoppingPatience(int p);

        StartStep earlyStoppingDelta(double d);

        StartStep gradientClipNorm(double n);

        StartStep shuffleEachEpoch(boolean s);

        StartStep seed(long s);

        StartStep verbose(boolean verbose);

        StartStep realtime(boolean r);

        StartStep logEveryNEpochs(int n);

        DenseStep input(InputLayer layer);

    }

    public interface DenseStep {
        DenseStep dense(DenseLayer layer);

        FinalStep output(OutputLayer layer);
    }

    public interface FinalStep {
        MultiLayerConfiguration build();
    }

    private static final class StepBuilder implements StartStep, DenseStep, FinalStep {

        private final Builder b = new Builder();
        private int nextIdx = 0;

        @Override
        public StartStep backendRuntime(Backend x) {
            b.backendRuntime(x);
            return this;
        }

        @Override
        public StartStep optimizationAlgorithm(OptimizerType x) {
            b.optimizer(x);
            return this;
        }

        @Override
        public StartStep updater(double x) {
            b.updater(x);
            return this;
        }

        @Override
        public StartStep baseLR(double x) {
            b.baseLR(x);
            return this;
        }

        @Override
        public StartStep momentum(double x) {
            b.momentum(x);
            return this;
        }

        @Override
        public StartStep lrSchedule(LRScheduleType x) {
            b.lrSchedule(x);
            return this;
        }

        @Override
        public StartStep lrSchedule(String x) {
            b.lrSchedule(x);
            return this;
        }

        @Override
        public StartStep lrStepSize(int x) {
            b.lrStepSize(x);
            return this;
        }

        @Override
        public StartStep lrGamma(double x) {
            b.lrGamma(x);
            return this;
        }

        @Override
        public StartStep lrDecay(double x) {
            b.lrDecay(x);
            return this;
        }

        @Override
        public StartStep lrMin(double x) {
            b.lrMin(x);
            return this;
        }

        @Override
        public StartStep batchSize(int x) {
            b.batchSize(x);
            return this;
        }

        @Override
        public StartStep maxEpoch(int x) {
            b.maxEpoch(x);
            return this;
        }

        @Override
        public StartStep minError(double x) {
            b.minError(x);
            return this;
        }

        @Override
        public StartStep validationSplit(double x) {
            b.validationSplit(x);
            return this;
        }

        @Override
        public StartStep earlyStoppingPatience(int x) {
            b.earlyStoppingPatience(x);
            return this;
        }

        @Override
        public StartStep earlyStoppingDelta(double x) {
            b.earlyStoppingDelta(x);
            return this;
        }

        @Override
        public StartStep gradientClipNorm(double x) {
            b.gradientClipNorm(x);
            return this;
        }

        @Override
        public StartStep shuffleEachEpoch(boolean x) {
            b.shuffleEachEpoch(x);
            return this;
        }

        @Override
        public StartStep seed(long x) {
            b.seed(x);
            return this;
        }

        @Override
        public StartStep verbose(boolean verbose) {
            b.verbose = verbose;
            return this;
        }

        @Override
        public StartStep realtime(boolean x) {
            b.realtime(x);
            return this;
        }

        @Override
        public StartStep logEveryNEpochs(int x) {
            b.logEveryNEpochs(x);
            return this;
        }

        @Override
        public DenseStep input(InputLayer l) {
            b.layer(nextIdx++, l);
            return this;
        }

        @Override
        public DenseStep dense(DenseLayer l) {
            b.layer(nextIdx++, l);
            return this;
        }

        @Override
        public FinalStep output(OutputLayer l) {
            b.layer(nextIdx++, l);
            return this;
        }

        @Override
        public MultiLayerConfiguration build() {
            return b.build();
        }
    }

    // ==================================================================
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("MultiLayerConfiguration {\n");
        sb.append("  layers = [\n");
        for (int i = 0; i < layers.size(); i++)
            sb.append("    ").append(i).append(": ")
                    .append(layers.get(i).getClass().getSimpleName())
                    .append("(nIn=").append(layers.get(i).getNIn())
                    .append(", nOut=").append(layers.get(i).getNOut())
                    .append(")\n");
        sb.append("  ],\n");
        sb.append("  backend=").append(backend)
                .append(", optimizer=").append(optimizer)
                .append(", lrSchedule=").append(lrSchedule)
                .append(", lr=").append(baseLR)
                .append("\n}");
        return sb.toString();
    }
}