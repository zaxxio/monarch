package org.monarch.nn.history;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class TrainingHistory implements Serializable {
    public final List<Double> trainLoss = new ArrayList<>();
    public final List<Double> lrHistory = new ArrayList<>();

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("epoch |    loss   |    lr\n");
        for (int i = 0; i < trainLoss.size(); i++) {
            sb.append(String.format("%5d | %.6f | %.4g%n",
                    i + 1, trainLoss.get(i), lrHistory.get(i)));
        }
        return sb.toString();
    }
}