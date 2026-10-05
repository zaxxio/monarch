package org.monarch.nn.logging;

import org.monarch.nn.MultiLayerNetwork;

@FunctionalInterface
public interface EpochCallback {
    /**
     * Called after each epoch completes.
     * @param epoch 1-based epoch number (matches UI display)
     * @param net   this network, so the callback can call predict() etc.
     */
    void onEpochEnd(int epoch, MultiLayerNetwork net);
}