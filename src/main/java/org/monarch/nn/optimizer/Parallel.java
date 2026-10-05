package org.monarch.nn.optimizer;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.IntConsumer;

/**
 * Simple parallel loop over an integer range using a chunked
 * work-stealing queue. Small loops run inline to avoid overhead.
 *
 * Threads are daemonized and joined at the end of each call —
 * no thread pool to manage or shut down.
 */
public final class Parallel {

    private static final int NUM_THREADS =
            Math.max(1, Runtime.getRuntime().availableProcessors());

    private Parallel() {}

    /**
     * Runs {@code body.accept(i)} for each i in [0, n), spread across
     * all available cores. Blocks until all iterations finish.
     *
     * If the runtime is single-threaded or the range is small, this
     * falls back to a plain sequential loop.
     */
    public static void forRange(int n, IntConsumer body) {
        if (n <= 0) return;

        if (n < 32 || NUM_THREADS == 1) {
            for (int i = 0; i < n; i++) body.accept(i);
            return;
        }

        // Chunked work-stealing: bigger chunks reduce atomic overhead,
        // smaller chunks reduce tail latency. 4× oversubscription is a
        // good default for coarse-grained ML work.
        final int chunk = Math.max(1, n / (NUM_THREADS * 4));
        final AtomicInteger next = new AtomicInteger(0);

        Thread[] workers = new Thread[NUM_THREADS];
        for (int t = 0; t < NUM_THREADS; t++) {
            workers[t] = new Thread(() -> {
                while (true) {
                    int start = next.getAndAdd(chunk);
                    if (start >= n) return;
                    int end = Math.min(n, start + chunk);
                    for (int i = start; i < end; i++) body.accept(i);
                }
            });
            workers[t].setDaemon(true);
            workers[t].start();
        }

        for (Thread w : workers) {
            try {
                w.join();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return;
            }
        }
    }

    /** Number of worker threads this helper will spawn. */
    public static int numThreads() { return NUM_THREADS; }
}