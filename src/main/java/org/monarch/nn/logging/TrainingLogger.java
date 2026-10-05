package org.monarch.nn.logging;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Formats training progress with colors and alignment.
 * Prints directly to stdout (not through Logback) so the layout
 * stays clean and column-aligned.
 */
public final class TrainingLogger {

    private static final Logger log = LoggerFactory.getLogger("training");

    // Column widths
    private static final int EPOCH_W = 5;
    private static final int LOSS_W  = 10;
    private static final int LR_W    = 10;
    private static final int MS_W    = 6;
    private static final int EPS_W   = 9;

    private final int totalEpochs;
    private final long startTime;
    private int lastPrintedPercent = -10;

    public TrainingLogger(int totalEpochs) {
        this.totalEpochs = totalEpochs;
        this.startTime = System.currentTimeMillis();
    }

    /** Called once at the start of training. */
    public void header(String model, int params, int batchSize) {
        System.out.println();
        System.out.println(Ansi.BOLD + Ansi.CYAN
                + "  ███  Training " + model + Ansi.RESET);
        System.out.println(Ansi.DIM
                + "       params = " + String.format("%,d", params)
                + "  |  batch = " + batchSize
                + "  |  epochs = " + totalEpochs
                + Ansi.RESET);
        System.out.println();
        System.out.printf(Ansi.BOLD
                + "  %-" + EPOCH_W + "s  %-" + LOSS_W + "s  %-" + LR_W + "s  %-"
                + MS_W + "s  %-" + EPS_W + "s  %s%n" + Ansi.RESET,
                "EPOCH", "LOSS", "LR", "MS", "EP/S", "PROGRESS");
        System.out.println(Ansi.DIM
                + "  " + "─".repeat(EPOCH_W) + "  " + "─".repeat(LOSS_W) + "  "
                + "─".repeat(LR_W) + "  " + "─".repeat(MS_W) + "  "
                + "─".repeat(EPS_W) + "  " + "─".repeat(30)
                + Ansi.RESET);
    }

    /** Called once per epoch. */
    public void epoch(int epoch, int totalEpochs, double loss, double bestLoss,
                      double lr, long epochMs, double epochsPerSec) {

        int percent = (int) (100.0 * epoch / totalEpochs);

        // Loss color: green when improving, yellow when stalled, red when worse
        String lossColor;
        if (loss <= bestLoss * 0.999)      lossColor = Ansi.GREEN;
        else if (loss <= bestLoss * 1.001) lossColor = Ansi.YELLOW;
        else                                lossColor = Ansi.RED;

        String progressBar = progressBar(percent);

        System.out.printf("  %-" + EPOCH_W + "d  %s%-" + LOSS_W + ".6f%s  %-"
                        + LR_W + ".5g  %-" + MS_W + "d  %-" + EPS_W + ".1f  %s%n",
                epoch,
                lossColor, loss, Ansi.RESET,
                lr,
                epochMs,
                epochsPerSec,
                progressBar);

        lastPrintedPercent = percent;
    }

    /** Called when training converges. */
    public void converged(int epoch, double bestLoss) {
        long totalMs = System.currentTimeMillis() - startTime;
        System.out.println();
        System.out.println(Ansi.GREEN + Ansi.BOLD
                + "  ✓ Converged at epoch " + epoch
                + Ansi.RESET + Ansi.DIM
                + "  ("
                + String.format("%.2f s", totalMs / 1000.0)
                + ", best loss = "
                + String.format("%.6f", bestLoss)
                + ")"
                + Ansi.RESET);
    }

    /** Called when training finishes without converging. */
    public void finished(int epochs, double bestLoss) {
        long totalMs = System.currentTimeMillis() - startTime;
        System.out.println();
        System.out.println(Ansi.YELLOW + Ansi.BOLD
                + "  ⏹ Training stopped at epoch " + epochs
                + Ansi.RESET + Ansi.DIM
                + "  ("
                + String.format("%.2f s", totalMs / 1000.0)
                + ", best loss = "
                + String.format("%.6f", bestLoss)
                + ")"
                + Ansi.RESET);
    }

    private String progressBar(int percent) {
        int width = 25;
        int filled = (int) Math.round(width * percent / 100.0);
        StringBuilder sb = new StringBuilder();

        sb.append(Ansi.DIM).append("[");
        sb.append(Ansi.GREEN);
        for (int i = 0; i < filled; i++) sb.append('█');
        sb.append(Ansi.DIM);
        for (int i = filled; i < width; i++) sb.append('░');
        sb.append("] ");
        sb.append(String.format("%3d%%", percent));
        sb.append(Ansi.RESET);
        return sb.toString();
    }
}