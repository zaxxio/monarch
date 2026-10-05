package org.monarch.nn.data;

import org.monarch.nn.data.MLData;
import org.monarch.nn.data.MLDataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.nio.file.*;
import java.util.zip.GZIPInputStream;

/**
 * Loads the standard MNIST IDX files.
 *
 * Expects these files in a directory:
 *   train-images-idx3-ubyte.gz
 *   train-labels-idx1-ubyte.gz
 *   t10k-images-idx3-ubyte.gz
 *   t10k-labels-idx1-ubyte.gz
 *
 * Download: http://yann.lecun.com/exdb/mnist/
 */
public final class MnistLoader {

    private static final Logger log = LoggerFactory.getLogger(MnistLoader.class);

    private MnistLoader() {}

    public static MLDataSet loadTrain(String dir) {
        return load(dir, "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz");
    }

    public static MLDataSet loadTest(String dir) {
        return load(dir, "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz");
    }

    // ------------------------------------------------------------------
    private static MLDataSet load(String dir, String imgFile, String lblFile) {
        Path imgPath = Paths.get(dir, imgFile);
        Path lblPath = Paths.get(dir, lblFile);

        // If files are missing, download them once.
        if (!Files.exists(imgPath) || !Files.exists(lblPath)) {
            log.warn("MNIST files not found under {} — downloading (~11 MB)", dir);
            MnistDownloader.ensure(dir);
        }

        try {
            int[] images = readImages(imgPath.toString());
            int[] labels = readLabels(lblPath.toString());

            int n = labels.length;
            if (images.length != n * 28 * 28)
                throw new IllegalStateException("Image/label count mismatch");

            double[][] X = new double[n][784];
            double[][] Y = new double[n][10];

            for (int i = 0; i < n; i++) {
                // Normalize pixel values [0,255] → [0,1]
                for (int p = 0; p < 784; p++)
                    X[i][p] = images[i * 784 + p] / 255.0;
                Y[i][labels[i]] = 1.0;   // one-hot
            }

            log.info("Loaded {} samples from {}", n, dir);
            return new MLDataSet(X, Y);

        } catch (IOException e) {
            throw new RuntimeException("Failed to load MNIST: " + e.getMessage(), e);
        }
    }

    // ------------------------------------------------------------------
    // IDX parsers
    // ------------------------------------------------------------------
    private static int[] readImages(String path) throws IOException {
        try (DataInputStream in = open(path)) {
            int magic = in.readInt();
            if (magic != 0x00000803)
                throw new IOException("Bad image magic: " + Integer.toHexString(magic));
            int count = in.readInt();
            int rows  = in.readInt();
            int cols  = in.readInt();
            if (rows != 28 || cols != 28)
                throw new IOException("Unexpected image size: " + rows + "x" + cols);

            byte[] buf = new byte[count * rows * cols];
            in.readFully(buf);

            int[] out = new int[buf.length];
            for (int i = 0; i < buf.length; i++) out[i] = buf[i] & 0xFF;
            return out;
        }
    }

    private static int[] readLabels(String path) throws IOException {
        try (DataInputStream in = open(path)) {
            int magic = in.readInt();
            if (magic != 0x00000801)
                throw new IOException("Bad label magic: " + Integer.toHexString(magic));
            int count = in.readInt();

            byte[] buf = new byte[count];
            in.readFully(buf);

            int[] out = new int[count];
            for (int i = 0; i < count; i++) out[i] = buf[i] & 0xFF;
            return out;
        }
    }

    private static DataInputStream open(String path) throws IOException {
        InputStream raw = Files.newInputStream(Paths.get(path));
        InputStream in = path.endsWith(".gz") ? new GZIPInputStream(raw) : raw;
        return new DataInputStream(new BufferedInputStream(in, 1 << 16));
    }
}