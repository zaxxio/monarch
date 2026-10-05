package org.monarch.nn.data;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedInputStream;
import java.io.InputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;

/**
 * Downloads the four MNIST IDX files if they're not already present.
 * Files are cached under the given directory.
 */
public final class MnistDownloader {

    private static final Logger log = LoggerFactory.getLogger(MnistDownloader.class);

    private static final String BASE = "https://storage.googleapis.com/cvdf-datasets/mnist";

    private static final String[] FILES = {
            "train-images-idx3-ubyte.gz",
            "train-labels-idx1-ubyte.gz",
            "t10k-images-idx3-ubyte.gz",
            "t10k-labels-idx1-ubyte.gz"
    };

    private MnistDownloader() {}

    /**
     * Ensure all four MNIST files are present in {@code dir}.
     * Downloads any that are missing. Idempotent.
     */
    public static void ensure(String dir) {
        try {
            Path root = Paths.get(dir);
            Files.createDirectories(root);

            HttpClient client = HttpClient.newBuilder()
                    .followRedirects(HttpClient.Redirect.NORMAL)
                    .build();

            for (String name : FILES) {
                Path target = root.resolve(name);

                if (Files.exists(target) && Files.size(target) > 0) {
                    log.debug("Already present: {}", target);
                    continue;
                }

                String url = BASE + "/" + name;
                log.info("Downloading {} ...", url);

                HttpRequest req = HttpRequest.newBuilder(URI.create(url))
                        .header("User-Agent", "monarch-nn/1.0")
                        .GET()
                        .build();

                Path tmp = root.resolve(name + ".part");

                HttpResponse<InputStream> resp = client.send(req,
                        HttpResponse.BodyHandlers.ofInputStream());

                if (resp.statusCode() != 200) {
                    throw new RuntimeException("HTTP " + resp.statusCode()
                            + " downloading " + url);
                }

                try (InputStream in = new BufferedInputStream(resp.body())) {
                    Files.copy(in, tmp, StandardCopyOption.REPLACE_EXISTING);
                }

                // Atomic move so a partial file never becomes visible.
                Files.move(tmp, target,
                        StandardCopyOption.REPLACE_EXISTING,
                        StandardCopyOption.ATOMIC_MOVE);

                log.info("Saved {} ({} bytes)", target, Files.size(target));
            }
        } catch (Exception e) {
            throw new RuntimeException("Failed to download MNIST: " + e.getMessage(), e);
        }
    }
}