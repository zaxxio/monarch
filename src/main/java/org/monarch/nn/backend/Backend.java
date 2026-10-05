package org.monarch.nn.backend;

public enum Backend {
    CPU,       // pure Java, always available
    METAL,     // Apple Silicon GPU via TornadoVM
    CUDA,      // not implemented yet — falls back to CPU
    OPENCL,    // not implemented yet — falls back to CPU
    AUTO       // pick best available at runtime
}