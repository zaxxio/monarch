package org.monarch.nn.backend;

public enum Backend {
    CPU,       // pure Java, always available
    METAL,     // Apple Silicon GPU via TornadoVM (Metal backend)
    CUDA,      // NVIDIA GPU via TornadoVM (PTX backend), Linux / Windows
    OPENCL,    // not implemented yet - falls back to CPU
    AUTO       // pick best available at runtime
}