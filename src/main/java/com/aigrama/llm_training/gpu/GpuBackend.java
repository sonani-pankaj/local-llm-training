package com.aigrama.llm_training.gpu;

/**
 * Abstract GPU Backend Interface
 *
 * This interface defines the contract for all GPU backends.
 * Implementations provide specific support for:
 * - NVIDIA CUDA (CudaBackend)
 * - AMD ROCm (RocmBackend)
 * - CPU fallback (CpuBackend)
 *
 * This abstraction allows the training system to work with
 * any GPU type without code changes.
 */
public interface GpuBackend {

    /**
     * Get the human-readable name of this backend
     * Example: "NVIDIA CUDA 12.1", "AMD ROCm 6.0", "CPU (x86-64)"
     */
    String getBackendName();

    /**
     * Check if this backend is available and functional
     * Returns false if required drivers/libraries are missing
     */
    boolean isAvailable();

    /**
     * Get the device ID for this GPU
     * Usually 0 for single GPU systems
     */
    int getDeviceId();

    /**
     * Initialize the backend with specific configurations
     * Called before training starts
     */
    void initialize();

    /**
     * Get total available GPU memory in bytes
     */
    long getTotalMemory();

    /**
     * Get currently free GPU memory in bytes
     */
    long getFreeMemory();

    /**
     * Get the compute capability or architecture version
     * Example: "8.9" for NVIDIA, "gfx1100" for AMD
     */
    String getComputeCapability();

    /**
     * Clean up resources when backend is no longer needed
     */
    void shutdown();

    /**
     * Check if mixed precision training (FP16) is supported
     */
    boolean supportsFp16();

    /**
     * Check if bfloat16 training is supported
     */
    boolean supportsBf16();

    /**
     * Get recommended batch size based on available memory
     */
    int getRecommendedBatchSize(int modelSizeMB);

    /**
     * Get the number of compute units/streaming multiprocessors     */
    int getComputeUnitCount();
}