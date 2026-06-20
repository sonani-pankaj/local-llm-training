package com.aigrama.llm_training.gpu;

import com.aigrama.llm_training.config.GpuProperties;
import com.aigrama.llm_training.config.GpuProperties.Cpu;
import lombok.extern.slf4j.Slf4j;

/**
 * CPU Backend - Fallback when no GPU is available
 *
 * This backend provides CPU-based computation when no GPU
 * is detected. While slower, it ensures the application
 * always works regardless of hardware configuration.
 *
 * Features:
 * - Multi-threaded computation using all CPU cores
 * - Vectorized operations (AVX-512 if available)
 * - MKL-DNN optimizations for matrix operations
 * - Lower memory requirements than GPU training
 *
 * Teaching Value: This backend is excellent for demonstrating
 * the performance difference between CPU and GPU computation.
 */
@Slf4j
public class CpuBackend implements GpuBackend {

    private final Cpu cpuProperties;
    private boolean initialized = false;
    private int availableCores;

    public CpuBackend(Cpu cpuProperties) {
        this.cpuProperties = cpuProperties;
    }

    private Cpu cpuConfig() {
        return cpuProperties;
    }

    @Override
    public String getBackendName() {
        return String.format("CPU (x86-64, %d threads)", availableCores);
    }

    @Override
    public boolean isAvailable() {
        // CPU is always available
        return true;
    }

    @Override
    public int getDeviceId() {
        return -1; // CPU doesn't have a device ID
    }

    @Override
    public void initialize() {
        log.info("Initializing CPU backend...");

        // Detect available processors
        availableCores = Runtime.getRuntime().availableProcessors();

        // Configure thread pool for optimal performance
        int configuredThreads = cpuConfig() != null && cpuConfig().getThreads() > 0
                ? cpuConfig().getThreads()
                : availableCores;
        System.setProperty("java.util.concurrent.ForkJoinPool.common.parallelism",
                String.valueOf(configuredThreads));

        // Enable MKL optimizations if available
        boolean mklEnabled = cpuConfig() != null && cpuConfig().isMklEnabled();
        if (mklEnabled) {
            System.setProperty("ai.djl.pytorch.mkldnn_enabled", "true");
        }

        // Enable AVX-512 if available
        boolean avx512Enabled = cpuConfig() != null && cpuConfig().isAvx512Enabled();
        if (avx512Enabled) {
            log.info("AVX-512 optimizations enabled");
            System.setProperty("ai.djl.pytorch.avx512_enabled", "true");
        }

        initialized = true;

        log.info("""
            CPU Backend Initialized:
              Available Cores: {}
              Configured Threads: {}
              MKL-DNN: {}
              AVX-512: {}
            """,
                availableCores,
                configuredThreads,
                mklEnabled,
                avx512Enabled
        );
    }

    @Override
    public long getTotalMemory() {
        // Return system RAM
        return Runtime.getRuntime().maxMemory();
    }

    @Override
    public long getFreeMemory() {
        return Runtime.getRuntime().freeMemory();
    }

    @Override
    public String getComputeCapability() {
        return String.format("CPU: %d cores", availableCores);
    }

    @Override
    public void shutdown() {
        log.info("CPU backend shutdown");
        initialized = false;
    }

    @Override
    public boolean supportsFp16() {
        // CPU supports FP16 but it's often slower than FP32
        return false;
    }

    @Override
    public boolean supportsBf16() {
        // Modern CPUs support BF16 via AVX-512 BF16
        return cpuConfig() != null && cpuConfig().isAvx512Enabled();
    }

    @Override
    public int getRecommendedBatchSize(int modelSizeMB) {
        long availableMB = getFreeMemory() / (1024 * 1024);
        // CPU can handle larger batches due to more memory
        long usableMB = availableMB / 2; // Use half of available memory

        if (usableMB <= 0) return 1;

        return (int) Math.max(1, usableMB / (modelSizeMB * 2));
    }

    @Override
    public int getComputeUnitCount() {
        return availableCores;
    }
}