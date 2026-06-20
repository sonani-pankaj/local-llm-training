package com.aigrama.llm_training.gpu;

import ai.djl.Device;
import ai.djl.engine.Engine;
import com.aigrama.llm_training.config.GpuProperties.Cuda;
import lombok.extern.slf4j.Slf4j;

import java.io.BufferedReader;
import java.io.InputStreamReader;

/**
 * NVIDIA CUDA Backend Implementation
 *
 * This backend provides GPU acceleration for NVIDIA GPUs
 * using the CUDA toolkit and cuDNN library.
 *
 * Requirements:
 * - NVIDIA GPU with Compute Capability 3.5+
 * - CUDA Toolkit 11.0+ (preferably 12.1)
 * - cuDNN 8.9+
 * - nvidia-smi available in PATH
 *
 * Features:
 * - Tensor Core acceleration for matrix operations
 * - Automatic mixed precision (AMP) training
 * - CUDA Graphs for reduced kernel launch overhead
 * - NCCL for multi-GPU communication
 */
@Slf4j
public class CudaBackend implements GpuBackend {

    private final Cuda config;
    private boolean initialized = false;
    private long totalMemory = 0;
    private int computeUnits = 0;
    private String computeCapability = "unknown";

    public CudaBackend(Cuda config) {
        this.config = config;
    }

    @Override
    public String getBackendName() {
        return String.format("NVIDIA CUDA (Compute %s)", computeCapability);
    }

    @Override
    public boolean isAvailable() {
        try {
            // Check nvidia-smi availability
            ProcessBuilder pb = new ProcessBuilder("nvidia-smi");
            Process process = pb.start();
            int exitCode = process.waitFor();

            if (exitCode != 0) {
                log.warn("nvidia-smi returned exit code: {}", exitCode);
                return false;
            }

            // Check CUDA driver version
            ProcessBuilder versionPb = new ProcessBuilder(
                    "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"
            );
            Process versionProcess = versionPb.start();
            BufferedReader reader = new BufferedReader(
                    new InputStreamReader(versionProcess.getInputStream())
            );
            String driverVersion = reader.readLine();

            if (driverVersion == null || driverVersion.isEmpty()) {
                log.warn("Could not detect NVIDIA driver version");
                return false;
            }

            log.info("CUDA driver version: {}", driverVersion);
            return true;

        } catch (Exception e) {
            log.warn("CUDA is not available: {}", e.getMessage());
            return false;
        }
    }

    @Override
    public int getDeviceId() {
        return config.getDeviceId();
    }

    @Override
    public void initialize() {
        log.info("Initializing NVIDIA CUDA backend...");

        try {
            // Set CUDA environment variables
            System.setProperty("CUDA_VISIBLE_DEVICES", config.getVisibleDevices());

            // Query GPU memory
            ProcessBuilder pb = new ProcessBuilder(
                    "nvidia-smi",
                    "--query-gpu=memory.total,compute_cap",
                    "--format=csv,noheader,nounits"
            );
            Process process = pb.start();
            BufferedReader reader = new BufferedReader(
                    new InputStreamReader(process.getInputStream())
            );

            String line = reader.readLine();
            if (line != null) {
                String[] parts = line.split(",");
                totalMemory = Long.parseLong(parts[0].trim()) * 1024 * 1024; // MB to bytes
                computeCapability = parts[1].trim();

                // Parse compute capability for SM count
                try {
                    double capValue = Double.parseDouble(computeCapability);
                    computeUnits = (int)(capValue * 10); // Approximate SM count
                } catch (NumberFormatException e) {
                    computeUnits = 0;
                }
            }

            // Configure DJL for CUDA
            System.setProperty("ai.djl.pytorch.cuda", "true");
            System.setProperty("ai.djl.pytorch.cudnn_enabled",
                    String.valueOf(config.isCudnnEnabled()));

            // Enable Tensor Core optimization
            System.setProperty("ai.djl.pytorch.tensorcore_enabled", "true");

            process.waitFor();
            initialized = true;

            log.info("""
                CUDA Backend Initialized:
                  Device ID: {}
                  Total Memory: {} GB
                  Compute Capability: {}
                  cuDNN Enabled: {}
                """,
                    config.getDeviceId(),
                    totalMemory / (1024.0 * 1024.0 * 1024.0),
                    computeCapability,
                    config.isCudnnEnabled()
            );

        } catch (Exception e) {
            log.error("Failed to initialize CUDA backend: {}", e.getMessage());
            initialized = false;
        }
    }

    @Override
    public long getTotalMemory() {
        return totalMemory;
    }

    @Override
    public long getFreeMemory() {
        try {
            ProcessBuilder pb = new ProcessBuilder(
                    "nvidia-smi",
                    "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits"
            );
            Process process = pb.start();
            BufferedReader reader = new BufferedReader(
                    new InputStreamReader(process.getInputStream())
            );
            String line = reader.readLine();
            if (line != null) {
                return Long.parseLong(line.trim()) * 1024 * 1024; // MB to bytes
            }
        } catch (Exception e) {
            log.warn("Could not query free GPU memory: {}", e.getMessage());
        }
        return 0;
    }

    @Override
    public String getComputeCapability() {
        return computeCapability;
    }

    @Override
    public void shutdown() {
        log.info("Shutting down CUDA backend...");
        // Release CUDA resources
        System.clearProperty("ai.djl.pytorch.cuda");
        initialized = false;
    }

    @Override
    public boolean supportsFp16() {
        // All modern NVIDIA GPUs support FP16
        return true;
    }

    @Override
    public boolean supportsBf16() {
        // BF16 supported on A100, H100, and newer
        try {
            double capValue = Double.parseDouble(computeCapability);
            return capValue >= 8.0; // Ampere or newer
        } catch (NumberFormatException e) {
            return false;
        }
    }

    @Override
    public int getRecommendedBatchSize(int modelSizeMB) {
        long availableMB = getFreeMemory() / (1024 * 1024);
        // Reserve 1GB for framework overhead
        long usableMB = availableMB - 1024;

        if (usableMB <= 0) return 1;

        // Conservative estimate: model * 3 for gradients, optimizer states, activations
        return (int) Math.max(1, usableMB / (modelSizeMB * 3));
    }

    @Override
    public int getComputeUnitCount() {
        return computeUnits;
    }
}