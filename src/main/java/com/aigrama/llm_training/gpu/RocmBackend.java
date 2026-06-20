package com.aigrama.llm_training.gpu;

import ai.djl.Device;

import com.aigrama.llm_training.config.GpuProperties.Rocm;
import lombok.extern.slf4j.Slf4j;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

/**
 * AMD ROCm Backend Implementation
 *
 * This backend provides GPU acceleration for AMD GPUs
 * using the ROCm open-source compute stack.
 *
 * Requirements:
 * - AMD GPU with ROCm support (RX 7000 series, RX 9000 series, Instinct)
 * - ROCm 5.7+ (preferably 6.0+)
 * - MIOpen for optimized deep learning operations
 * - rocm-smi available in PATH
 *
 * Features:
 * - HIP (Heterogeneous Interface for Portability) runtime
 * - MIOpen for convolution and attention optimizations
 * - Support for mixed precision training
 * - Open source and vendor-neutral
 *
 * Note: ROCm is AMD's answer to NVIDIA CUDA, providing
 * similar functionality through an open-source stack.
 */
@Slf4j
public class RocmBackend implements GpuBackend {

    private final Rocm config;
    private boolean initialized = false;
    private long totalMemory = 0;
    private int computeUnits = 0;
    private String gpuArchitecture = "unknown";

    public RocmBackend(Rocm config) {
        this.config = config;
    }

    @Override
    public String getBackendName() {
        return String.format("AMD ROCm %s (%s)", config.getVersion(), gpuArchitecture);
    }

    @Override
    public boolean isAvailable() {
        try {
            // Check rocm-smi availability
            ProcessBuilder pb = new ProcessBuilder("rocm-smi", "--showproductname");
            Process process = pb.start();
            BufferedReader reader = new BufferedReader(
                    new InputStreamReader(process.getInputStream())
            );

            StringBuilder output = new StringBuilder();
            String line;
            boolean foundGpu = false;

            while ((line = reader.readLine()) != null) {
                output.append(line).append("\n");
                if (line.contains("GPU") || line.contains("Card")) {
                    foundGpu = true;
                }
            }

            int exitCode = process.waitFor();

            if (!foundGpu || exitCode != 0) {
                log.warn("No AMD GPU found via rocm-smi");
                return false;
            }

            log.info("AMD GPU detected via ROCm: {}", output.toString().trim());
            return true;

        } catch (Exception e) {
            log.warn("ROCm is not available: {}", e.getMessage());
            return false;
        }
    }

    @Override
    public int getDeviceId() {
        return config.getDeviceId();
    }

    @Override
    public void initialize() {
        log.info("Initializing AMD ROCm backend...");

        try {
            // Set ROCm environment variables
            System.setProperty("ROCR_VISIBLE_DEVICES", config.getVisibleDevices());
            System.setProperty("HIP_VISIBLE_DEVICES", config.getVisibleDevices());

            // Set ROCm path
            System.setProperty("ROCM_PATH", "/opt/rocm");
            System.setProperty("HIP_PATH", config.getHipPath());

            // Query GPU information
            queryGpuInfo();

            // Configure DJL for ROCm
            System.setProperty("ai.djl.pytorch.rocm", "true");
            System.setProperty("ai.djl.pytorch.miopen_enabled",
                    String.valueOf(config.isMiopenEnabled()));

            initialized = true;

            log.info("""
                ROCm Backend Initialized:
                  Device ID: {}
                  Total Memory: {} GB
                  GPU Architecture: {}
                  MIOpen Enabled: {}
                  ROCm Version: {}
                """,
                    config.getDeviceId(),
                    totalMemory / (1024.0 * 1024.0 * 1024.0),
                    gpuArchitecture,
                    config.isMiopenEnabled(),
                    config.getVersion()
            );

        } catch (Exception e) {
            log.error("Failed to initialize ROCm backend: {}", e.getMessage());
            initialized = false;
        }
    }

    /**
     * Query AMD GPU information using rocm-smi
     */
    private void queryGpuInfo() throws IOException, InterruptedException {
        // Query VRAM
        ProcessBuilder vramPb = new ProcessBuilder(
                "rocm-smi", "--showmeminfo", "vram"
        );
        Process vramProcess = vramPb.start();
        BufferedReader vramReader = new BufferedReader(
                new InputStreamReader(vramProcess.getInputStream())
        );

        String line;
        while ((line = vramReader.readLine()) != null) {
            if (line.contains("VRAM")) {
                try {
                    // Parse VRAM value (format: "VRAM: XXXX MB")
                    String[] parts = line.split(":");
                    if (parts.length >= 2) {
                        String vramStr = parts[1].replaceAll("[^0-9]", "");
                        if (!vramStr.isEmpty()) {
                            totalMemory = Long.parseLong(vramStr) * 1024 * 1024;
                        }
                    }
                } catch (NumberFormatException e) {
                    log.warn("Could not parse VRAM from: {}", line);
                }
            }
        }
        vramProcess.waitFor();

        // Query GPU architecture
        ProcessBuilder archPb = new ProcessBuilder(
                "rocm-smi", "--showproductname"
        );
        Process archProcess = archPb.start();
        BufferedReader archReader = new BufferedReader(
                new InputStreamReader(archProcess.getInputStream())
        );

        while ((line = archReader.readLine()) != null) {
            if (line.contains("Card series")) {
                gpuArchitecture = line.split(":")[1].trim();
            }
        }
        archProcess.waitFor();

        // Estimate compute units based on GPU model
        if (gpuArchitecture.contains("RX 9070")) {
            computeUnits = 64; // RX 9070 XT: 64 CUs
        } else if (gpuArchitecture.contains("RX 7900")) {
            computeUnits = 84; // RX 7900 XTX: 84 CUs
        } else if (gpuArchitecture.contains("RX 6800")) {
            computeUnits = 60; // RX 6800 XT: 60 CUs
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
                    "rocm-smi", "--showmeminfo", "vram", "--json"
            );
            Process process = pb.start();
            BufferedReader reader = new BufferedReader(
                    new InputStreamReader(process.getInputStream())
            );

            StringBuilder json = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                json.append(line);
            }

            // Simple JSON parsing (in production, use proper JSON parser)
            String jsonStr = json.toString();
            if (jsonStr.contains("\"VRAM Total Used Memory")) {
                int usedIdx = jsonStr.indexOf("\"VRAM Total Used Memory");
                // Parse the value after the key
                // This is a simplified approach
            }

            process.waitFor();
        } catch (Exception e) {
            log.warn("Could not query free GPU memory: {}", e.getMessage());
        }

        // Conservative estimate: return 80% of total as free
        return (long)(totalMemory * 0.8);
    }

    @Override
    public String getComputeCapability() {
        return gpuArchitecture;
    }

    @Override
    public void shutdown() {
        log.info("Shutting down ROCm backend...");
        System.clearProperty("ai.djl.pytorch.rocm");
        initialized = false;
    }

    @Override
    public boolean supportsFp16() {
        // RDNA3 (RX 7000) and newer support FP16
        return gpuArchitecture.contains("RX 7") ||
                gpuArchitecture.contains("RX 9") ||
                gpuArchitecture.contains("Instinct");
    }

    @Override
    public boolean supportsBf16() {
        // BF16 support depends on specific architecture
        // RDNA3 has some BF16 support
        return gpuArchitecture.contains("RX 9") ||
                gpuArchitecture.contains("Instinct MI300");
    }

    @Override
    public int getRecommendedBatchSize(int modelSizeMB) {
        long availableMB = getFreeMemory() / (1024 * 1024);
        long usableMB = availableMB - 1024; // Reserve 1GB

        if (usableMB <= 0) return 1;

        return (int) Math.max(1, usableMB / (modelSizeMB * 3));
    }

    @Override
    public int getComputeUnitCount() {
        return computeUnits;
    }
}
