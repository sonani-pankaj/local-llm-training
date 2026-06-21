package com.aigrama.llm_training.gpu;

import lombok.extern.slf4j.Slf4j;

import java.io.File;

/**
 * AMD HIP/DirectML Backend for Windows
 * 
 * Uses ONNX Runtime with DirectML execution provider
 * which leverages HIP SDK for AMD GPU acceleration.
 * 
 * Requirements:
 * - Windows 10/11
 * - AMD HIP SDK installed
 * - AMD Radeon RX 7000/9000 series GPU
 * - DirectML compatible driver
 */
@Slf4j
public class HipDirectMlBackend implements GpuBackend {

    private static final long BYTES_PER_GIB = 1024L * 1024L * 1024L;

    private final GpuDetector gpuDetector;
    private boolean initialized = false;

    public HipDirectMlBackend(GpuDetector gpuDetector) {
        this.gpuDetector = gpuDetector;
    }

    @Override
    public String getBackendName() {
        return "AMD HIP/DirectML (Windows)";
    }

    @Override
    public boolean isAvailable() {
        // Check HIP SDK
        String hipPath = System.getenv("HIP_PATH");
        if (hipPath != null && new File(hipPath).exists()) {
            return true;
        }

        // Check common paths
        String[] paths = {
            "C:\\Program Files\\AMD\\ROCm\\6.0",
            "C:\\Program Files\\AMD\\HIP SDK"
        };

        for (String path : paths) {
            if (new File(path).exists()) {
                return true;
            }
        }

        return false;
    }

    @Override
    public int getDeviceId() {
        return 0;
    }

    @Override
    public void initialize() {
        log.info("Initializing AMD HIP/DirectML backend...");

        // Configure ONNX Runtime to use DirectML
        System.setProperty("ORT_DML_VISIBLE_DEVICES", "0");
        System.setProperty("ai.djl.onnxruntime.directml", "true");

        initialized = true;
        log.info("HIP/DirectML backend initialized successfully");
    }

    @Override
    public long getTotalMemory() {
        double vramGb = gpuDetector.getVramGb();
        if (vramGb <= 0) {
            return 0;
        }
        return (long) (vramGb * BYTES_PER_GIB);
    }

    @Override
    public long getFreeMemory() {
        long totalMemory = getTotalMemory();
        if (totalMemory <= 0) {
            return 0;
        }

        // DirectML APIs do not expose free VRAM cleanly; use conservative estimate.
        return (long) (totalMemory * 0.75d);
    }

    @Override
    public String getComputeCapability() {
        return "directml";
    }

    @Override
    public void shutdown() {
        initialized = false;
        System.clearProperty("ai.djl.onnxruntime.directml");
    }

    @Override
    public boolean supportsFp16() {
        String model = gpuDetector.getGpuModel();
        if (model == null) {
            return true;
        }

        String upper = model.toUpperCase();
        return upper.contains("RX 7") || upper.contains("RX 9") || upper.contains("RADEON");
    }

    @Override
    public boolean supportsBf16() {
        String model = gpuDetector.getGpuModel();
        if (model == null) {
            return false;
        }

        String upper = model.toUpperCase();
        return upper.contains("RX 9") || upper.contains("MI300");
    }

    @Override
    public int getRecommendedBatchSize(int modelSizeMB) {
        if (modelSizeMB <= 0) {
            return 1;
        }

        long availableMB = getFreeMemory() / (1024L * 1024L);
        long usableMB = Math.max(0, availableMB - 1024); // reserve 1GB for runtime overhead
        if (usableMB <= 0) {
            return 1;
        }

        return (int) Math.max(1L, usableMB / (modelSizeMB * 3L));
    }

    @Override
    public int getComputeUnitCount() {
        return Math.max(gpuDetector.getComputeUnits(), 0);
    }
}