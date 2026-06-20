package com.aigrama.llm_training.controller;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import com.aigrama.llm_training.gpu.CpuBackend;
import com.aigrama.llm_training.gpu.GpuBackend;
import com.aigrama.llm_training.gpu.GpuDetector;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static java.util.Map.entry;

/**
 * GPU Information Controller
 * REST API endpoint for querying GPU configuration and status.
 * Provides real-time information about:
 * - Detected GPU type and model
 * - Memory usage
 * - Compute capabilities
 * - Backend status
 */
@Slf4j
@RestController
@RequestMapping("/api/gpu")
public class GpuInfoController {

    private static final double BYTES_PER_GIB = 1024.0 * 1024.0 * 1024.0;
    private static final int DEFAULT_MODEL_SIZE_MB = 500;

    private final GpuDetector gpuDetector;
    private final GpuBackend gpuBackend;
    private final Device trainingDevice;

    public GpuInfoController(GpuDetector gpuDetector, GpuBackend gpuBackend, Device trainingDevice) {
        this.gpuDetector = gpuDetector;
        this.gpuBackend = gpuBackend;
        this.trainingDevice = trainingDevice;
    }

    /**
     * Get complete GPU information
     */
    @GetMapping("/info")
    public ResponseEntity<Map<String, Object>> getGpuInfo() {
        Map<String, Object> gpuInfo = buildGpuInfo();
        return ResponseEntity.ok(gpuInfo);
    }

    private Map<String, Object> buildGpuInfo() {
        return Map.ofEntries(
                entry("detectedType", gpuDetector.getGpuType()),
                entry("gpuModel", gpuDetector.getGpuModel()),
                entry("activeBackend", gpuBackend.getBackendName()),
                entry("vramGb", gpuDetector.getVramGb()),
                entry("computeUnits", gpuBackend.getComputeUnitCount()),
                entry("computeCapability", gpuBackend.getComputeCapability()),
                entry("cudaAvailable", gpuDetector.isCudaAvailable()),
                entry("rocmAvailable", gpuDetector.isRocmAvailable()),
                entry("supportsFp16", gpuBackend.supportsFp16()),
                entry("supportsBf16", gpuBackend.supportsBf16()),
                entry("recommendedBatchSize", gpuBackend.getRecommendedBatchSize(DEFAULT_MODEL_SIZE_MB)),
                entry("totalMemoryGb", toGib(gpuBackend.getTotalMemory())),
                entry("freeMemoryGb", toGib(gpuBackend.getFreeMemory())),
                entry("trainingDevice", trainingDevice.toString())
        );
    }

    private double toGib(long bytes) {
        return bytes / BYTES_PER_GIB;
    }

    /**
     * Switch GPU backend at runtime
     */
    @PostMapping("/switch")
    public ResponseEntity<String> switchBackend(@RequestParam String backend) {
        log.info("Request to switch GPU backend to: {}", backend);

        return switch (backend.toUpperCase()) {
            case "CUDA" -> {
                if (gpuDetector.isCudaAvailable()) {
                    System.setProperty("gpu.forced", "CUDA");
                    yield ResponseEntity.ok("Switched to CUDA backend. Restart required.");
                }
                yield ResponseEntity.badRequest().body("CUDA not available");
            }
            case "ROCM" -> {
                if (gpuDetector.isRocmAvailable()) {
                    System.setProperty("gpu.forced", "ROCM");
                    yield ResponseEntity.ok("Switched to ROCm backend. Restart required.");
                }
                yield ResponseEntity.badRequest().body("ROCm not available");
            }
            case "CPU" -> {
                System.setProperty("gpu.forced", "CPU");
                yield ResponseEntity.ok("Switched to CPU backend. Restart required.");
            }
            default -> ResponseEntity.badRequest()
                    .body("Invalid backend. Use: CUDA, ROCM, or CPU");
        };
    }

    /**
     * Check GPU health
     */
    @GetMapping("/health")
    public ResponseEntity<Map<String, Boolean>> getGpuHealth() {
        return ResponseEntity.ok(Map.of(
                "available", gpuBackend.isAvailable(),
                "initialized", gpuBackend instanceof CpuBackend || gpuBackend.getFreeMemory() > 0
        ));
    }

    /**
     * Benchmark GPU performance
     */
    @PostMapping("/benchmark")
    public ResponseEntity<Map<String, Object>> runBenchmark() {
        log.info("Starting GPU benchmark...");

        if (!gpuBackend.isAvailable()) {
            return ResponseEntity.badRequest().body(Map.of(
                    "status", "unavailable",
                    "message", "Current backend is not available for benchmarking",
                    "backend", gpuBackend.getBackendName()
            ));
        }

        try {
            Map<String, Object> result = runTensorBenchmark();
            return ResponseEntity.ok(result);
        } catch (UnsupportedOperationException | IllegalArgumentException e) {
            log.warn("DJL backend benchmark is not supported on this runtime, falling back to Java benchmark: {}", e.getMessage());
            return ResponseEntity.ok(runJavaBenchmark(e.getMessage()));
        } catch (Exception e) {
            log.error("GPU benchmark failed", e);
            return ResponseEntity.ok(runJavaBenchmark(e.getMessage()));
        }
    }

    private Map<String, Object> runTensorBenchmark() {
        final int warmupIterations = 2;
        final int measuredIterations = 5;
        final int matrixSize = determineMatrixSize();
        final Shape matrixShape = new Shape(matrixSize, matrixSize);
        final List<Double> iterationMs = new ArrayList<>();

        try (NDManager manager = NDManager.newBaseManager(trainingDevice)) {
            for (int i = 0; i < warmupIterations; i++) {
                executeMatMulPass(manager, matrixShape);
            }

            for (int i = 0; i < measuredIterations; i++) {
                long start = System.nanoTime();
                executeMatMulPass(manager, matrixShape);
                long elapsedNs = System.nanoTime() - start;
                iterationMs.add(elapsedNs / 1_000_000.0);
            }
        }

        double averageMs = iterationMs.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        double bestMs = iterationMs.isEmpty() ? 0.0 : Collections.min(iterationMs);
        double worstMs = iterationMs.isEmpty() ? 0.0 : Collections.max(iterationMs);
        double gflops = averageMs > 0
                ? (2.0 * matrixSize * matrixSize * matrixSize) / (averageMs / 1000.0) / 1_000_000_000.0
                : 0.0;

        Map<String, Object> result = new HashMap<>();
        result.put("status", "success");
        result.put("backend", gpuBackend.getBackendName());
        result.put("device", trainingDevice.toString());
        result.put("matrixSize", matrixSize);
        result.put("warmupIterations", warmupIterations);
        result.put("measuredIterations", measuredIterations);
        result.put("iterationMs", iterationMs);
        result.put("averageMs", averageMs);
        result.put("bestMs", bestMs);
        result.put("worstMs", worstMs);
        result.put("estimatedGflops", gflops);
        result.put("computeUnits", gpuBackend.getComputeUnitCount());
        result.put("freeMemoryGb", toGib(gpuBackend.getFreeMemory()));
        return result;
    }

    private Map<String, Object> runJavaBenchmark(String fallbackReason) {
        final int warmupIterations = 2;
        final int measuredIterations = 5;
        final int matrixSize = determineMatrixSize();

        double[][] a = new double[matrixSize][matrixSize];
        double[][] b = new double[matrixSize][matrixSize];
        double[][] c = new double[matrixSize][matrixSize];

        for (int i = 0; i < matrixSize; i++) {
            for (int j = 0; j < matrixSize; j++) {
                a[i][j] = (i + j) * 0.001d;
                b[i][j] = (i - j) * 0.001d;
            }
        }

        for (int i = 0; i < warmupIterations; i++) {
            multiply(a, b, c);
        }

        List<Double> iterationValues = new ArrayList<>();
        for (int i = 0; i < measuredIterations; i++) {
            long start = System.nanoTime();
            multiply(a, b, c);
            long elapsedNs = System.nanoTime() - start;
            iterationValues.add(elapsedNs / 1_000_000.0);
        }

        double averageMs = iterationValues.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        double bestMs = iterationValues.isEmpty() ? 0.0 : Collections.min(iterationValues);
        double worstMs = iterationValues.isEmpty() ? 0.0 : Collections.max(iterationValues);
        double gflops = averageMs > 0
                ? (2.0 * matrixSize * matrixSize * matrixSize) / (averageMs / 1000.0) / 1_000_000_000.0
                : 0.0;

        Map<String, Object> result = new HashMap<>();
        result.put("status", "success");
        result.put("mode", "fallback-java");
        result.put("fallbackReason", fallbackReason);
        result.put("backend", gpuBackend.getBackendName());
        result.put("device", trainingDevice.toString());
        result.put("matrixSize", matrixSize);
        result.put("warmupIterations", warmupIterations);
        result.put("measuredIterations", measuredIterations);
        result.put("iterationMs", iterationValues);
        result.put("averageMs", averageMs);
        result.put("bestMs", bestMs);
        result.put("worstMs", worstMs);
        result.put("estimatedGflops", gflops);
        result.put("computeUnits", gpuBackend.getComputeUnitCount());
        result.put("freeMemoryGb", toGib(gpuBackend.getFreeMemory()));
        return result;
    }

    private void multiply(double[][] a, double[][] b, double[][] c) {
        int n = a.length;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0d;
                for (int k = 0; k < n; k++) {
                    sum += a[i][k] * b[k][j];
                }
                c[i][j] = sum;
            }
        }
    }

    private void executeMatMulPass(NDManager manager, Shape matrixShape) {
        try (NDArray left = manager.randomUniform(0f, 1f, matrixShape, DataType.FLOAT32);
             NDArray right = manager.randomUniform(0f, 1f, matrixShape, DataType.FLOAT32)) {
            NDArray output = left.matMul(right);
            output.getFloat();
        }
    }

    private int determineMatrixSize() {
        long freeMemoryBytes = Math.max(0L, gpuBackend.getFreeMemory());
        long freeMemoryMb = freeMemoryBytes / (1024L * 1024L);

        if (gpuBackend instanceof CpuBackend) {
            return 256;
        }

        if (freeMemoryMb >= 16_000) {
            return 2048;
        }
        if (freeMemoryMb >= 8_000) {
            return 1536;
        }
        if (freeMemoryMb >= 4_000) {
            return 1024;
        }
        return 512;
    }
}