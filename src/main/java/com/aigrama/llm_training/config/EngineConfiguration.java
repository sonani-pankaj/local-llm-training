package com.aigrama.llm_training.config;

import ai.djl.Device;
import ai.djl.engine.Engine;

import com.aigrama.llm_training.gpu.CpuBackend;
import com.aigrama.llm_training.gpu.CudaBackend;
import com.aigrama.llm_training.gpu.GpuBackend;
import com.aigrama.llm_training.gpu.HipDirectMlBackend;
import com.aigrama.llm_training.gpu.RocmBackend;
import lombok.extern.slf4j.Slf4j;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.DependsOn;

/**
 * Engine Configuration - Configures DJL Engine for GPU Usage
 *
 * This configuration sets up the Deep Java Library engine
 * based on the detected GPU backend. It handles:
 *
 * 1. Engine selection (PyTorch for both CUDA and ROCm)
 * 2. Device placement (GPU or CPU)
 * 3. Memory management configuration
 * 4. Optimizer selection based on backend capabilities
 *
 * The engine bridges the gap between Java code and native
 * GPU libraries (CUDA for NVIDIA, ROCm for AMD).
 */
@Slf4j
@Configuration
public class EngineConfiguration {

    private final GpuProperties gpuProperties;
    private final GpuBackend gpuBackend;

    public EngineConfiguration(GpuProperties gpuProperties, GpuBackend gpuBackend) {
        this.gpuProperties = gpuProperties;
        this.gpuBackend = gpuBackend;
    }

    /**
     * Initialize and configure the DJL engine
     */
    @Bean
    @DependsOn("gpuBackend")
    public Engine djlEngine() {
        log.info("Configuring DJL Engine for backend: {}",
                gpuBackend.getBackendName());

        if (gpuBackend instanceof HipDirectMlBackend) {
            // Ensure DJL resolves the ONNX Runtime engine for DirectML on Windows.
            System.setProperty("ai.djl.default_engine", "OnnxRuntime");
        }

        Engine engine = Engine.getInstance();

        // Configure based on backend type
        switch (gpuBackend) {
            case CudaBackend cuda -> configureForCuda(engine);
            case RocmBackend rocm -> configureForRocm(engine);
            case HipDirectMlBackend hip -> configureForHipDirectMl(engine);
            case CpuBackend cpu -> configureForCpu(engine);
            default -> log.warn("Unknown backend type, using default configuration");
        }

        log.info("DJL Engine configured: {}", engine.getEngineName());
        return engine;
    }

    private void configureForCuda(Engine engine) {
        log.info("Applying CUDA-specific optimizations");

        // CUDA-specific settings
        System.setProperty("ai.djl.pytorch.cuda", "true");
        System.setProperty("ai.djl.pytorch.cudnn_enabled", "true");

        // Enable TensorFloat-32 for better performance on Ampere+
        System.setProperty("ai.djl.pytorch.matmul_precision", "high");

        // Memory management
        System.setProperty("PYTORCH_CUDA_ALLOC_CONF","max_split_size_mb:128,garbage_collection_threshold:0.8");
    }

    private void configureForRocm(Engine engine) {
        log.info("Applying ROCm-specific optimizations");

        // ROCm-specific settings
        System.setProperty("ai.djl.pytorch.rocm", "true");
        System.setProperty("ai.djl.pytorch.miopen_enabled", "true");

        // Memory management for ROCm
        System.setProperty("PYTORCH_HIP_ALLOC_CONF","max_split_size_mb:128,garbage_collection_threshold:0.8");

        // Set visible devices
        String visibleDevices = gpuProperties.getRocm().getVisibleDevices();
        System.setProperty("HIP_VISIBLE_DEVICES", visibleDevices);
    }

    private void configureForCpu(Engine engine) {
        log.info("Applying CPU optimizations");

        // Thread configuration
        int threads = gpuProperties.getCpu().getThreads();
        System.setProperty("ai.djl.pytorch.num_interop_threads",
                String.valueOf(threads / 2));
        System.setProperty("ai.djl.pytorch.num_threads",
                String.valueOf(threads));

        // MKL optimizations
        if (gpuProperties.getCpu().isMklEnabled()) {
            System.setProperty("ai.djl.pytorch.mkldnn_enabled", "true");
        }
    }

    private void configureForHipDirectMl(Engine engine) {
        log.info("Applying HIP/DirectML optimizations");
        System.setProperty("ORT_DML_VISIBLE_DEVICES", "0");
        System.setProperty("ai.djl.onnxruntime.directml", "true");
    }

    /**
     * Create appropriate device for training
     */
    @Bean
    @DependsOn("djlEngine")
    public Device trainingDevice() {
        Device device;

        if (gpuBackend instanceof CpuBackend) {
            device = Device.cpu();
            log.info("Training will run on CPU");
        } else {
            int deviceId = gpuBackend.getDeviceId();
            device = Device.gpu(deviceId);
            log.info("Training will run on GPU device {}: {}", deviceId, gpuBackend.getBackendName());
        }

        return device;
    }
}