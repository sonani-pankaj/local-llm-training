package com.aigrama.llm_training;

import ai.djl.Device;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.ConfigurationPropertiesScan;
import org.springframework.context.ConfigurableApplicationContext;
import org.springframework.scheduling.annotation.EnableAsync;

import com.aigrama.llm_training.config.GpuProperties;
import com.aigrama.llm_training.gpu.GpuDetector;

/**
 * LLM Training Platform - Main Application
 * 
 * This application demonstrates LLM training with dynamic GPU switching.
 * 
 * Key Features:
 * - Auto-detection of NVIDIA CUDA or AMD ROCm GPUs
 * - Graceful fallback to CPU if no GPU is available
 * - REST API for training control and monitoring
 * - Spring Boot 4.0 with Java 25 virtual threads
 * 
 * Usage:
 * 1. Set ACTIVE_GPU environment variable: CUDA, ROCM, or CPU
 * 2. Or let the system auto-detect available GPU
 * 3. Start training via REST API: POST /api/training/start
 * 
 * Environment Variables:
 *   ACTIVE_GPU - Force GPU type (CUDA/ROCM/CPU)
 *   CUDA_VISIBLE_DEVICES - Limit visible CUDA devices (NVIDIA)
 *   ROCR_VISIBLE_DEVICES - Limit visible ROCm devices (AMD)
 *   HIP_VISIBLE_DEVICES - Alternative for AMD GPU selection
 */
@Slf4j
@SpringBootApplication
@EnableAsync
@ConfigurationPropertiesScan
public class LlmTrainingApplication {
public static void main(String[] args) {
        // === PRE-INITIALIZATION GPU DETECTION ===
        // This runs before Spring Boot to ensure GPU is configured correctly
        
        String forcedGpu = System.getenv("ACTIVE_GPU");
        if (forcedGpu != null) {
            log.info("""
                GPU manually set via ACTIVE_GPU env var
                  Requested: {}
                """, forcedGpu);
            System.setProperty("gpu.forced", forcedGpu);
        }
        
        // Configure DJL engine before Spring context loads
        System.setProperty("ai.djl.default_engine", "PyTorch");
        System.setProperty("ai.djl.pytorch.num_interop_threads", "4");
        System.setProperty("ai.djl.pytorch.num_threads", "8");
        
        // Java 25: Virtual thread support for async operations
        System.setProperty("jdk.virtualThreadScheduler.parallelism", "16");
        System.setProperty("jdk.virtualThreadScheduler.maxPoolSize", "256");
        
        // === START SPRING BOOT APPLICATION ===
        ConfigurableApplicationContext context = 
            SpringApplication.run(LlmTrainingApplication.class, args);
        
        // === POST-STARTUP GPU VERIFICATION ===
        GpuProperties gpuProps = context.getBean(GpuProperties.class);
        GpuDetector detector = context.getBean(GpuDetector.class);
        Device trainingDevice = context.getBean(Device.class);

        log.info("""
            LLM Training Platform Started Successfully
              Active GPU Backend: {}
              GPU Model: {}
              VRAM Available: {}
              Compute Units: {}
              Training Device: {}
              REST API: http://localhost:8080/api/training
            """,
            gpuProps.getActiveBackend(),
            detector.getGpuModel() != null ? detector.getGpuModel() : "N/A",
            detector.getVramGb() > 0 ? detector.getVramGb() + " GB" : "N/A",
            detector.getComputeUnits() > 0 ? detector.getComputeUnits() : "N/A",
            trainingDevice
        );
    }
}
