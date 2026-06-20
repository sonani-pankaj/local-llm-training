package com.aigrama.llm_training.config;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

/**
 * GPU Properties Configuration
 * 
 * This class maps YAML configuration for GPU settings.
 * It supports both NVIDIA CUDA and AMD ROCm configurations
 * in the same application.yml file.
 * 
 * Configuration priority:
 * 1. Environment variable ACTIVE_GPU
 * 2. System property gpu.forced
 * 3. Auto-detection
 * 4. Default to CPU
 */
@Data
@Component
@ConfigurationProperties(prefix = "gpu")
public class GpuProperties {
     
    private String activeBackend = "CPU"; // Active GPU backend: CUDA, ROCM, or CPU
    private boolean enabled = true; //Whether to use GPU if available
    
    /**
     * NVIDIA CUDA specific settings
     */
    private Cuda cuda = new Cuda();
    
    /**
     * AMD ROCm specific settings
     */
    private Rocm rocm = new Rocm();
    
    /**
     * CPU fallback settings
     */
    private Cpu cpu = new Cpu();
    
    /**
     * Memory management settings
     */
    private Memory memory = new Memory();
    
    @Data
    public static class Cuda {
        /**
         * CUDA device ID (usually 0)
         */
        private int deviceId = 0;
        
        /**
         * CUDA compute capability (auto-detected)
         */
        private String computeCapability = "auto";
        
        /**
         * Enable cuDNN acceleration
         */
        private boolean cudnnEnabled = true;
        
        /**
         * CUDA-specific PyTorch backend
         */
        private String pytorchBackend = "cu121";
        
        /**
         * CUDA visible devices (comma-separated)
         */
        private String visibleDevices = "0";
    }
    
    @Data
    public static class Rocm {
        /**
         * ROCm device ID
         */
        private int deviceId = 0;
        
        /**
         * ROCm version
         */
        private String version = "6.0.0";
        
        /**
         * Enable MIOpen acceleration
         */
        private boolean miopenEnabled = true;
        
        /**
         * ROCm-specific PyTorch backend
         */
        private String pytorchBackend = "rocm6.0";
        
        /**
         * ROCm visible devices
         */
        private String visibleDevices = "0";
        
        /**
         * HIP compiler path
         */
        private String hipPath = "/opt/rocm/bin";
    }
    
    @Data
    public static class Cpu {
        /**
         * Number of threads for CPU training
         */
        private int threads = 16;
        
        /**
         * Enable MKL-DNN optimizations
         */
        private boolean mklEnabled = true;
        
        /**
         * Use AVX-512 if available
         */
        private boolean avx512Enabled = true;
    }
    
    @Data
    public static class Memory {
        /**
         * GPU memory fraction to use (0.0 to 1.0)
         */
        private double gpuMemoryFraction = 0.9;
        
        /**
         * Enable memory growth (allocate as needed)
         */
        private boolean allowGrowth = true;
        
        /**
         * CPU memory limit for offloading
         */
        private String cpuOffloadLimit = "8GB";
    }
    
    /**
     * Get the training device string for DJL
     */
    public String getTrainingDevice() {
        return switch (activeBackend.toUpperCase()) {
            case "CUDA" -> String.format("gpu{%d}", cuda.deviceId);
            case "ROCM" -> String.format("gpu{%d}", rocm.deviceId);
            case "HIP", "AMD_HIP" -> "gpu{0}";
            default -> "cpu";
        };
    }
    
    /**
     * Get the appropriate PyTorch backend string
     */
    public String getPytorchBackend() {
        return switch (activeBackend.toUpperCase()) {
            case "CUDA" -> cuda.pytorchBackend;
            case "ROCM" -> rocm.pytorchBackend;
            case "HIP", "AMD_HIP" -> "directml";
            default -> "cpu";
        };
    }
}