package com.aigrama.llm_training.config;


import com.aigrama.llm_training.gpu.*;
import lombok.extern.slf4j.Slf4j;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.DependsOn;

/**
 * GPU Configuration - Creates appropriate GPU backend beans
 *
 * This configuration class is responsible for:
 * 1. Detecting available GPU hardware
 * 2. Creating the appropriate backend implementation
 * 3. Configuring the DJL engine for GPU usage
 * 4. Providing fallback mechanisms
 *
 * The configuration follows a chain of responsibility pattern:
 *   User Config → Environment Variables → Auto-Detection → CPU Fallback
 */
@Slf4j
@Configuration
public class GpuConfiguration {

    private final GpuProperties gpuProperties;
    private final GpuDetector gpuDetector;

    public GpuConfiguration(GpuProperties gpuProperties, GpuDetector gpuDetector) {
        this.gpuProperties = gpuProperties;
        this.gpuDetector = gpuDetector;
    }

    /**
     * Primary GPU backend bean
     * Uses @DependsOn to ensure detector runs first
     */
    @Bean
    @DependsOn("gpuDetector")
    public GpuBackend gpuBackend() {
        // Step 1: Check for manual override
        String forcedGpu = System.getProperty("gpu.forced");
        if (forcedGpu != null) {
            return createForcedBackend(forcedGpu);
        }

        // Step 2: Check if GPU is disabled in config
        if (!gpuProperties.isEnabled()) {
            log.info("GPU disabled in configuration, using CPU backend");
            return new CpuBackend(gpuProperties.getCpu());
        }

        // Step 3: Auto-detect and create appropriate backend
        return createAutoDetectedBackend();
    }

    /**
     * Create a backend based on manual override
     */
    private GpuBackend createForcedBackend(String gpuType) {
        log.info("Creating forced GPU backend: {}", gpuType);

        String normalizedGpuType = gpuType.toUpperCase();
        gpuProperties.setActiveBackend(normalizedGpuType);

        return switch (normalizedGpuType) {
            case "CUDA" -> {
                log.info("Forcing NVIDIA CUDA backend");
                yield new CudaBackend(gpuProperties.getCuda());
            }
            case "HIP", "AMD_HIP" -> {
                log.info("Forcing AMD HIP/DirectML backend");
                yield new HipDirectMlBackend(gpuDetector);
            }
            case "ROCM" -> {
                log.info("Forcing AMD ROCm backend");
                yield new RocmBackend(gpuProperties.getRocm());
            }
            default -> {
                log.info("Forcing CPU backend (unknown type: {})", gpuType);
                yield new CpuBackend(gpuProperties.getCpu());
            }
        };
    }

    /**
     * Auto-detect GPU and create appropriate backend
     */
    private GpuBackend createAutoDetectedBackend() {
        String detectedType = gpuDetector.getGpuType();
        String normalizedDetectedType = detectedType.toUpperCase();

        log.info("Auto-detected GPU type: {}", detectedType);
        gpuProperties.setActiveBackend(normalizedDetectedType);

        return switch (normalizedDetectedType) {
            case "CUDA" -> {
                log.info("NVIDIA CUDA backend selected | GPU: {} | VRAM: {} GB",
                        gpuDetector.getGpuModel(),
                        gpuDetector.getVramGb());
                yield new CudaBackend(gpuProperties.getCuda());
            }
            case "HIP", "AMD_HIP" -> {
                log.info("AMD HIP/DirectML backend selected | GPU: {} | VRAM: {} GB",
                        gpuDetector.getGpuModel(),
                        gpuDetector.getVramGb());
                yield new HipDirectMlBackend(gpuDetector);
            }
            case "ROCM" -> {
                log.info("AMD ROCm backend selected | GPU: {} | VRAM: {} GB",
                        gpuDetector.getGpuModel(),
                        gpuDetector.getVramGb());
                yield new RocmBackend(gpuProperties.getRocm());
            }
            default -> {
                log.info("No compatible GPU found, using CPU backend");
                yield new CpuBackend(gpuProperties.getCpu());
            }
        };
    }


    /**
     * Log GPU configuration summary
     */
    @Bean
    public String gpuConfigurationSummary(GpuBackend gpuBackend) {
        log.info("""
            GPU Configuration Summary:
              Backend: {}
              Available: {}
              Device ID: {}
              Backend Name: {}
            """,
                gpuBackend.getClass().getSimpleName(),
                gpuBackend.isAvailable(),
                gpuBackend.getDeviceId(),
                gpuBackend.getBackendName()
        );

        return "GPU Configuration Complete";
    }
}