package com.aigrama.llm_training.gpu;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import oshi.SystemInfo;
import oshi.hardware.CentralProcessor;
import oshi.hardware.HardwareAbstractionLayer;

import java.io.File;
import java.io.BufferedReader;
import java.io.InputStreamReader;


/**
 * GPU Detector - Automatically identifies available GPU hardware
 * 
 * This component detects whether the system has:
 * - NVIDIA GPU (via nvidia-smi)
 * - AMD GPU (via rocm-smi or lspci)
 * - Intel GPU (via lspci)
 * 
 * It also queries detailed specifications like VRAM, compute units,
 * and driver versions to optimize training configuration.
 * 
 * Detection order:
 * 1. Check CUDA (nvidia-smi)
 * 2. Check ROCm (rocm-smi)
 * 3. Check OpenCL devices
 * 4. Fall back to CPU information
 */
@Slf4j
@Component
public class GpuDetector {
    
    private final SystemInfo systemInfo;
    private final HardwareAbstractionLayer hardware;
    
    @Getter
    private String gpuType = "CPU";
    
    @Getter
    private String gpuModel = "Unknown";
    
    @Getter
    private long vramBytes = 0;
    
    @Getter
    private int computeUnits = 0;
    
    @Getter
    private String driverVersion = "N/A";
    
    @Getter
    private boolean cudaAvailable = false;
    
    @Getter
    private boolean rocmAvailable = false;

    @Getter
    private boolean hipAvailable = false;

    @Getter
    private boolean openclAvailable = false;
    
    public GpuDetector() {
        this.systemInfo = new SystemInfo();
        this.hardware = systemInfo.getHardware();
        
        // Run detection immediately
        detectGpu();
    }
    
    /**
     * Main detection method - identifies GPU type and capabilities
     */
    private void detectGpu() {
        log.info("Starting GPU detection...");
        
        // Step 1: Check for NVIDIA GPU via nvidia-smi
        if (detectNvidiaGpu()) {
            gpuType = "CUDA";
            cudaAvailable = true;
            log.info("NVIDIA GPU detected with CUDA support");
            return;
        }

        // Step 2: Check for AMD GPU via HIP/DirectML (Windows)
        if (detectAmdHipWindows()) {
            gpuType = "HIP";
            hipAvailable = true;
            log.info("AMD GPU detected with HIP/DirectML support");
            return;
        }

        // Step 3: Check for AMD GPU via ROCm
        if (detectAmdGpu()) {
            gpuType = "ROCM";
            rocmAvailable = true;
            log.info("AMD GPU detected with ROCm support");
            return;
        }
        
        // Step 4: Check for any GPU via OpenCL
        if (detectOpenCL()) {
            openclAvailable = true;
            log.info("GPU detected via OpenCL (limited support)");
            gpuType = "OPENCL";
            return;
        }
        
        // Step 5: Fall back to CPU
        detectCpuInfo();
        gpuType = "CPU";
        log.info("No GPU detected, using CPU for computation");
    }

    /**
     * Detect AMD GPU with HIP tooling on Windows.
     */
    private boolean detectAmdHipWindows() {
        String osName = System.getProperty("os.name", "").toLowerCase();
        if (!osName.contains("win")) {
            return false;
        }

        if (!isHipInstalledOnWindows()) {
            return false;
        }

        boolean infoPopulated = populateAmdInfoFromHipInfo() || populateAmdInfoFromWmic();

        if (!infoPopulated) {
            gpuModel = "AMD GPU (HIP)";
        }

        if (vramBytes <= 0) {
            populateVramFromWmic();
        }

        // Some Windows drivers report very small placeholder AdapterRAM values.
        if (vramBytes <= (1L * 1024 * 1024 * 1024)) {
            long inferredVram = inferAmdVramFromModel(gpuModel);
            if (inferredVram > 0) {
                vramBytes = inferredVram;
            }
        }

        if (computeUnits <= 0) {
            computeUnits = inferAmdComputeUnitsFromModel(gpuModel);
        }

        driverVersion = driverVersion.equals("N/A") ? "HIP/DirectML" : driverVersion;
        return true;
    }

    private boolean isHipInstalledOnWindows() {
        try {
            ProcessBuilder pb = new ProcessBuilder("cmd", "/c", "hipconfig --version");
            pb.redirectErrorStream(true);
            Process process = pb.start();
            int exitCode = process.waitFor();
            if (exitCode == 0) {
                return true;
            }
        } catch (Exception e) {
            log.debug("hipconfig check failed: {}", e.getMessage());
        }

        String hipPath = System.getenv("HIP_PATH");
        if (hipPath != null && !hipPath.isBlank() && new File(hipPath).exists()) {
            return true;
        }

        String programFiles = System.getenv("ProgramFiles");
        String[] commonPaths = {
                "C:\\Program Files\\AMD\\ROCm\\6.0\\bin\\hipconfig.exe",
                "C:\\Program Files\\AMD\\ROCm\\5.7\\bin\\hipconfig.exe",
                "C:\\Program Files\\AMD\\HIP SDK\\bin\\hipconfig.exe",
                (programFiles != null ? programFiles + "\\AMD\\HIP SDK\\bin\\hipconfig.exe" : "")
        };

        for (String path : commonPaths) {
            if (!path.isBlank() && new File(path).exists()) {
                return true;
            }
        }

        return false;
    }

    private boolean populateAmdInfoFromHipInfo() {
        try {
            ProcessBuilder pb = new ProcessBuilder("cmd", "/c", "hipinfo");
            pb.redirectErrorStream(true);
            Process process = pb.start();

            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            String line;
            boolean foundName = false;

            while ((line = reader.readLine()) != null) {
                String trimmed = line.trim();
                if (trimmed.startsWith("Name:")) {
                    gpuModel = trimmed.substring("Name:".length()).trim();
                    foundName = true;
                }
                if (trimmed.startsWith("gcnArchName:")) {
                    driverVersion = trimmed.substring("gcnArchName:".length()).trim();
                }
                if (trimmed.startsWith("Total Global Memory:")) {
                    String memStr = trimmed.replaceAll("[^0-9]", "");
                    if (!memStr.isEmpty()) {
                        try {
                            vramBytes = Long.parseLong(memStr);
                        } catch (NumberFormatException ignored) {
                            // Best-effort parse only.
                        }
                    }
                }
            }

            process.waitFor();
            return foundName;
        } catch (Exception e) {
            log.debug("hipinfo query failed: {}", e.getMessage());
            return false;
        }
    }

    private boolean populateAmdInfoFromWmic() {
        try {
            ProcessBuilder pb = new ProcessBuilder("cmd", "/c", "wmic path win32_VideoController get Name,DriverVersion /format:list");
            pb.redirectErrorStream(true);
            Process process = pb.start();

            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            String line;
            String pendingName = null;
            while ((line = reader.readLine()) != null) {
                String trimmed = line.trim();
                if (trimmed.startsWith("Name=")) {
                    pendingName = trimmed.substring("Name=".length()).trim();
                } else if (trimmed.startsWith("DriverVersion=") && pendingName != null) {
                    String upper = pendingName.toUpperCase();
                    if (upper.contains("AMD") || upper.contains("RADEON")) {
                        gpuModel = pendingName;
                        String parsedDriver = trimmed.substring("DriverVersion=".length()).trim();
                        if (!parsedDriver.isBlank()) {
                            driverVersion = parsedDriver;
                        }
                        process.waitFor();
                        return true;
                    }
                    pendingName = null;
                }
            }

            process.waitFor();
            return false;
        } catch (Exception e) {
            log.debug("wmic video controller query failed: {}", e.getMessage());
            return false;
        }
    }

    private void populateVramFromWmic() {
        try {
            ProcessBuilder pb = new ProcessBuilder("cmd", "/c", "wmic path win32_VideoController get Name,AdapterRAM /format:list");
            pb.redirectErrorStream(true);
            Process process = pb.start();

            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            String line;
            String pendingName = null;

            while ((line = reader.readLine()) != null) {
                String trimmed = line.trim();
                if (trimmed.startsWith("Name=")) {
                    pendingName = trimmed.substring("Name=".length()).trim();
                } else if (trimmed.startsWith("AdapterRAM=") && pendingName != null) {
                    String upper = pendingName.toUpperCase();
                    if (upper.contains("AMD") || upper.contains("RADEON")) {
                        String bytes = trimmed.substring("AdapterRAM=".length()).trim();
                        if (!bytes.isBlank()) {
                            try {
                                long parsed = Long.parseLong(bytes);
                                if (parsed > 0) {
                                    vramBytes = parsed;
                                    process.waitFor();
                                    return;
                                }
                            } catch (NumberFormatException ignored) {
                                // Best-effort parse only.
                            }
                        }
                    }
                    pendingName = null;
                }
            }

            process.waitFor();
        } catch (Exception e) {
            log.debug("wmic AdapterRAM query failed: {}", e.getMessage());
        }

        if (vramBytes <= 0) {
            populateVramFromPowerShell();
        }
    }

    private void populateVramFromPowerShell() {
        try {
            String command = "Get-CimInstance Win32_VideoController | " +
                    "Where-Object { $_.Name -match 'AMD|Radeon' } | " +
                    "Select-Object -First 1 -ExpandProperty AdapterRAM";

            ProcessBuilder pb = new ProcessBuilder("powershell", "-NoProfile", "-Command", command);
            pb.redirectErrorStream(true);
            Process process = pb.start();

            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            String line;
            while ((line = reader.readLine()) != null) {
                String digits = line.replaceAll("[^0-9]", "");
                if (!digits.isEmpty()) {
                    try {
                        long parsed = Long.parseLong(digits);
                        if (parsed > 0) {
                            vramBytes = parsed;
                            break;
                        }
                    } catch (NumberFormatException ignored) {
                        // Best-effort parse only.
                    }
                }
            }

            process.waitFor();
        } catch (Exception e) {
            log.debug("PowerShell AdapterRAM query failed: {}", e.getMessage());
        }
    }

    private int inferAmdComputeUnitsFromModel(String model) {
        if (model == null) {
            return 0;
        }

        String upper = model.toUpperCase();
        if (upper.contains("RX 9070 XT")) {
            return 64;
        }
        if (upper.contains("RX 9070")) {
            return 56;
        }
        if (upper.contains("RX 7900 XTX")) {
            return 96;
        }
        if (upper.contains("RX 7900 XT")) {
            return 84;
        }
        if (upper.contains("RX 7900")) {
            return 84;
        }
        if (upper.contains("RX 6800 XT")) {
            return 72;
        }
        if (upper.contains("RX 6800")) {
            return 60;
        }
        return 0;
    }

    private long inferAmdVramFromModel(String model) {
        if (model == null) {
            return 0;
        }

        String upper = model.toUpperCase();
        if (upper.contains("RX 9070 XT")) {
            return 16L * 1024 * 1024 * 1024;
        }
        if (upper.contains("RX 9070")) {
            return 16L * 1024 * 1024 * 1024;
        }
        if (upper.contains("RX 7900 XTX")) {
            return 24L * 1024 * 1024 * 1024;
        }
        if (upper.contains("RX 7900 XT")) {
            return 20L * 1024 * 1024 * 1024;
        }
        if (upper.contains("RX 7900")) {
            return 20L * 1024 * 1024 * 1024;
        }
        if (upper.contains("RX 6800 XT")) {
            return 16L * 1024 * 1024 * 1024;
        }
        if (upper.contains("RX 6800")) {
            return 16L * 1024 * 1024 * 1024;
        }
        return 0;
    }

    /**
     * Detect NVIDIA GPU using nvidia-smi command
     */
    private boolean detectNvidiaGpu() {
        try {
            ProcessBuilder pb = new ProcessBuilder(
                "nvidia-smi",
                "--query-gpu=name,memory.total,compute_cap,driver_version",
                "--format=csv,noheader,nounits"
            );
            pb.redirectErrorStream(true);
            Process process = pb.start();
            
            BufferedReader reader = new BufferedReader(
                new InputStreamReader(process.getInputStream())
            );
            
            String line = reader.readLine();
            if (line != null && !line.isEmpty()) {
                // Parse nvidia-smi output
                String[] parts = line.split(",");
                if (parts.length >= 4) {
                    gpuModel = parts[0].trim();
                    vramBytes = Long.parseLong(parts[1].trim()) * 1024 * 1024; // MB to bytes
                    computeUnits = parseComputeUnits(parts[2].trim());
                    driverVersion = parts[3].trim();
                    
                    log.info("""
                        NVIDIA GPU Details:
                          Model: {}
                          VRAM: {} GB
                          Compute Capability: {}
                          Driver: {}
                        """, gpuModel, getVramGb(), parts[2].trim(), driverVersion);
                    
                    return true;
                }
            }
            
            process.waitFor();
        } catch (Exception e) {
            log.debug("nvidia-smi not found or failed: {}", e.getMessage());
        }
        return false;
    }
    
    /**
     * Detect AMD GPU using ROCm tools
     */
    private boolean detectAmdGpu() {
        try {
            // First try rocm-smi (preferred)
            ProcessBuilder pb = new ProcessBuilder(
                "rocm-smi",
                "--showproductname",
                "--showmeminfo",
                "vram"
            );
            pb.redirectErrorStream(true);
            Process process = pb.start();
            
            BufferedReader reader = new BufferedReader(
                new InputStreamReader(process.getInputStream())
            );
            
            StringBuilder output = new StringBuilder();
            String line;
            boolean foundGpu = false;
            
            while ((line = reader.readLine()) != null) {
                output.append(line).append("\n");
                
                if (line.contains("GPU[")) {
                    foundGpu = true;
                    
                    // Parse GPU model
                    if (line.contains("Card series:")) {
                        gpuModel = line.split(":")[1].trim();
                    }
                }
                
                // Parse VRAM
                if (line.contains("VRAM")) {
                    try {
                        String vramStr = line.replaceAll("[^0-9]", "");
                        if (!vramStr.isEmpty()) {
                            vramBytes = Long.parseLong(vramStr) * 1024 * 1024;
                        }
                    } catch (NumberFormatException e) {
                        // Parse failed, use default
                    }
                }
            }
            
            process.waitFor();
            
            if (foundGpu) {
                log.info("""
                    AMD GPU Details:
                      Model: {}
                      VRAM: {} GB
                      ROCm Output: {}
                    """, gpuModel, getVramGb(), output.toString().trim());
                return true;
            }
            
        } catch (Exception e) {
            log.debug("rocm-smi not found, trying alternative detection");
        }
        
        // Alternative: Try to detect AMD GPU via lspci
        return detectAmdGpuViaLspci();
    }
    
    /**
     * Fallback AMD GPU detection using lspci
     */
    private boolean detectAmdGpuViaLspci() {
        try {
            ProcessBuilder pb = new ProcessBuilder(
                "lspci",
                "-v",
                "-d", "1002:"  // AMD vendor ID
            );
            pb.redirectErrorStream(true);
            Process process = pb.start();
            
            BufferedReader reader = new BufferedReader(
                new InputStreamReader(process.getInputStream())
            );
            
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.contains("VGA") || line.contains("Display")) {
                    // Parse GPU model from lspci output
                    String[] parts = line.split(":");
                    if (parts.length >= 3) {
                        gpuModel = parts[2].trim();
                        
                        // Try to determine VRAM from lspci
                        if (line.contains("RX 9070")) {
                            vramBytes = 16L * 1024 * 1024 * 1024; // 16 GB
                        } else if (line.contains("RX 7900")) {
                            vramBytes = 20L * 1024 * 1024 * 1024; // 20 GB
                        } else if (line.contains("RX 6800")) {
                            vramBytes = 16L * 1024 * 1024 * 1024; // 16 GB
                        }
                        
                        log.info("AMD GPU detected via lspci: {}", gpuModel);
                        return true;
                    }
                }
            }
            
            process.waitFor();
        } catch (Exception e) {
            log.debug("lspci detection failed: {}", e.getMessage());
        }
        return false;
    }
    
    /**
     * Detect any GPU via OpenCL platform query
     */
    private boolean detectOpenCL() {
        try {
            // JOCL detection
            Class.forName("org.jocl.CL");
            log.debug("OpenCL libraries available");
            return true;
        } catch (ClassNotFoundException e) {
            log.debug("OpenCL not available");
        }
        return false;
    }
    
    /**
     * Get CPU information for fallback
     */
    private void detectCpuInfo() {
        CentralProcessor processor = hardware.getProcessor();
        computeUnits = processor.getLogicalProcessorCount();
        gpuModel = String.format("CPU: %s", processor.getProcessorIdentifier().getName());
        
        log.info("""
            CPU Information:
              Model: {}
              Cores: {} Physical, {} Logical
            """,
            processor.getProcessorIdentifier().getName(),
            processor.getPhysicalProcessorCount(),
            processor.getLogicalProcessorCount()
        );
    }
    
    /**
     * Parse NVIDIA compute capability
     */
    private int parseComputeUnits(String computeCap) {
        try {
            // Compute capability like "8.9" -> 89 SM units
            return (int) (Double.parseDouble(computeCap) * 10);
        } catch (NumberFormatException e) {
            return 0;
        }
    }
    
    /**
     * Get VRAM in GB for display
     */
    public double getVramGb() {
        return vramBytes / (1024.0 * 1024.0 * 1024.0);
    }
    
    /**
     * Check if any GPU is available
     */
    public boolean isGpuAvailable() {
        return cudaAvailable || rocmAvailable || openclAvailable;
    }
    
    /**
     * Get available system memory for training
     */
    public long getAvailableSystemMemory() {
        return hardware.getMemory().getAvailable();
    }
    
    /**
     * Get GPU information as a formatted string
     */
    public String getGpuInfo() {
        return String.format("""
            GPU Type: %s
            Model: %s
            VRAM: %.2f GB
            Compute Units: %d
            Driver: %s
            CUDA Available: %b
            ROCm Available: %b
            OpenCL Available: %b
            """,
            gpuType,
            gpuModel,
            getVramGb(),
            computeUnits,
            driverVersion,
            cudaAvailable,
            rocmAvailable,
            openclAvailable
        );
    }
}
