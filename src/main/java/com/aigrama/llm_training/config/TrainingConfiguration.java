package com.aigrama.llm_training.config;

import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.Positive;
import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Configuration;
import org.springframework.validation.annotation.Validated;

@Configuration
@ConfigurationProperties(prefix = "training")
@Validated
@Data
public class TrainingConfiguration {

    private ModelConfig model = new ModelConfig();
    private HyperparametersConfig hyperparameters = new HyperparametersConfig();
    private DataConfig data = new DataConfig();

    @Data
    public static class ModelConfig {
        @Min(100)
        private int vocabSize = 1000;

        @Min(64)
        private int embedDim = 256;

        @Positive
        private int numHeads = 8;

        @Positive
        private int numLayers = 4;

        @Min(8)
        private int maxSequenceLength = 32;
    }

    @Data
    public static class HyperparametersConfig {
        @Positive
        private int batchSize = 8;

        @Min(1)
        private int epochs = 5;

        @Positive
        private double learningRate = 0.0005;

        @Min(0)
        private int warmupSteps = 100;
    }

    @Data
    public static class DataConfig {
        private String filePath = "classpath:training_data.txt";
        private boolean generateSample = true;
    }

}