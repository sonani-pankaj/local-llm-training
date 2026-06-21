package com.aigrama.llm_training.training;
import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;

import com.aigrama.llm_training.config.TrainingConfiguration;
import com.aigrama.llm_training.data.DataGenerator;
import com.aigrama.llm_training.data.TextDataset;
import com.aigrama.llm_training.model.MiniGPT;
import com.aigrama.llm_training.tokenizer.SimpleTokenizer;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;

import java.nio.file.Paths;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicReference;

@Slf4j
@Service
public class TrainingService {

    private final TrainingConfiguration config;
    private final SimpleTokenizer tokenizer;
    private final MiniGPT model;
    private final Device device;
    private final AtomicReference<TrainingMetrics> currentMetrics;

    public TrainingService(TrainingConfiguration config,
                           SimpleTokenizer tokenizer,
                           MiniGPT model,
                           Device device) {
        this.config = config;
        this.tokenizer = tokenizer;
        this.model = model;
        this.device = device;
        this.currentMetrics = new AtomicReference<>(new TrainingMetrics());
    }

    /**
     * Execute training asynchronously using Java 25 virtual threads
     */
    @Async
    public CompletableFuture<TrainingMetrics> train() {
        var metrics = new TrainingMetrics();
        metrics.start();

        try {
            // Step 1: Generate training data if needed
            if (config.getData().isGenerateSample()) {
                log.info("Generating sample training data...");
                DataGenerator.createSampleData();
            }

            // Step 2: Build tokenizer vocabulary
            log.info("Building vocabulary...");
            String trainingText = DataGenerator.loadTrainingData(config.getData().getFilePath());
            tokenizer.buildVocabulary(trainingText);

            // Step 3: Prepare dataset
            log.info("Preparing dataset...");
            var dataset = new TextDataset(
                    tokenizer,
                    config.getModel().getMaxSequenceLength(),
                    config.getHyperparameters().getBatchSize(),
                    0.1
            );
            dataset.loadFromFile(config.getData().getFilePath());
            metrics.setDatasetSize(dataset.getTrainingSize());

            // Step 4: Create and configure model
            log.info("Configuring model for training...");
            try (var trainingModel = Model.newInstance("mini-gpt")) {
                trainingModel.setBlock(model);

                // Setup training configuration
                var trainingConfig = new DefaultTrainingConfig(Loss.softmaxCrossEntropyLoss());

                var optimizer = Optimizer.adam()
                        .optLearningRateTracker(
                                Tracker.fixed((float) config.getHyperparameters().getLearningRate())
                        )
                        .build();
                trainingConfig.optOptimizer(optimizer);
                trainingConfig.optDevices(new Device[]{device});

                // Training loop
                try (var trainer = trainingModel.newTrainer(trainingConfig)) {
                    trainer.initialize(new Shape(
                            config.getHyperparameters().getBatchSize(),
                            config.getModel().getMaxSequenceLength()
                    ));

                    log.info("""
                        Starting training:
                        - Dataset size: {}
                        - Batch size: {}
                        - Epochs: {}
                        - Device: {}
                        """,
                            dataset.getTrainingSize(),
                            config.getHyperparameters().getBatchSize(),
                            config.getHyperparameters().getEpochs(),
                            device);

                    // Execute training epochs
                    for (int epoch = 0; epoch < config.getHyperparameters().getEpochs(); epoch++) {
                        trainEpoch(dataset, trainer, metrics, epoch);
                    }

                    // Save model
                    log.info("Saving trained model...");
                    trainingModel.save(Paths.get("model"), "mini-gpt");
                    metrics.markComplete();
                }
            }

        } catch (Exception e) {
            log.error("Training failed", e);
            metrics.markFailed(e.getMessage());
        }

        currentMetrics.set(metrics);
        return CompletableFuture.completedFuture(metrics);
    }

    private void trainEpoch(TextDataset dataset, Trainer trainer,
                            TrainingMetrics metrics, int epoch) {
        // Training implementation for each epoch
        log.info("Starting epoch {}/{}", epoch + 1, config.getHyperparameters().getEpochs());
        // ... training logic
    }

    public TrainingMetrics getCurrentMetrics() {
        return currentMetrics.get();
    }
}