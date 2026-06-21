package com.aigrama.llm_training;

import com.aigrama.llm_training.training.TrainingMetrics;
import com.aigrama.llm_training.training.TrainingService;
import org.springframework.boot.SpringApplication;
import org.springframework.context.ConfigurableApplicationContext;

import java.time.Duration;

public class Trainer {

    public static void main(String[] args) throws Exception {
        System.out.println("Starting LLM training runner...");

        ConfigurableApplicationContext context = SpringApplication.run(LlmTrainingApplication.class, args);
        try {
            TrainingService trainingService = context.getBean(TrainingService.class);
            TrainingMetrics metrics = trainingService.train().get();

            System.out.println("Training finished.");
            System.out.println("Status: " + metrics.getStatus());
            System.out.println("Dataset size: " + metrics.getDatasetSize());
            System.out.println("Current epoch: " + metrics.getCurrentEpoch());
            System.out.println("Current loss: " + metrics.getCurrentLoss());

            Duration duration = metrics.getDuration();
            System.out.println("Duration: " + (duration != null ? duration : "N/A"));

            if (metrics.getErrorMessage() != null && !metrics.getErrorMessage().isBlank()) {
                System.out.println("Error: " + metrics.getErrorMessage());
            }
        } finally {
            SpringApplication.exit(context);
            context.close();
        }
    }
}
