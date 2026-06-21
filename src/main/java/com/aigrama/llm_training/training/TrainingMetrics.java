package com.aigrama.llm_training.training;

import lombok.Data;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;

@Data
public class TrainingMetrics {
    private Instant startTime;
    private Instant endTime;
    private Duration duration;
    private int datasetSize;
    private int currentEpoch;
    private double currentLoss;
    private List<Double> lossHistory = new ArrayList<>();
    private String status = "IDLE";
    private String errorMessage;

    public void start() {
        this.startTime = Instant.now();
        this.status = "RUNNING";
    }

    public void markComplete() {
        this.endTime = Instant.now();
        this.duration = Duration.between(startTime, endTime);
        this.status = "COMPLETED";
    }

    public void markFailed(String error) {
        this.endTime = Instant.now();
        this.duration = Duration.between(startTime, endTime);
        this.status = "FAILED";
        this.errorMessage = error;
    }

    public void addLoss(double loss) {
        this.currentLoss = loss;
        this.lossHistory.add(loss);
    }
}