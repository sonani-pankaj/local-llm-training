package com.aigrama.llm_training.controller;


import com.aigrama.llm_training.training.TrainingMetrics;
import com.aigrama.llm_training.training.TrainingService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.concurrent.CompletableFuture;

@Slf4j
@RestController
@RequestMapping("/api/training")
public class TrainingController {

    private final TrainingService trainingService;

    public TrainingController(TrainingService trainingService) {
        this.trainingService = trainingService;
    }

    @PostMapping("/start")
    public ResponseEntity<String> startTraining() {
        log.info("Received request to start training");
        CompletableFuture<TrainingMetrics> future = trainingService.train();

        return ResponseEntity.accepted()
                .body("Training started. Check status at /api/training/status");
    }

    @GetMapping("/status")
    public ResponseEntity<TrainingMetrics> getStatus() {
        return ResponseEntity.ok(trainingService.getCurrentMetrics());
    }

    @GetMapping("/health")
    public ResponseEntity<String> health() {
        return ResponseEntity.ok("LLM Training Service is running");
    }
}