# LLM Training App - Step-by-Step Guide

This guide walks you through running the app and starting model training using the built-in REST APIs.

## 1) Prerequisites

- Windows with PowerShell
- JDK 21 installed
- Network access for first-time DJL native downloads
- Project path: `C:\Pan-temp\llm-training`

## 2) Recommended training mode

The project can detect AMD/NVIDIA GPUs for runtime, but **training on the AMD DirectML/ONNX runtime path may return `Not supported!`**.

For reliable training flow, run training in CPU/PyTorch mode:

```powershell
$env:ACTIVE_GPU = "CPU"
```

(Keep this in the same terminal session before running the app.)

## 3) Configure training values

Edit `src/main/resources/application.yaml` under `training`:

- `training.model.*` (vocab size, layers, sequence length)
- `training.hyperparameters.batch-size` (must be numeric)
- `training.hyperparameters.epochs`
- `training.hyperparameters.learning-rate`
- `training.data.file-path` (default: `classpath:training_data.txt`)
- `training.data.generate-sample` (`true` to auto-generate sample data)

## 4) Build and start the app

```powershell
Set-Location "C:\Pan-temp\llm-training"
.\gradlew.bat compileJava --no-daemon
.\gradlew.bat bootRun --no-daemon
```

If port `8080` is in use:

```powershell
.\gradlew.bat bootRun --no-daemon --args=--server.port=8081
```

## 5) Verify app and GPU/training endpoints

Using default port `8080`:

```powershell
Invoke-RestMethod -Uri "http://localhost:8080/api/training/health" -Method Get
Invoke-RestMethod -Uri "http://localhost:8080/api/gpu/info" -Method Get | ConvertTo-Json -Depth 4
```

## 6) Start training

```powershell
Invoke-RestMethod -Uri "http://localhost:8080/api/training/start" -Method Post
```

Expected response:

- `Training started. Check status at /api/training/status`

## 7) Monitor training status

```powershell
Invoke-RestMethod -Uri "http://localhost:8080/api/training/status" -Method Get | ConvertTo-Json -Depth 4
```

Check fields:

- `status` (`RUNNING`, `COMPLETED`, or `FAILED`)
- `datasetSize`
- `currentEpoch`
- `currentLoss`
- `errorMessage` (if failed)

## 8) Optional: Benchmark backend

```powershell
Invoke-RestMethod -Uri "http://localhost:8080/api/gpu/benchmark" -Method Post | ConvertTo-Json -Depth 6
```

If device runtime ops are unsupported, the endpoint now returns a fallback Java benchmark result.

## 9) Troubleshooting

### A) `Not supported!` during training

- Cause: backend/runtime does not support model training path (common on AMD DirectML path).
- Fix: force CPU mode and restart app in same terminal:

```powershell
$env:ACTIVE_GPU = "CPU"
.\gradlew.bat bootRun --no-daemon
```

### B) Bean conflict `trainingDevice` already defined

- Ensure only one `trainingDevice` bean exists (already fixed in current workspace by removing duplicate from `TrainingConfiguration`).

### C) Config binding error for batch size

- `training.hyperparameters.batch-size` must be an integer (not `auto`).

## 10) Useful API summary

- `GET /api/training/health`
- `POST /api/training/start`
- `GET /api/training/status`
- `GET /api/gpu/info`
- `GET /api/gpu/health`
- `POST /api/gpu/benchmark`

## 11) Quick run sequence (copy/paste)

```powershell
Set-Location "C:\Pan-temp\llm-training"
$env:ACTIVE_GPU = "CPU"
.\gradlew.bat compileJava --no-daemon
.\gradlew.bat bootRun --no-daemon
```

Then in another terminal:

```powershell
Invoke-RestMethod -Uri "http://localhost:8080/api/training/start" -Method Post
Invoke-RestMethod -Uri "http://localhost:8080/api/training/status" -Method Get | ConvertTo-Json -Depth 4
```
