# Vision Inspection System –  Server
A backend training service for the Vision Inspection System, responsible for YOLO model training, experiment tracking, and metrics generation. Built with Python FastAPI and Ultralytics YOLO.

## Features

### 🤖 Model Training

 - Ultralytics YOLO training via CLI
 - Single active training run enforcement
 - Configurable epochs, batch size, and image size
 - Deterministic run naming and output structure

### 📊 Experiment Tracking

 - Training progress reporting (epoch, percentage)
 - Loss and mAP metric extraction from YOLO results
 - Experiment history listing
 - Best‑weight (best.pt) export

### Architecture

```
PC / Control Plane (Node.js)
    ↓ HTTP (REST)
Training Server (FastAPI – Port 8002)
    ↓
YOLO CLI (Ultralytics)
    ↓
File System (datasets/, runs/)
```

### Service Responsibilities

**Training Server**: YOLO training execution, experiment lifecycle, metrics parsing, model export
**PC / Control Plane**: UI, dataset selection logic, station/process injection, orchestration

## Prerequisites

### System Requirements

 - **Windows 10/11** (tested on Window 11)
 - **Python 3.12**
 - **CUDA‑capable GPU** 

### Service Compatibility
This repository contains the Training Server component.
Compatible PC / Control Plane

PC Repository: https://github.com/yang-githubb/ImageLabeler
Compatible Versions: PC ≥ pc-v0.1.0

The PC communicates with this server using REST APIs and assumes the following
endpoints are available:

 - POST /train/start
 - POST /train/stop
 - GET /train/progress
 - GET /train/metrics
 - GET /experiments

### Usage
Starting the Training Server
```
uvicorn app:app --host 0.0.0.0 --port 8002
```
The server listens on port 8002 and waits for commands from the PC.

### Runtime Behavior

Only one training run can be active at a time
Training artifacts are written to runs/
Datasets and runs are runtime data and are not version‑controlled

### API Reference

 - POST /train/start – Start a training run
 - POST /train/stop – Stop the active training run
 - GET /train/progress – Get training progress
 - GET /train/metrics – Get training metrics
 - GET /experiments – List all experiments