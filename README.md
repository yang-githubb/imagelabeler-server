# 🧠 Training Server (YOLO + FastAPI)
## Overview
This project is a training server for YOLO-based vision models, built with FastAPI.

It provides APIs for:

- Dataset management (upload, browse, classes)
- Training control (start / stop / progress)
- Experiment tracking (list runs, download weights)
- Model inspection (extract class names from .pt)

The system is designed for industrial use cases, with a clean separation between:
```
API layer → Service layer → Training process (CLI)
```

## 🏗️ Architecture
```
FastAPI (app.py)
│
├── routes/           → API endpoints (HTTP layer)
├── services/         → business logic
├── schemas/          → request validation (Pydantic)
├── core/
│    ├── config.py    → global paths & constants
│    └── state.py     → runtime training state
│
├── datasets/         → training data (images/labels)
├── runs/             → training outputs
├── train.py          → YOLO training entry point
```

## 🚀 How It Works

### Training Flow
```
Frontend / API call
      ↓
/train/start
      ↓
train_service.py
      ↓
subprocess → train.py
      ↓
Ultralytics YOLO training
      ↓
outputs → /runs/<run_name>/
```

## 📁 Folder Structure
```
training_server/
│
├── app.py
├── train.py
├── README.md
├── requirements.in
│
├── core/
│   ├── config.py
│   └── state.py
│
├── schemas/
│   └── train_schema.py
│
├── routes/
│   ├── train_routes.py
│   ├── dataset_routes.py
│   ├── experiment_routes.py
│   └── model_routes.py
│
├── services/
│   ├── train_service.py
│   ├── dataset_service.py
│   └── experiment_service.py
│
├── datasets/
│   └── <station>/<process>/
│       ├── dataset.yaml
│       ├── images/
│       ├── labels/
│       └── validation/
│
└── runs/
    └── <run_name>/
        ├── run_config.json
        ├── results.csv
        └── weights/best.pt
```

## 🔧 Installation

### 1. Create environment

```python -m venv venvvenv\Scripts\activate   # Windows```

### 2. Install dependencies
```pip install -r requirements.in```

## ▶️ Running the Server
```
uvicorn app:app --port 8005 --reload
Server runs at:
http://127.0.0.1:8005
```

## 📡 API Endpoints

### 🧪 Training

| Method | Endpoint           | Description     |
|--------|-------------------|-----------------|
| POST   | /train/start      | Start training  |
| GET    | /train/progress   | Get progress    |
| GET    | /train/metrics    | Get metrics     |
| POST   | /train/stop       | Stop training   |

### Example (start)
```
POST /train/start
{  
    "station": "station_01",
    "process": "final_inspection",  
    "model": "yolo26n.pt",  
    "epochs": 50,  
    "imgsz": 640, 
    "batch": 8,  
    "runName": "run_001"
}
```
---
### 📦 Dataset
| Method | Endpoint           | Description     |
|--------|-------------------|-----------------|
| POST   | /datasets/upload      | Upload image + label  |
| GET    | /datasets/classes   | Get dataset classes    |
| GET    | /datasets/images    | List images     |

---

### 🧪 Experiments

| Method | Endpoint           | Description     |
|--------|-------------------|-----------------|
| GET    | /experiments   | List runs    |
| GET    | /experiments/{run}/weights    | Download model     |

---

### 🧠 Model

| Method | Endpoint           | Description     |
|--------|-------------------|-----------------|
| POST   | /model/classes     | Extract class names  |

---

## 📂 Dataset Structure
```
Each dataset must follow YOLO format:
datasets/
└── station_01/
    └── final_inspection/
        ├── dataset.yaml
        ├── images/
        ├── labels/
        └── validation/
```

### Dataset.yaml example
```
train: images
val: validation/images
names:  
    0: good  
    1: defect
```

## 📊 Training Outputs

Each run generates:
```
runs/<run_name>/
├── run_config.json
├── results.csv
└── weights/
    └── best.pt
```