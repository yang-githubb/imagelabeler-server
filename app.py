from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from subprocess import Popen
from pathlib import Path
import json
import sys
import csv
import psutil
import subprocess
import os
import signal
from fastapi import UploadFile, File, Form, Query
from shutil import move
import tempfile
import yaml
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# ======================================================
# PATHS
# ======================================================

BASE_DIR = Path(__file__).parent
RUNS_DIR = BASE_DIR / "runs"
RUNS_DIR.mkdir(exist_ok=True)
DATASET_ROOT = BASE_DIR / "datasets"

# ======================================================
# TRAINING STATE (SINGLE ACTIVE RUN)
# ======================================================

train_process = None
active_run = None

# ======================================================
# REQUEST MODEL
# ======================================================

class TrainConfig(BaseModel):
    station: str
    process: str
    model: str
    epochs: int
    imgsz: int
    batch: int
    runName: str

# ======================================================
# TRAINING — START
# ======================================================

@app.post("/train/start")
def start_train(cfg: TrainConfig):
    global train_process, active_run

    if train_process is not None:
        raise HTTPException(status_code=400, detail="Training already running")

    run_dir = RUNS_DIR / cfg.runName
    if run_dir.exists():
        raise HTTPException(status_code=400, detail="Run already exists")

    run_dir.mkdir(parents=True)

    with open(run_dir / "run_config.json", "w") as f:
        json.dump(
            {
                **cfg.dict(),
                "startedAt": __import__("datetime")
                    .datetime.utcnow()
                    .isoformat() + "Z"
            },
            f,
            indent=2
        )


    train_process = subprocess.Popen(
        [
            sys.executable,
            str(BASE_DIR / "train.py"),
            "--data", f"datasets/{cfg.station}/{cfg.process}/dataset.yaml",
            "--model", cfg.model,
            "--epochs", str(cfg.epochs),
            "--imgsz", str(cfg.imgsz),
            "--batch", str(cfg.batch),
            "--name", cfg.runName,
            "--project", str(RUNS_DIR)
        ],
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
    )

    active_run = cfg.runName

    def _on_train_exit():
        global train_process, active_run
        train_process = None
        active_run = None

    import threading
    threading.Thread(
        target=lambda: (train_process.wait(), _on_train_exit()),
        daemon=True
    ).start()

    return {"status": "started", "runName": cfg.runName}

# ======================================================
# TRAINING — PROGRESS
# ======================================================

@app.get("/train/progress")
def train_progress():
    if active_run is None:
        return {"status": "idle"}

    run_dir = RUNS_DIR / active_run
    csv_path = run_dir / "results.csv"
    cfg_path = run_dir / "run_config.json"

    if not csv_path.exists() or not cfg_path.exists():
        return {"status": "starting", "runName": active_run}

    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    total_epochs = int(cfg["epochs"])

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return {"status": "starting", "runName": active_run}

    last = rows[-1]
    epoch = int(float(last["epoch"]))
    progress = min(100, round((epoch / total_epochs) * 100))

    return {
        "status": "running",
        "runName": active_run,
        "epoch": epoch,
        "totalEpochs": total_epochs,
        "progress": progress
    }

# ======================================================
# TRAINING — METRICS (LOSS + MAP50)
# ======================================================

@app.get("/train/metrics")
def train_metrics(run: str | None = None):
    run_name = run or active_run

    if run_name is None:
        return []

    run_dir = RUNS_DIR / run_name
    csv_path = run_dir / "results.csv"

    if not csv_path.exists():
        return []

    metrics = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                metrics.append({
                    "epoch": int(float(row["epoch"])),
                    "loss": float(row["train/box_loss"]),
                    "map50": float(row["metrics/mAP50(B)"])
                })
            except Exception:
                continue

    return metrics

# ======================================================
# EXPERIMENTS — LIST (WITH METRIC SUMMARY)
# ======================================================

@app.get("/experiments")
def list_experiments():
    experiments = []

    for d in RUNS_DIR.iterdir():
        if not d.is_dir():
            continue

        cfg_path = d / "run_config.json"
        results_path = d / "results.csv"
        weights_path = d / "weights" / "best.pt"

        if not cfg_path.exists():
            continue

        with open(cfg_path, "r") as f:
            cfg = json.load(f)

        run_name = d.name

        # ==================================================
        # STATUS LOGIC 
        # ==================================================
        if active_run == run_name:
            status = "Running"
        elif weights_path.exists():
            status = "Completed"
        elif results_path.exists():
            status = "Stopped"
        else:
            status = "Failed"

        # ==================================================
        # METRICS SUMMARY
        # ==================================================
        final_loss = None
        best_map50 = None

        if results_path.exists():
            with open(results_path, newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

                if rows:
                    last = rows[-1]
                    final_loss = float(last.get("train/box_loss", 0))

                    map_values = [
                        float(r["metrics/mAP50(B)"])
                        for r in rows
                        if r.get("metrics/mAP50(B)")
                    ]
                    best_map50 = max(map_values) if map_values else None

        experiments.append({
            "name": run_name,
            "status": status,

            # config
            "model": cfg.get("model"),
            "dataset": f"{cfg.get('station')}/{cfg.get('process')}",
            "batch": cfg.get("batch"),
            "epochs": cfg.get("epochs"),
            "imgsz": cfg.get("imgsz"),
            "startedAt": cfg.get("startedAt"),

            # summary metrics
            "finalLoss": final_loss,
            "bestMap50": best_map50,

            # actions
            "hasWeights": weights_path.exists(),
            "hasMetrics": results_path.exists()
        })

    experiments.sort(
        key=lambda x: x.get("startedAt") or "",
        reverse=True
    )

    return experiments

# ======================================================
# EXPERIMENTS — DOWNLOAD BEST WEIGHTS (CANONICAL)
# ======================================================

@app.get("/experiments/{run}/weights")
def download_experiment_weights(run: str):
    run_dir = RUNS_DIR / run
    weights_path = run_dir / "weights" / "best.pt"

    if not weights_path.exists():
        raise HTTPException(status_code=404, detail="Weights not found")

    return FileResponse(
        weights_path,
        filename=f"{run}_best.pt"
    )

# ======================================================
# TRAINING — STOP (GRACEFUL)
# ======================================================

@app.post("/train/stop")
def stop_train():
    global train_process, active_run

    if train_process is None:
        return {"status": "idle"}

    try:
        # ✅ Send CTRL+BREAK to entire process group
        os.kill(train_process.pid, signal.CTRL_BREAK_EVENT)
    except Exception as e:
        print("Stop training failed:", e)

    return {
        "status": "stopping",
        "runName": active_run
    }

# ======================================================
# DATASET - UPLOAD
# ======================================================

@app.post("/dataset/upload")
async def dataset_upload(
    station: str = Form(...),
    process: str = Form(...),
    image: UploadFile = File(...),
    label: UploadFile = File(...),
    thumb: UploadFile = File(...)
):
    images_dir = DATASET_ROOT / station / process / "images"
    labels_dir = DATASET_ROOT / station / process / "labels"
    thumbs_dir = DATASET_ROOT / station / process / "images" / "thumbs"

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    thumbs_dir.mkdir(parents=True, exist_ok=True) 

    image_path = images_dir / image.filename
    label_path = labels_dir / label.filename
    thumb_path = thumbs_dir / thumb.filename  

    with tempfile.NamedTemporaryFile(delete=False) as tmp_img:
        tmp_img.write(await image.read())
        tmp_img_path = tmp_img.name

    with tempfile.NamedTemporaryFile(delete=False) as tmp_lbl:
        tmp_lbl.write(await label.read())
        tmp_lbl_path = tmp_lbl.name

    with tempfile.NamedTemporaryFile(delete=False) as tmp_th:
        tmp_th.write(await thumb.read())
        tmp_th_path = tmp_th.name             


    move(tmp_img_path, image_path)
    move(tmp_lbl_path, label_path)
    move(tmp_th_path, thumb_path)

    return {"status": "ok"}

# ======================================================
# DATASET - CLASSES
# ======================================================

@app.get("/dataset/classes")
def get_dataset_classes(station: str, process: str):
    dataset_yaml = (
        BASE_DIR
        / "datasets"
        / station
        / process
        / "dataset.yaml"
    )

    if not dataset_yaml.exists():
        raise HTTPException(
            status_code=404,
            detail="dataset.yaml not found"
        )

    with open(dataset_yaml, "r") as f:
        data = yaml.safe_load(f)

    names = data.get("names")
    if not isinstance(names, dict):
        raise HTTPException(
            status_code=400,
            detail="Invalid dataset.yaml format"
        )

    # Return class list in index order
    return {
        "classes": [
            names[k] for k in sorted(names.keys())
        ]
    }

# ======================================================
# DATASET - UPLOAD
# ======================================================

@app.get("/datasets/images")
def list_dataset_images(
    station: str = Query(...),
    process: str = Query(...),
    page: int = Query(1, ge=1),
    limit: int = Query(24, ge=1)
):
    images_dir = (
        BASE_DIR
        / "datasets"
        / station
        / process
        / "images"
    )

    if not images_dir.exists():
        return {"total": 0, "images": []}

    files = sorted(
        f.name for f in images_dir.iterdir()
        if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
    )

    total = len(files)
    start = (page - 1) * limit
    slice_ = files[start:start + limit]

    return {
        "total": total,
        "images": slice_
    }

app.mount(
    "/datasets",
    StaticFiles(directory=DATASET_ROOT),
    name="datasets"
)
