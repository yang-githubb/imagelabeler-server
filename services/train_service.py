import sys
import json
import subprocess
import threading
import os
import signal
import csv
from core.config import BASE_DIR, RUNS_DIR, CONFIG_FILE, RESULTS_FILE 
from core import state

RUNS_DIR.mkdir(exist_ok=True)

def start_train_service(cfg):
    if state.train_process is not None:
        raise Exception("Training already running")

    run_dir = RUNS_DIR / cfg.runName
    if run_dir.exists():
        raise Exception("Run already exists")

    run_dir.mkdir(parents=True)

    with open(run_dir / CONFIG_FILE, "w") as f:
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

    state.train_process = subprocess.Popen(
        [
            sys.executable,
            str(BASE_DIR / "train.py"),
            "--data", str(
                (BASE_DIR / "datasets" / cfg.station / cfg.process / "dataset.yaml")
            ),
            "--model", cfg.model,
            "--epochs", str(cfg.epochs),
            "--imgsz", str(cfg.imgsz),
            "--batch", str(cfg.batch),
            "--name", cfg.runName,
            "--project", str(RUNS_DIR)
        ],
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
    )

    state.active_run = cfg.runName

    def _on_exit():
        state.train_process = None
        state.active_run = None

    threading.Thread(
        target=lambda: (state.train_process.wait(), _on_exit()),
        daemon=True
    ).start()

    return {"status": "started", "runName": cfg.runName}


def train_progress_service():
    if state.active_run is None:
        return {"status": "idle"}

    run_dir = RUNS_DIR / state.active_run
    csv_path = run_dir / RESULTS_FILE
    cfg_path = run_dir / CONFIG_FILE

    if not csv_path.exists() or not cfg_path.exists():
        return {"status": "starting", "runName": state.active_run}

    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    total_epochs = int(cfg["epochs"])

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return {"status": "starting", "runName": state.active_run}

    last = rows[-1]
    epoch = int(float(last["epoch"]))
    progress = min(100, round((epoch / total_epochs) * 100))

    return {
        "status": "running",
        "runName": state.active_run,
        "epoch": epoch,
        "totalEpochs": total_epochs,
        "progress": progress
    }


def train_metrics_service(run=None):
    run_name = run or state.active_run

    if run_name is None:
        return []

    run_dir = RUNS_DIR / run_name
    csv_path = run_dir / RESULTS_FILE

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


def stop_train_service():
    if state.train_process is None:
        return {"status": "idle"}

    try:
        os.kill(state.train_process.pid, signal.CTRL_BREAK_EVENT)
    except Exception as e:
        print("Stop training failed:", e)

    return {
        "status": "stopping",
        "runName": state.active_run
    }