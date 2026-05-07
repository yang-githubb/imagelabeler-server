import sys
import json
import subprocess
import threading
import os
import signal
import csv
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
RUNS_DIR = BASE_DIR / "runs"
RUNS_DIR.mkdir(exist_ok=True)

train_process = None
active_run = None


def start_train_service(cfg):
    global train_process, active_run

    if train_process is not None:
        raise Exception("Training already running")

    run_dir = RUNS_DIR / cfg.runName
    if run_dir.exists():
        raise Exception("Run already exists")

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

    def _on_exit():
        global train_process, active_run
        train_process = None
        active_run = None

    threading.Thread(
        target=lambda: (train_process.wait(), _on_exit()),
        daemon=True
    ).start()

    return {"status": "started", "runName": cfg.runName}

def train_progress_service():
    global active_run

    if active_run is None:
        return {"status": "idle"}

    run_dir = RUNS_DIR / active_run
    csv_path = run_dir / "results.csv"
    cfg_path = run_dir / "run_config.json"

    if not csv_path.exists() or not cfg_path.exists():
        return {"status": "starting", "runName": active_run}

    import json, csv

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

def train_metrics_service(run=None):
    run_name = run or active_run

    if run_name is None:
        return []

    run_dir = RUNS_DIR / run_name
    csv_path = run_dir / "results.csv"

    if not csv_path.exists():
        return []

    import csv

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
    global train_process, active_run

    if train_process is None:
        return {"status": "idle"}

    try:
        import os, signal
        os.kill(train_process.pid, signal.CTRL_BREAK_EVENT)
    except Exception as e:
        print("Stop training failed:", e)

    return {
        "status": "stopping",
        "runName": active_run
    }