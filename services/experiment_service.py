from pathlib import Path
import json
import csv
from fastapi import HTTPException
from fastapi.responses import FileResponse
from core import state
from core.config import BASE_DIR, RUNS_DIR, CONFIG_FILE, RESULTS_FILE, WEIGHTS_DIR, BEST_WEIGHTS

def list_experiments_service():
    experiments = []

    for d in RUNS_DIR.iterdir():
        if not d.is_dir():
            continue

        cfg_path = d / CONFIG_FILE
        results_path = d / RESULTS_FILE
        weights_path = d / WEIGHTS_DIR / BEST_WEIGHTS

        if not cfg_path.exists():
            continue

        with open(cfg_path, "r") as f:
            cfg = json.load(f)

        epoch = 0
        total_epochs = int(cfg.get("epochs", 0))

        if results_path.exists():
            with open(results_path, newline="") as f:
                rows = list(csv.DictReader(f))

            if rows:
                last = rows[-1]
                epoch = int(float(last["epoch"]))

        # ----- STATUS LOGIC -----
        if d.name == state.active_run:
            status = "Running"

        elif results_path.exists():
            if epoch >= total_epochs:
                status = "Completed"
            else:
                status = "Stopped"

        else:
            status = "Failed"

        experiments.append({
            "name": d.name,
            "status": status,
            "model": cfg.get("model"),
            "dataset": f"{cfg.get('station')}/{cfg.get('process')}",
            "batch": cfg.get("batch"),
            "epochs": cfg.get("epochs"),
            "imgsz": cfg.get("imgsz"),
            "startedAt": cfg.get("startedAt"),
            "epoch": epoch,
            "totalEpochs": total_epochs,
            "hasWeights": weights_path.exists(),
            "hasMetrics": results_path.exists()
        })

    return experiments


def download_weights_service(run):
    path = RUNS_DIR / run / WEIGHTS_DIR / BEST_WEIGHTS

    if not path.exists():
        raise HTTPException(404, "Weights not found")

    return FileResponse(path, filename=f"{run}_best.pt")