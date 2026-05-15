from pathlib import Path
import json
import csv
from fastapi import HTTPException
from fastapi.responses import FileResponse
from core import state
from core.config import RUNS_DIR, CONFIG_FILE, RESULTS_FILE, WEIGHTS_DIR, BEST_WEIGHTS

"""
======================================================
services/experiment_service.py
------------------------------------------------------
Responsibility:
- Manage training experiment metadata
- Provide experiment listing (status, progress, config)
- Handle download of trained weights

Design rules:
- No HTTP routing logic
- No training execution logic
- Only read from filesystem (runs directory)
======================================================
"""


# --------------------------------------------------
# List experiments
# --------------------------------------------------
def list_experiments_service():
    """
    Retrieve all training runs with metadata and status.

    Returns:
    - list of experiments with:
        name, status, model, dataset, training config,
        progress (epoch), and artefact availability
    """

    experiments = []

    for run_dir in RUNS_DIR.iterdir():
        if not run_dir.is_dir():
            continue

        cfg_path = run_dir / CONFIG_FILE
        results_path = run_dir / RESULTS_FILE
        weights_path = run_dir / WEIGHTS_DIR / BEST_WEIGHTS

        # Skip incomplete runs with no config
        if not cfg_path.exists():
            continue

        # Load run configuration
        with open(cfg_path, "r") as f:
            cfg = json.load(f)

        epoch = 0
        total_epochs = int(cfg.get("epochs", 0))

        # Read latest training progress (if available)
        if results_path.exists():
            with open(results_path, newline="") as f:
                rows = list(csv.DictReader(f))

            if rows:
                last = rows[-1]
                epoch = int(float(last["epoch"]))

        # --------------------------------------------------
        # Determine experiment status
        # --------------------------------------------------
        if run_dir.name == state.active_run:
            status = "Running"

        elif results_path.exists():
            if epoch >= total_epochs:
                status = "Completed"
            else:
                status = "Stopped"

        else:
            status = "Failed"

        experiments.append({
            "name": run_dir.name,
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


# --------------------------------------------------
# Download best weights
# --------------------------------------------------
def download_weights_service(run):
    """
    Download the best trained weights for a given run.

    Input:
    - run: training run name

    Returns:
    - best.pt file

    Raises:
    - 404 if weights not found
    """

    path = RUNS_DIR / run / WEIGHTS_DIR / BEST_WEIGHTS

    if not path.exists():
        raise HTTPException(404, "Weights not found")

    return FileResponse(path, filename=f"{run}_best.pt")