from fastapi import APIRouter
from schemas.train_schema import TrainConfig
from services.train_service import (
    start_train_service,
    train_progress_service,
    train_metrics_service,
    stop_train_service
)

"""
======================================================
routes/train_routes.py
------------------------------------------------------
Responsibility:
- Define API endpoints for training control and monitoring
- Provide start / stop / progress / metrics interfaces

Design rules:
- No training logic here
- No subprocess handling here
- Delegate everything to service layer
- Keep endpoints simple and stable
======================================================
"""

# --------------------------------------------------
# Router configuration
# --------------------------------------------------
# All endpoints will be prefixed with /train
router = APIRouter(prefix="/train", tags=["train"])


# --------------------------------------------------
# Start training
# --------------------------------------------------
@router.post("/start")
def start_train(cfg: TrainConfig):
    """
    Start a new training run.

    Request body:
    - TrainConfig (station, process, model, epochs, etc.)

    Returns:
    - status and run name
    """
    return start_train_service(cfg)


# --------------------------------------------------
# Get training progress
# --------------------------------------------------
@router.get("/progress")
def train_progress():
    """
    Retrieve current training progress.

    Returns:
    - status (idle / starting / running)
    - run name
    - current epoch / total epochs
    - progress percentage
    """
    return train_progress_service()


# --------------------------------------------------
# Get training metrics
# --------------------------------------------------
@router.get("/metrics")
def train_metrics(run: str | None = None):
    """
    Retrieve training metrics.

    Query params:
    - run (optional): specific run name
      - if not provided, uses active run

    Returns:
    - list of metrics per epoch (loss, mAP, etc.)
    """
    return train_metrics_service(run)


# --------------------------------------------------
# Stop training
# --------------------------------------------------
@router.post("/stop")
def stop_train():
    """
    Stop the currently running training process.

    Returns:
    - status and run name
    """
    return stop_train_service()