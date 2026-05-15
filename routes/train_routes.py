from fastapi import APIRouter
from schemas.train_schema import TrainConfig
from services.train_service import (
    start_train_service,
    train_progress_service,
    train_metrics_service,
    stop_train_service
)

router = APIRouter(prefix="/train", tags=["train"])

@router.post("/start")
def start_train(cfg: TrainConfig):
    return start_train_service(cfg)


@router.get("/progress")
def train_progress():
    return train_progress_service()


@router.get("/metrics")
def train_metrics(run: str | None = None):
    return train_metrics_service(run)


@router.post("/stop")
def stop_train():
    return stop_train_service()
