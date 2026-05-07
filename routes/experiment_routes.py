from fastapi import APIRouter
from services.experiment_service import (
    list_experiments_service,
    download_weights_service
)

router = APIRouter()

@router.get("/experiments")
def list_experiments():
    return list_experiments_service()


@router.get("/experiments/{run}/weights")
def download_experiment_weights(run: str):
    return download_weights_service(run)
