from fastapi import APIRouter
from services.experiment_service import (
    list_experiments_service,
    download_weights_service
)

"""
======================================================
routes/experiment_routes.py
------------------------------------------------------
Responsibility:
- Define API endpoints for training experiment management
- Provide access to experiment metadata and outputs

Design rules:
- No business logic here
- Only delegate to service layer
- Keep endpoints thin and predictable
======================================================
"""

# --------------------------------------------------
# Router configuration
# --------------------------------------------------
# All endpoints will be prefixed with /experiments
router = APIRouter(prefix="/experiments", tags=["experiments"])


# --------------------------------------------------
# List all experiments
# --------------------------------------------------
@router.get("/")
def list_experiments():
    """
    Retrieve a list of all training runs (experiments).

    Returns:
    - list of experiments with status, config, and metadata
    """
    return list_experiments_service()


# --------------------------------------------------
# Download best weights from a specific run
# --------------------------------------------------
@router.get("/{run}/weights")
def download_experiment_weights(run: str):
    """
    Download the best trained weights (.pt) for a given run.

    Path params:
    - run: name of the training run

    Returns:
    - best.pt file as a downloadable response
    """
    return download_weights_service(run)