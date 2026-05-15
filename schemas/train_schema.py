from pydantic import BaseModel

"""
======================================================
schemas/train_schema.py
------------------------------------------------------
Responsibility:
- Define request schema for training configuration
- Validate incoming data from API before reaching service layer

Design rules:
- No business logic
- Strict input validation only
- Keep structure aligned with training pipeline inputs
======================================================
"""


class TrainConfig(BaseModel):
    """
    Training configuration payload.

    Fields:
    - station: dataset station identifier (e.g., station_01)
    - process: process name (e.g., final_inspection)
    - model: model name or path to .pt file
    - epochs: number of training epochs
    - imgsz: image size for training
    - batch: batch size
    - runName: unique name for this training run
    """

    station: str
    process: str
    model: str
    epochs: int
    imgsz: int
    batch: int
    runName: str
