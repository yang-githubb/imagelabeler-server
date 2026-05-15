from pydantic import BaseModel

class TrainConfig(BaseModel):
    station: str
    process: str
    model: str
    epochs: int
    imgsz: int
    batch: int
    runName: str
