from fastapi import APIRouter
from pathlib import Path
import json

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "inspection_state.json"

@router.post("/inspection/config")
async def save_inspection_config(config: dict):
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    return {"status": "ok"}

@router.get("/inspection/config")
async def read_inspection_config():
    if not CONFIG_PATH.exists():
        return None

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)