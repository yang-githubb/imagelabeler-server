from fastapi import APIRouter, UploadFile, File
import torch
import tempfile
import os

router = APIRouter(prefix="/model", tags=["model"])


@router.post("/classes")
async def extract_model_classes(file: UploadFile = File(...)):
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        model = torch.load(tmp_path, map_location="cpu")

        names = {}

        # Case 1: direct dict
        if isinstance(model, dict) and "names" in model:
            names = model["names"]

        # Case 2: checkpoint → model attribute
        elif isinstance(model, dict) and "model" in model:
            try:
                names = getattr(model["model"], "names", {})
            except:
                names = {}

        # Case 3: direct model object
        elif hasattr(model, "names"):
            names = model.names

        # normalize to list
        if isinstance(names, dict):
            classes = list(names.values())
        else:
            classes = list(names) if names else []

        return {
            "classes": classes
        }

    except Exception as e:
        return {
            "classes": [],
            "error": str(e)
        }

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
