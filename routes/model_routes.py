from fastapi import APIRouter, UploadFile, File
import torch
import tempfile
import os

"""
======================================================
routes/model_routes.py
------------------------------------------------------
Responsibility:
- Handle model-related utilities
- Extract metadata from uploaded YOLO model files

Design rules:
- No persistence (temporary file usage only)
- No training logic
- Focus on lightweight model inspection
======================================================
"""

# --------------------------------------------------
# Router configuration
# --------------------------------------------------
# All endpoints will be prefixed with /model
router = APIRouter(prefix="/model", tags=["model"])


# --------------------------------------------------
# Extract class names from YOLO model
# --------------------------------------------------
@router.post("/classes")
async def extract_model_classes(file: UploadFile = File(...)):
    """
    Extract class names from an uploaded YOLO model (.pt file).

    Input:
    - file: PyTorch model file (.pt)

    Returns:
    - classes: list of class names
    - error (optional): error message if extraction fails

    Notes:
    - Handles multiple model formats (dict checkpoint / model object)
    - Uses temporary file to allow torch.load()
    - Always cleans up temporary file
    """
    tmp_path = None

    try:
        # ------------------------------------------
        # Save uploaded model to a temporary file
        # ------------------------------------------
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # ------------------------------------------
        # Load model (CPU only for compatibility)
        # ------------------------------------------
        model = torch.load(tmp_path, map_location="cpu")

        names = {}

        # ------------------------------------------
        # Extract class names from different formats
        # ------------------------------------------

        # Case 1: checkpoint dict contains "names"
        if isinstance(model, dict) and "names" in model:
            names = model["names"]

        # Case 2: checkpoint dict contains a model object
        elif isinstance(model, dict) and "model" in model:
            try:
                names = getattr(model["model"], "names", {})
            except Exception:
                names = {}

        # Case 3: direct model object with attribute "names"
        elif hasattr(model, "names"):
            names = model.names

        # ------------------------------------------
        # Normalize output to a list
        # ------------------------------------------
        if isinstance(names, dict):
            classes = list(names.values())
        else:
            classes = list(names) if names else []

        return {"classes": classes}

    except Exception as e:
        return {
            "classes": [],
            "error": str(e)
        }

    finally:
        # ------------------------------------------
        # Cleanup temporary file
        # ------------------------------------------
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)