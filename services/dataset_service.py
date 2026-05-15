from shutil import move
import tempfile
import yaml
from fastapi import HTTPException
from core.config import DATASET_ROOT

"""
======================================================
services/dataset_service.py
------------------------------------------------------
Responsibility:
- Handle dataset-related business logic
- Manage file storage for images, labels, and thumbnails
- Provide dataset metadata (classes, image listing)

Design rules:
- No HTTP logic here (handled in routes)
- No training logic here
- All filesystem operations centralized here
======================================================
"""


# --------------------------------------------------
# Upload dataset item
# --------------------------------------------------
async def dataset_upload_service(station, process, image, label, thumb):
    """
    Store dataset files (image, label, thumbnail) on disk.

    Inputs:
    - station: dataset station identifier
    - process: process name
    - image: uploaded image file
    - label: YOLO label file
    - thumb: thumbnail image

    Behavior:
    - Ensures directory structure exists
    - Uses temporary files to avoid partial writes
    - Moves files to final location

    Returns:
    - status confirmation
    """

    # Target directories
    images_dir = DATASET_ROOT / station / process / "images"
    labels_dir = DATASET_ROOT / station / process / "labels"
    thumbs_dir = images_dir / "thumbs"

    # Ensure directory structure exists
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    # Final file paths
    image_path = images_dir / image.filename
    label_path = labels_dir / label.filename
    thumb_path = thumbs_dir / thumb.filename

    # Save uploaded files to temporary files first
    # (prevents file corruption during write)
    with tempfile.NamedTemporaryFile(delete=False) as tmp_img:
        tmp_img.write(await image.read())
        tmp_img_path = tmp_img.name

    with tempfile.NamedTemporaryFile(delete=False) as tmp_lbl:
        tmp_lbl.write(await label.read())
        tmp_lbl_path = tmp_lbl.name

    with tempfile.NamedTemporaryFile(delete=False) as tmp_th:
        tmp_th.write(await thumb.read())
        tmp_th_path = tmp_th.name

    # Move files to final destination
    move(tmp_img_path, image_path)
    move(tmp_lbl_path, label_path)
    move(tmp_th_path, thumb_path)

    return {"status": "ok"}


# --------------------------------------------------
# Get dataset classes
# --------------------------------------------------
def get_classes_service(station, process):
    """
    Retrieve class names from dataset.yaml.

    Inputs:
    - station
    - process

    Returns:
    - list of class names

    Raises:
    - 404 if dataset.yaml not found
    """

    dataset_yaml = DATASET_ROOT / station / process / "dataset.yaml"

    if not dataset_yaml.exists():
        raise HTTPException(404, "dataset.yaml not found")

    with open(dataset_yaml, "r") as f:
        data = yaml.safe_load(f)

    names = data.get("names", {})

    # Ensure consistent ordering (YOLO class index order)
    return {"classes": [names[k] for k in sorted(names.keys())]}


# --------------------------------------------------
# List dataset images (paginated)
# --------------------------------------------------
def list_dataset_images_service(station, process, page, limit):
    """
    List dataset images with pagination.

    Inputs:
    - station
    - process
    - page (1-based)
    - limit (items per page)

    Returns:
    - total image count
    - list of filenames (current page only)
    """

    images_dir = DATASET_ROOT / station / process / "images"

    if not images_dir.exists():
        return {"total": 0, "images": []}

    # Filter valid image files only
    files = sorted(
        f.name for f in images_dir.iterdir()
        if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
    )

    total = len(files)

    # Pagination
    start = (page - 1) * limit
    page_items = files[start:start + limit]

    return {
        "total": total,
        "images": page_items
    }