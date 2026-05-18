from shutil import move
import tempfile
import yaml
from fastapi import HTTPException
from core.config import DATASET_ROOT
import random
import shutil

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

# --------------------------------------------------
# Split dataset into test/val
# --------------------------------------------------
import random
import shutil
from pathlib import Path

def split_dataset(station, process, val_ratio=0.2):
    """
    Ensure dataset is split into train / validation with stable ratio.

    Behavior:
    - First run: create initial validation split
    - Subsequent runs: adjust only if ratio is off
    - Never reshuffles already selected validation data
    """

    base = DATASET_ROOT / station / process

    images_dir = base / "images"
    labels_dir = base / "labels"

    val_img_dir = base / "validation" / "images"
    val_lbl_dir = base / "validation" / "labels"

    # Ensure validation folders exist
    val_img_dir.mkdir(parents=True, exist_ok=True)
    val_lbl_dir.mkdir(parents=True, exist_ok=True)

    # Collect files
    train_files = [
        f for f in images_dir.iterdir()
        if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]

    val_files = [
        f for f in val_img_dir.iterdir()
        if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]

    train_count = len(train_files)
    val_count = len(val_files)
    total_count = train_count + val_count

    # -------------------------------
    # Logging: current dataset state
    # -------------------------------
    print(
        f"[DATASET SPLIT] {station}/{process} | "
        f"Total={total_count}, Train={train_count}, Val={val_count}"
    )

    # Skip very small datasets
    if total_count < 5:
        print("[DATASET SPLIT] Skipped: dataset too small")
        return

    # Compute desired validation size
    target_val_count = max(1, int(total_count * val_ratio))
    needed = target_val_count - val_count

    print(
        f"[DATASET SPLIT] Target Val={target_val_count} "
        f"({int(val_ratio * 100)}%), Needed={needed}"
    )

    # Already balanced
    if needed <= 0:
        print("[DATASET SPLIT] No action needed (already balanced)")
        return

    # Shuffle only training candidates
    random.shuffle(train_files)

    # Move required number of images
    moved = 0

    for img_path in train_files[:needed]:
        label_path = labels_dir / f"{img_path.stem}.txt"

        # Move image
        shutil.move(str(img_path), val_img_dir / img_path.name)

        # Move label if exists
        if label_path.exists():
            shutil.move(
                str(label_path),
                val_lbl_dir / label_path.name
            )

        moved += 1

    print(
        f"[DATASET SPLIT] Moved {moved} image(s) to validation | "
        f"New Train={train_count - moved}, New Val={val_count + moved}"
    )