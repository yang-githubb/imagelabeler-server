from pathlib import Path
from shutil import move
import tempfile
import yaml
from fastapi import HTTPException

BASE_DIR = Path(__file__).parent.parent
DATASET_ROOT = BASE_DIR / "datasets"


async def dataset_upload_service(station, process, image, label, thumb):
    images_dir = DATASET_ROOT / station / process / "images"
    labels_dir = DATASET_ROOT / station / process / "labels"
    thumbs_dir = images_dir / "thumbs"

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    image_path = images_dir / image.filename
    label_path = labels_dir / label.filename
    thumb_path = thumbs_dir / thumb.filename

    with tempfile.NamedTemporaryFile(delete=False) as tmp_img:
        tmp_img.write(await image.read())
        tmp_img_path = tmp_img.name

    with tempfile.NamedTemporaryFile(delete=False) as tmp_lbl:
        tmp_lbl.write(await label.read())
        tmp_lbl_path = tmp_lbl.name

    with tempfile.NamedTemporaryFile(delete=False) as tmp_th:
        tmp_th.write(await thumb.read())
        tmp_th_path = tmp_th.name

    move(tmp_img_path, image_path)
    move(tmp_lbl_path, label_path)
    move(tmp_th_path, thumb_path)

    return {"status": "ok"}


def get_classes_service(station, process):
    dataset_yaml = (
        BASE_DIR / "datasets" / station / process / "dataset.yaml"
    )

    if not dataset_yaml.exists():
        raise HTTPException(404, "dataset.yaml not found")

    with open(dataset_yaml, "r") as f:
        data = yaml.safe_load(f)

    names = data.get("names")
    return {"classes": [names[k] for k in sorted(names.keys())]}


def list_dataset_images_service(station, process, page, limit):
    images_dir = (
        BASE_DIR / "datasets" / station / process / "images"
    )

    if not images_dir.exists():
        return {"total": 0, "images": []}

    files = sorted(
        f.name for f in images_dir.iterdir()
        if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
    )

    total = len(files)
    start = (page - 1) * limit
    slice_ = files[start:start + limit]

    return {
        "total": total,
        "images": slice_
    }
