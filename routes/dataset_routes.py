from fastapi import APIRouter, UploadFile, File, Form, Query
from services.dataset_service import (
    dataset_upload_service,
    get_classes_service,
    list_dataset_images_service
)

"""
======================================================
routes/dataset_routes.py
------------------------------------------------------
Responsibility:
- Define API endpoints for dataset-related operations
- Act as a thin layer between HTTP requests and services

Design rules:
- No business logic here
- No filesystem operations here
- Delegate all work to service layer
- Keep endpoints simple and predictable
======================================================
"""

# --------------------------------------------------
# Router configuration
# --------------------------------------------------
# All endpoints will be prefixed with /datasets
router = APIRouter(prefix="/datasets", tags=["dataset"])


# --------------------------------------------------
# Upload dataset (image + label + thumbnail)
# --------------------------------------------------
@router.post("/upload")
async def dataset_upload(
    station: str = Form(...),
    process: str = Form(...),
    image: UploadFile = File(...),
    label: UploadFile = File(...),
    thumb: UploadFile = File(...)
):
    """
    Upload a single dataset item.

    Required inputs:
    - station: dataset station identifier
    - process: process name (e.g., inspection type)
    - image: original image file
    - label: YOLO label (.txt)
    - thumb: thumbnail version of image

    Returns:
    - status result from service layer
    """
    return await dataset_upload_service(
        station, process, image, label, thumb
    )


# --------------------------------------------------
# Get class list from dataset.yaml
# --------------------------------------------------
@router.get("/classes")
def get_dataset_classes(station: str, process: str):
    """
    Retrieve class names defined in dataset.yaml.

    Query params:
    - station
    - process

    Returns:
    - list of class names
    """
    return get_classes_service(station, process)


# --------------------------------------------------
# List dataset images (paginated)
# --------------------------------------------------
@router.get("/images")
def list_dataset_images(
    station: str = Query(...),
    process: str = Query(...),
    page: int = Query(1, ge=1),
    limit: int = Query(24, ge=1)
):
    """
    List dataset images with pagination.

    Query params:
    - station
    - process
    - page: page number (1-based)
    - limit: number of images per page

    Returns:
    - total image count
    - list of image filenames (current page)
    """
    return list_dataset_images_service(
        station, process, page, limit
    )