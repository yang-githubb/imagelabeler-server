from fastapi import APIRouter, UploadFile, File, Form, Query
from services.dataset_service import (
    dataset_upload_service,
    get_classes_service,
    list_dataset_images_service
)

router = APIRouter(prefix="/datasets", tags=["dataset"])

@router.post("/upload")
async def dataset_upload(
    station: str = Form(...),
    process: str = Form(...),
    image: UploadFile = File(...),
    label: UploadFile = File(...),
    thumb: UploadFile = File(...)
):
    return await dataset_upload_service(
        station, process, image, label, thumb
    )


@router.get("/classes")
def get_dataset_classes(station: str, process: str):
    return get_classes_service(station, process)


@router.get("/images")
def list_dataset_images(
    station: str = Query(...),
    process: str = Query(...),
    page: int = Query(1, ge=1),
    limit: int = Query(24, ge=1)
):
    return list_dataset_images_service(
        station, process, page, limit
    )