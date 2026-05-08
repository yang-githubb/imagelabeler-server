from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from routes import train_routes, experiment_routes, dataset_routes, model_routes
app = FastAPI()

BASE_DIR = Path(__file__).parent
DATASET_ROOT = BASE_DIR / "datasets"

# include routers
app.include_router(train_routes.router)
app.include_router(experiment_routes.router)
app.include_router(dataset_routes.router)
app.include_router(model_routes.router)

# static datasets
app.mount(
    "/datasets",
    StaticFiles(directory=DATASET_ROOT),
    name="datasets"
)
