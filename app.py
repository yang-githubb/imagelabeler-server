from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from routes import (
    train_routes,
    experiment_routes,
    dataset_routes,
    model_routes
)
from core.config import DATASET_ROOT

"""
======================================================
app.py
------------------------------------------------------
Responsibility:
- Entry point of the FastAPI application
- Register all API routes
- Configure static file serving

Design rules:
- No business logic
- Only application wiring (routers, middleware, configs)
======================================================
"""

# --------------------------------------------------
# Application instance
# --------------------------------------------------
app = FastAPI()


# --------------------------------------------------
# Register routers
# --------------------------------------------------
# Each router handles a specific domain
app.include_router(train_routes.router)
app.include_router(experiment_routes.router)
app.include_router(dataset_routes.router)
app.include_router(model_routes.router)


# --------------------------------------------------
# Static file serving
# --------------------------------------------------
# Exposes dataset directory for direct file access
# Useful for:
# - viewing images in UI
# - loading thumbnails
app.mount(
    "/datasets",
    StaticFiles(directory=DATASET_ROOT),
    name="datasets"
)
