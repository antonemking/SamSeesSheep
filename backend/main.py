"""FastAPI application for sheep-seg demo."""

from __future__ import annotations

# Must set before torch is imported anywhere in the process — the CUDA
# allocator config is read when the CUDA context initializes. Setting
# it inside a lazy model loader is too late to take effect.
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from fastapi.middleware.cors import CORSMiddleware

from backend.config import HOST, PORT, LABELS_DIR, RESULTS_DIR, UPLOAD_DIR
from backend.routes import analyze, export, label, upload

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(
    title="SamSeesSheep",
    description="Live ear-angle visualization for sheep — SAM 3 + SPFES-referenced thresholds",
    version="0.1.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Include routers
app.include_router(upload.router)
app.include_router(analyze.router)
app.include_router(label.router)
app.include_router(export.router)

# Serve uploaded images and sample images
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")
app.mount("/labels", StaticFiles(directory=str(LABELS_DIR)), name="labels")

# Serve frontend
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

# Serve Three.js library files
LIB_DIR = FRONTEND_DIR / "lib"
if LIB_DIR.exists():
    app.mount("/lib", StaticFiles(directory=str(LIB_DIR)), name="lib")


@app.get("/")
async def serve_frontend():
    """Serve the dashboard HTML."""
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/label/{video_id}")
async def serve_label_page(video_id: str):
    """Serve the keypoint labeling UI for one video."""
    return FileResponse(FRONTEND_DIR / "label.html")


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok", "project": "sheep-seg", "version": "0.1.0"}


def main():
    import uvicorn

    uvicorn.run("backend.main:app", host=HOST, port=PORT, reload=True)


if __name__ == "__main__":
    main()
