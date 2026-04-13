"""FastAPI application for sheep-seg demo."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from fastapi.middleware.cors import CORSMiddleware

from backend.config import HOST, PORT, RESULTS_DIR, SAMPLE_DIR, UPLOAD_DIR
from backend.routes import analyze, session, upload

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(
    title="SamSeesSheep",
    description="SAM segmentation + depth mesh for sheep facial welfare monitoring",
    version="0.1.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Include routers
app.include_router(upload.router)
app.include_router(analyze.router)
app.include_router(session.router)

# Serve uploaded images and sample images
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/sample", StaticFiles(directory=str(SAMPLE_DIR)), name="sample")
app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")

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


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok", "project": "sheep-seg", "version": "0.1.0"}


def main():
    import uvicorn

    uvicorn.run("backend.main:app", host=HOST, port=PORT, reload=True)


if __name__ == "__main__":
    main()
