"""FastAPI app for sheep-yolo."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.config import HOST, PORT, RESULTS_DIR, UPLOAD_DIR
from backend.routes import analyze, upload

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(
    title="sheep-yolo",
    description="Out-of-the-box YOLO benchmark for sheep ear-angle welfare monitoring",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

app.include_router(upload.router)
app.include_router(analyze.router)

app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


@app.get("/")
async def serve_frontend():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
async def health():
    return {"status": "ok", "project": "sheep-yolo", "version": "0.2.0"}


def main():
    import uvicorn
    uvicorn.run("backend.main:app", host=HOST, port=PORT, reload=True)


if __name__ == "__main__":
    main()
