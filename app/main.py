"""FastAPI server: upload a horse video, get back an annotated clip + metrics."""

from __future__ import annotations

import shutil
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from .vision import analyze_video

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "outputs"
for d in (UPLOAD_DIR, OUTPUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Which YOLO weights to use. yolov8n is small/fast; swap for yolov8s/m for
# better accuracy at the cost of speed.
MODEL_NAME = "yolov8n.pt"
MAX_FRAMES = 900  # safety cap (~30s at 30fps) so a huge upload can't hang

app = FastAPI(title="Horse Motion Analyzer")

# job_id -> {"status", "progress", "result", "error"}
JOBS: dict[str, dict] = {}


def _run_job(job_id: str, in_path: Path, out_path: Path) -> None:
    def progress(p: float) -> None:
        JOBS[job_id]["progress"] = round(float(p), 3)

    try:
        JOBS[job_id]["status"] = "processing"
        result = analyze_video(
            str(in_path), str(out_path),
            model_name=MODEL_NAME, max_frames=MAX_FRAMES, progress=progress,
        )
        payload = result.to_dict()
        payload["video_url"] = f"/results/{out_path.name}"
        JOBS[job_id]["result"] = payload
        JOBS[job_id]["progress"] = 1.0
        JOBS[job_id]["status"] = "done"
    except Exception as exc:  # surface a readable message to the UI
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(exc)


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "No file provided.")
    ext = Path(file.filename).suffix.lower()
    if ext not in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        raise HTTPException(400, f"Unsupported file type: {ext}")

    job_id = uuid.uuid4().hex[:12]
    in_path = UPLOAD_DIR / f"{job_id}{ext}"
    out_path = OUTPUT_DIR / f"{job_id}_annotated.mp4"

    with in_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    JOBS[job_id] = {"status": "queued", "progress": 0.0, "result": None, "error": None}
    threading.Thread(target=_run_job, args=(job_id, in_path, out_path), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
async def status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job id.")
    return JSONResponse({
        "status": job["status"],
        "progress": job["progress"],
        "result": job["result"],
        "error": job["error"],
    })


@app.get("/results/{name}")
async def result_video(name: str):
    path = OUTPUT_DIR / name
    if not path.exists():
        raise HTTPException(404, "Result not found.")
    return FileResponse(path, media_type="video/mp4")


app.mount("/", StaticFiles(directory=str(BASE_DIR / "static"), html=True), name="static")
