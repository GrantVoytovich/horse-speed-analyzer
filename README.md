# 🐎 Horse Motion Analyzer

Upload a video of a horse and get back an annotated clip plus motion metrics —
**no camera calibration required**. Speed is reported in **body lengths per
second (BL/s)**, a scale-invariant measure that stays consistent whether the
horse is filmed up close or far away.

## What it measures

| Metric | Meaning |
| --- | --- |
| **Peak / average speed (BL/s)** | How fast the horse moves relative to its own body length. |
| **Gait** | Heuristic: standing, walk, trot, canter, gallop. |
| **Cadence** | Stride rate per minute, from the vertical bob of the horse. |
| **Smoothness** | 0–100%, how steady the motion is (higher = steadier). |

It also writes an **annotated MP4** with a live speed/gait overlay and a box on
the tracked horse.

## How it works

1. **YOLOv8** detects the horse in each frame.
2. **ByteTrack** keeps it as the same object across frames.
3. The dominant horse's centre and bounding-box width are turned into a smoothed
   trajectory. Width ≈ body length, so dividing displacement by width removes the
   effect of camera distance.
4. Speed = displacement ÷ body length × fps → **BL/s**.
5. Cadence = dominant frequency of the vertical bob (FFT).

## Setup (Windows / PowerShell)

```powershell
cd C:\Users\grant\horse-speed-analyzer
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The first run downloads the `yolov8n.pt` weights (~6 MB) automatically.

## Run

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --port 8000
```

Open <http://localhost:8000>, drop in a video, and wait for the result.

## Tips for good results

- **Film side-on** to the horse's motion (perpendicular to the camera).
- Keep the **whole horse in frame** and reasonably large.
- One horse per clip works best; with several, the most prominent is analysed.
- Clips are capped at ~30s by default (`MAX_FRAMES` in `app/main.py`).

## Where to go next

- **True mph:** add a known reference distance in the scene (e.g. click two
  fence posts a measured distance apart) and convert BL/s → m/s → mph.
- **Better gait / lameness detection:** add a horse **pose/keypoint** model to
  measure per-leg stride length and left/right symmetry.
- **Accuracy:** swap `yolov8n.pt` for `yolov8s.pt`/`yolov8m.pt` in `app/main.py`.
- **Tuning gaits:** the thresholds in `app/vision.py::_classify_gait` are rough —
  label a few of your friend's own clips and adjust.
