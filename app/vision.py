"""
Calibration-free horse motion analysis.

The trick to getting a *meaningful* speed number without knowing how far the
camera is from the horse is to measure movement in units of the horse's OWN
body length. A horse's bounding-box width is a stand-in for its body length, so
"body lengths per second" (BL/s) stays roughly the same whether the horse is
filmed up close or far away. That gives us a scale-invariant relative speed plus
gait metrics, with no need to calibrate the scene.

Pipeline:
  1. Detect + track the horse across frames (YOLO + ByteTrack).
  2. Pick the single dominant horse track (biggest / most consistent).
  3. Build a smoothed trajectory of its centre and body length.
  4. Speed  = centre displacement / body length, per second  -> BL/s.
  5. Cadence = dominant frequency of the vertical bob (FFT)   -> strides/min.
  6. Gait    = heuristic thresholds on speed + cadence.
  7. Render an annotated MP4 and return a JSON-friendly summary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import cv2
import imageio.v2 as imageio
import numpy as np
from scipy.ndimage import uniform_filter1d

# COCO class id for "horse". YOLO models trained on COCO use this index.
HORSE_CLASS_ID = 17


@dataclass
class FrameSample:
    frame_idx: int
    cx: float
    cy: float
    width: float   # bbox width  (body-length proxy)
    height: float  # bbox height
    box: tuple[float, float, float, float]  # x1, y1, x2, y2


@dataclass
class AnalysisResult:
    fps: float
    frame_count: int
    duration_s: float
    frames_with_horse: int
    peak_speed_bls: float
    avg_speed_bls: float
    cadence_per_min: float
    gait: str
    smoothness: float          # 0..1, higher = steadier motion
    speed_series: list[float] = field(default_factory=list)  # BL/s per sampled frame
    annotated_video: Optional[str] = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "fps": round(self.fps, 2),
            "frame_count": self.frame_count,
            "duration_s": round(self.duration_s, 2),
            "frames_with_horse": self.frames_with_horse,
            "peak_speed_bls": round(self.peak_speed_bls, 2),
            "avg_speed_bls": round(self.avg_speed_bls, 2),
            "cadence_per_min": round(self.cadence_per_min, 1),
            "gait": self.gait,
            "smoothness": round(self.smoothness, 2),
            "speed_series": [round(v, 3) for v in self.speed_series],
            "notes": self.notes,
        }


def _moving_average(x: np.ndarray, win: int) -> np.ndarray:
    # mode="nearest" repeats the edge value instead of zero-padding, so the
    # first/last samples aren't dragged toward zero (which otherwise creates
    # huge fake displacement spikes at the clip boundaries).
    if win <= 1 or len(x) < 2:
        return x
    win = min(win, len(x))
    return uniform_filter1d(x.astype(float), size=win, mode="nearest")


def _classify_gait(speed_bls: float, cadence_per_min: float) -> str:
    """Heuristic gait buckets based on scale-invariant speed.

    These thresholds are rough starting points, not veterinary fact. They are
    easy to tune once your friend labels a handful of his own clips.
    """
    if speed_bls < 0.4:
        return "standing / grazing"
    if speed_bls < 1.2:
        return "walk"
    if speed_bls < 2.3:
        return "trot"
    if speed_bls < 4.0:
        return "canter"
    return "gallop"


def _estimate_cadence(cy: np.ndarray, body_len: np.ndarray, fps: float) -> float:
    """Strides per minute from the vertical bob of the horse's centre.

    A moving horse bobs up and down once per stride cycle. We normalise the
    vertical signal by body length (so camera distance doesn't matter), detrend
    it, and take the dominant frequency via FFT within a plausible gait band.
    """
    if len(cy) < 8 or fps <= 0:
        return 0.0
    signal = cy / np.maximum(body_len, 1e-6)
    signal = signal - np.mean(signal)
    # Remove slow drift (the horse crossing the frame) with a light detrend.
    t = np.arange(len(signal))
    if len(t) >= 2:
        coeffs = np.polyfit(t, signal, 1)
        signal = signal - np.polyval(coeffs, t)
    windowed = signal * np.hanning(len(signal))
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / fps)  # Hz
    # Plausible stride band: 0.4 Hz (slow walk) to 3.5 Hz (fast gallop steps).
    band = (freqs >= 0.4) & (freqs <= 3.5)
    if not np.any(band):
        return 0.0
    dominant_hz = freqs[band][int(np.argmax(spectrum[band]))]
    return float(dominant_hz * 60.0)


def analyze_video(
    input_path: str,
    output_path: str,
    model_name: str = "yolov8n.pt",
    sample_stride: int = 1,
    max_frames: Optional[int] = None,
    progress=None,
) -> AnalysisResult:
    """Run the full analysis and write an annotated MP4 to ``output_path``."""
    from ultralytics import YOLO  # imported lazily so the module loads fast

    model = YOLO(model_name)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Browser-playable H.264 output via the ffmpeg bundled with imageio.
    writer = imageio.get_writer(
        output_path, fps=fps, codec="libx264", quality=7,
        macro_block_size=None, ffmpeg_params=["-pix_fmt", "yuv420p"],
    )

    notes: list[str] = []
    # track_id -> list[FrameSample]
    tracks: dict[int, list[FrameSample]] = {}
    frames: list[np.ndarray] = []   # kept in RGB for annotation pass
    frame_idx = 0
    processed = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % sample_stride != 0:
            frame_idx += 1
            continue

        results = model.track(
            frame, persist=True, classes=[HORSE_CLASS_ID],
            tracker="bytetrack.yaml", verbose=False,
        )
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(rgb)

        r = results[0]
        if r.boxes is not None and r.boxes.id is not None:
            ids = r.boxes.id.cpu().numpy().astype(int)
            xyxy = r.boxes.xyxy.cpu().numpy()
            for tid, (x1, y1, x2, y2) in zip(ids, xyxy):
                w = float(x2 - x1)
                h = float(y2 - y1)
                sample = FrameSample(
                    frame_idx=processed,
                    cx=float((x1 + x2) / 2),
                    cy=float((y1 + y2) / 2),
                    width=w, height=h, box=(float(x1), float(y1), float(x2), float(y2)),
                )
                tracks.setdefault(int(tid), []).append(sample)

        processed += 1
        frame_idx += 1
        if progress and total:
            progress(min(0.6, 0.6 * frame_idx / total))
        if max_frames and processed >= max_frames:
            notes.append(f"Analysis capped at {max_frames} frames.")
            break

    cap.release()
    eff_fps = fps / sample_stride

    if not tracks:
        writer.close()
        raise ValueError(
            "No horse was detected in the video. Make sure the horse is clearly "
            "visible and try a clip filmed from the side."
        )

    # Dominant track = the one seen in the most frames (ties broken by size).
    def track_score(samples: list[FrameSample]) -> float:
        avg_area = np.mean([s.width * s.height for s in samples])
        return len(samples) * 1e6 + avg_area

    main_id = max(tracks, key=lambda k: track_score(tracks[k]))
    main = sorted(tracks[main_id], key=lambda s: s.frame_idx)
    if len(tracks) > 1:
        notes.append(f"{len(tracks)} horse tracks seen; analysed the most prominent one.")

    # Build dense, interpolated series over the frames the horse was present.
    idxs = np.array([s.frame_idx for s in main])
    cx = np.array([s.cx for s in main])
    cy = np.array([s.cy for s in main])
    bw = np.array([s.width for s in main])

    full_idx = np.arange(idxs.min(), idxs.max() + 1)
    cx = np.interp(full_idx, idxs, cx)
    cy = np.interp(full_idx, idxs, cy)
    bw = np.interp(full_idx, idxs, bw)

    smooth_win = max(1, int(round(eff_fps / 5)))  # ~0.2s smoothing
    cx_s = _moving_average(cx, smooth_win)
    cy_s = _moving_average(cy, smooth_win)
    body_len = np.maximum(_moving_average(bw, smooth_win), 1e-6)

    # Per-frame displacement in pixels -> body lengths -> BL/s.
    dx = np.diff(cx_s)
    dy = np.diff(cy_s)
    disp_px = np.sqrt(dx * dx + dy * dy)
    disp_bl = disp_px / body_len[1:]
    speed_series = disp_bl * eff_fps  # body lengths per second
    speed_series = _moving_average(speed_series, smooth_win)

    peak_speed = float(np.percentile(speed_series, 95)) if len(speed_series) else 0.0
    avg_speed = float(np.mean(speed_series)) if len(speed_series) else 0.0
    cadence = _estimate_cadence(cy_s, body_len, eff_fps)
    gait = _classify_gait(peak_speed, cadence)

    # Smoothness: lower relative variance of speed = steadier motion.
    if avg_speed > 1e-6 and len(speed_series) > 1:
        cv_speed = float(np.std(speed_series) / avg_speed)
        smoothness = float(max(0.0, min(1.0, 1.0 - cv_speed / 2.0)))
    else:
        smoothness = 0.0

    # --- Annotation pass -----------------------------------------------------
    box_by_idx = {s.frame_idx: s.box for s in main}
    speed_by_idx: dict[int, float] = {}
    for i, fidx in enumerate(full_idx[1:]):
        speed_by_idx[int(fidx)] = float(speed_series[i])

    for i, rgb in enumerate(frames):
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        box = box_by_idx.get(i)
        if box is not None:
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 220, 0), 2)
        spd = speed_by_idx.get(i)
        _draw_overlay(bgr, spd, gait, cadence)
        writer.append_data(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if progress and frames:
            progress(0.6 + 0.4 * (i + 1) / len(frames))

    writer.close()

    notes.append(
        "Speed is in body lengths per second (BL/s) — a calibration-free "
        "relative measure, not mph. To get true mph, add a known reference "
        "distance in the scene."
    )

    return AnalysisResult(
        fps=eff_fps,
        frame_count=processed,
        duration_s=processed / eff_fps if eff_fps else 0.0,
        frames_with_horse=len(main),
        peak_speed_bls=peak_speed,
        avg_speed_bls=avg_speed,
        cadence_per_min=cadence,
        gait=gait,
        smoothness=smoothness,
        speed_series=[float(v) for v in speed_series],
        annotated_video=output_path,
        notes=notes,
    )


def _draw_overlay(bgr: np.ndarray, speed: Optional[float], gait: str, cadence: float) -> None:
    h, w = bgr.shape[:2]
    panel_h = 70
    overlay = bgr.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, bgr, 0.55, 0, bgr)

    spd_txt = f"{speed:.2f} BL/s" if speed is not None else "-- BL/s"
    lines = [
        f"Speed: {spd_txt}",
        f"Gait: {gait}    Cadence: {cadence:.0f}/min",
    ]
    y = 28
    for line in lines:
        cv2.putText(bgr, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
        y += 28
