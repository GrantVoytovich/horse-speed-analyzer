"""Smoke test: run the pipeline on the downloaded clips and print metrics."""
import json
import time
from app.vision import analyze_video

CLIPS = [
    ("data/test_clips/walk.mp4", "data/outputs/walk_annotated.mp4"),
    ("data/test_clips/run.mp4", "data/outputs/run_annotated.mp4"),
]

for src, out in CLIPS:
    print(f"\n=== {src} ===", flush=True)
    t0 = time.time()
    try:
        res = analyze_video(src, out, max_frames=600)
        print(json.dumps(res.to_dict(), indent=2))
        print(f"elapsed: {time.time()-t0:.1f}s -> {out}")
    except Exception as e:
        print(f"ERROR: {e}")
print("\nALL_DONE")
