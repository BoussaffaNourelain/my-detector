"""Module 4: run a trained detector on a photo, keep the boxes above a confidence, and check which phases are ready.

See docs/detect_spec.md. A prediction is {"box": [x1, y1, x2, y2], "score": 0.87, "label": "pothole"}, box in pixels.
The model runs once per photo (predict, at conf=0.001); moving the confidence only filters the result (keep).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import train as tr

BOX_COLOR = (255, 215, 0)                                    # RGB, the same yellow as the data page

_MODELS: dict[str, object] = {}                              # weights path -> loaded YOLO model


# ---------------------------------------------------------------- the runs of page 2

def list_runs(runs="runs") -> list[dict]:
    """One dict per finished run (runs/<name>/weights/best.pt exists), sorted by name."""
    runs = Path(runs)
    if not runs.is_dir():
        return []
    found = []
    for run_dir in sorted((d for d in runs.iterdir() if d.is_dir()), key=lambda d: d.name):
        weights = tr.best_weights(run_dir)
        if weights is None:
            continue
        summary = tr.summarize(tr.read_progress(run_dir))
        why_file = runs / f"{run_dir.name}_why.txt"
        found.append({"name": run_dir.name, "weights": str(weights),
                      "mAP50": summary["best_mAP50"] if summary else None,
                      "why": why_file.read_text(encoding="utf-8") if why_file.is_file() else ""})
    return found


def load_model(weights):
    """An Ultralytics YOLO model, loaded once per weights path."""
    key = str(Path(weights).resolve())
    if key not in _MODELS:
        from ultralytics import YOLO                         # imported here: importing this module stays fast
        _MODELS[key] = YOLO(key)
    return _MODELS[key]


# ---------------------------------------------------------------- predicting

def to_preds(result, names) -> list[dict]:
    """One Ultralytics result -> predictions from the highest score to the lowest."""
    boxes = result.boxes
    preds = [{"box": [round(float(v), 1) for v in box], "score": round(float(score), 3), "label": names[int(c)]}
             for box, score, c in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist())]
    return sorted(preds, key=lambda p: p["score"], reverse=True)


def predict(weights, rgb, imgsz=416) -> list[dict]:
    """Run the model once on an RGB image (given to Ultralytics as BGR) and return every box with score >= 0.001."""
    model = load_model(weights)
    bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)   # a new array: rgb is not changed
    results = model(bgr, imgsz=int(imgsz), conf=0.001, verbose=False)
    return to_preds(results[0], model.names)


def keep(preds, conf) -> list[dict]:
    """The predictions with score >= conf, in the same order."""
    return [p for p in preds if p["score"] >= conf]


# ---------------------------------------------------------------- describing and drawing

def count_by_label(preds, classes) -> dict[str, int]:
    """Every class in `classes`, in that order, with its number of predictions."""
    counts = {c: 0 for c in classes}
    for p in preds:
        if p["label"] in counts:
            counts[p["label"]] += 1
    return counts


def describe(preds, conf, classes) -> str:
    """One sentence for the page, with the confidence at 2 decimals."""
    parts = [f"{n} {c}" for c, n in count_by_label(preds, classes).items() if n > 0]
    if not parts:
        return f"Nothing found at confidence {conf:.2f}."
    return f"At confidence {conf:.2f}: {', '.join(parts)}."


def draw(rgb, preds, color=BOX_COLOR) -> np.ndarray:
    """A new image with each box and its label and score (for example 'pothole 0.87')."""
    out = np.ascontiguousarray(rgb).copy()
    thick = max(1, round(min(out.shape[:2]) / 300))
    scale = max(0.4, min(out.shape[:2]) / 1000)
    for p in preds:
        x1, y1, x2, y2 = (int(round(v)) for v in p["box"])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
        cv2.putText(out, f"{p['label']} {p['score']:.2f}", (x1, max(y1 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------- the system

def default_conf(project, fallback=0.25) -> float:
    """The confidence threshold chosen in Assignment 3.2 (project["threshold"]), or fallback."""
    return float(project.get("threshold", fallback))


def system_check(project, runs="runs", results="results") -> dict[str, bool]:
    """Which phases are ready. data/<name>_yolo/images is looked for next to the runs folder
    (with the default runs="runs" that is the working directory, as in the spec)."""
    data = Path(runs).parent / "data" / f"{project['name']}_yolo" / "images"
    return {"data": data.is_dir(), "training": bool(list_runs(runs)),
            "results": (Path(results) / "thresholds.png").is_file()}
