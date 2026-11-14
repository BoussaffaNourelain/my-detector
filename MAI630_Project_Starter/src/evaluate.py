"""Module 3: evaluate a trained detector on the test split and look at its mistakes.

See docs/eval_spec.md. Boxes are always [x1, y1, x2, y2] in pixels.
A prediction is {"box", "score", "label"}; a ground-truth box is {"box", "label"}.
The dataset's data.yaml is only read, never changed.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp")

GREEN, RED, YELLOW = (0, 255, 0), (255, 0, 0), (255, 255, 0)     # RGB: the image given to draw_result is RGB


# ---------------------------------------------------------------- comparing boxes

def iou(a, b) -> float:
    """Intersection over union of two boxes, from 0 to 1. Boxes that do not touch give 0."""
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(preds, truths, iou_threshold=0.5):
    """Match one image's predictions to its ground truth -> {"tp": [...], "fp": [...], "fn": [...]}.

    Predictions go from the highest score to the lowest; each takes the free ground-truth box of the same label
    with the highest IoU, if that IoU reaches the threshold.
    """
    free = list(range(len(truths)))
    tp, fp = [], []
    for p in sorted(preds, key=lambda p: p["score"], reverse=True):
        best, best_iou = None, 0.0
        for i in free:
            if truths[i]["label"] != p["label"]:
                continue
            v = iou(p["box"], truths[i]["box"])
            if v > best_iou:
                best, best_iou = i, v
        if best is not None and best_iou >= iou_threshold:
            free.remove(best)
            tp.append(p)
        else:
            fp.append(p)
    return {"tp": tp, "fp": fp, "fn": [truths[i] for i in free]}


def precision_recall(tp, fp, fn):
    """(precision, recall) from three counts; dividing by zero gives 0."""
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    return precision, recall


# ---------------------------------------------------------------- at a confidence threshold

def _kept(preds, conf):
    return [p for p in preds if p["score"] >= conf]


def at_threshold(images, conf, iou_threshold=0.5):
    """Counts over all images, keeping predictions with score >= conf -> {conf, tp, fp, fn, precision, recall}."""
    tp = fp = fn = 0
    for d in images.values():
        r = match(_kept(d["preds"], conf), d["truths"], iou_threshold)
        tp, fp, fn = tp + len(r["tp"]), fp + len(r["fp"]), fn + len(r["fn"])
    precision, recall = precision_recall(tp, fp, fn)
    return {"conf": conf, "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall}


def threshold_table(images, confs=(0.1, 0.25, 0.5, 0.75)):
    """One at_threshold row per confidence value, in that order."""
    return [at_threshold(images, c) for c in confs]


def what_changed(images, conf_low, conf_high, iou_threshold=0.5):
    """Going from conf_low to conf_high: how many correct and how many wrong detections disappear."""
    removed_correct = removed_wrong = 0
    for d in images.values():
        r = match(_kept(d["preds"], conf_low), d["truths"], iou_threshold)
        removed_correct += sum(p["score"] < conf_high for p in r["tp"])
        removed_wrong += sum(p["score"] < conf_high for p in r["fp"])
    return {"removed_correct": removed_correct, "removed_wrong": removed_wrong}


def pick_examples(images, conf=0.25, n=3):
    """{"correct": [...], "wrong": [...]}, up to n image names each.

    correct: images with TP and no FP or FN, most TP first. wrong: images with the most FP + FN.
    Ties are broken by name so the choice is always the same.
    """
    correct, wrong = [], []
    for name, d in images.items():
        r = match(_kept(d["preds"], conf), d["truths"])
        tp, errors = len(r["tp"]), len(r["fp"]) + len(r["fn"])
        if tp and not errors:
            correct.append((-tp, name))
        elif errors:
            wrong.append((-errors, name))
    return {"correct": [name for _, name in sorted(correct)[:n]], "wrong": [name for _, name in sorted(wrong)[:n]]}


def draw_result(rgb, result):
    """A new image from match's result: TP green, FP red, FN yellow, with the score on each prediction."""
    out = np.ascontiguousarray(rgb).copy()
    thick = max(1, round(max(out.shape[:2]) / 300))
    scale = max(0.3, max(out.shape[:2]) / 1000)

    def box(b, color, text=None):
        x1, y1, x2, y2 = (int(round(v)) for v in b)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
        if text:
            cv2.putText(out, text, (x1, max(y1 - 3, 10)), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

    for t in result["fn"]:
        box(t["box"], YELLOW)
    for p in result["fp"]:
        box(p["box"], RED, f"{p['score']:.2f}")
    for p in result["tp"]:
        box(p["box"], GREEN, f"{p['score']:.2f}")
    return out


# ---------------------------------------------------------------- running the model

def _images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXT)


def _truths(label: Path, classes, width, height):
    """YOLO label file (class cx cy w h, relative) -> ground-truth boxes in pixels; a missing file means no boxes."""
    if not label.is_file():
        return []
    out = []
    for line in label.read_text().splitlines():
        t = line.split()
        if len(t) < 5:
            continue
        c = int(float(t[0]))
        cx, cy, w, h = (float(v) for v in t[1:5])
        out.append({"box": [(cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height],
                    "label": classes[c] if 0 <= c < len(classes) else str(c)})
    return out


def predict_images(weights, image_dir, label_dir, classes, imgsz=416):
    """Run the model with conf=0.001 over every image of a folder -> {name: {"preds": [...], "truths": [...]}}."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    label_dir = Path(label_dir)
    images = {}
    for path in _images(Path(image_dir)):
        r = next(iter(model.predict(str(path), conf=0.001, imgsz=imgsz, verbose=False)))
        height, width = r.orig_shape
        preds = [{"box": [float(v) for v in xyxy], "score": float(s), "label": model.names[int(c)]}
                 for xyxy, s, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]
        images[path.name] = {"preds": preds, "truths": _truths(label_dir / f"{path.stem}.txt", classes, width, height)}
    return images


def _portable_yaml(yaml_path: Path, out_dir: Path) -> Path:
    """yaml_path itself if its `path` exists here; otherwise a copy in out_dir whose `path` is the yaml's own folder.

    data.yaml stores the absolute path of the machine that wrote it (e.g. C:\\... on Windows), which does not exist
    on Linux or Colab. The original file is never changed.
    """
    cfg = yaml.safe_load(Path(yaml_path).read_text()) or {}
    base = cfg.get("path")
    if base and Path(base).is_dir():
        return Path(yaml_path)
    cfg["path"] = str(Path(yaml_path).resolve().parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "data_portable.yaml"
    tmp.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return tmp


def yolo_metrics(weights, yaml_path, imgsz=416, split="test"):
    """Ultralytics val on that split -> {"mAP50", "mAP50_95", "per_class": {name: mAP50}}.

    Its output folder goes to runs/eval_<split>, next to the weights' run, which Git ignores.
    """
    from ultralytics import YOLO

    weights = Path(weights).resolve()
    runs = weights.parents[2] if len(weights.parents) > 2 and weights.parent.name == "weights" else Path("runs").resolve()
    data = _portable_yaml(Path(yaml_path), runs / f"eval_{split}")
    model = YOLO(str(weights))
    m = model.val(data=str(data), split=split, imgsz=imgsz, plots=False, verbose=False,
                  project=str(runs), name=f"eval_{split}", exist_ok=True)
    per_class = {model.names[int(c)]: float(ap) for c, ap in zip(m.box.ap_class_index, m.box.ap50)}
    return {"mAP50": float(m.box.map50), "mAP50_95": float(m.box.map), "per_class": per_class}
