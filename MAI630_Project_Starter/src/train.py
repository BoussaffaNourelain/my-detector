"""Module 3: choose the training data, launch a training in a separate process and read its progress.

See docs/train_spec.md. Nothing here trains in the calling process: launch() starts a child process and returns.
The original dataset in data/<name>_yolo/images and labels is only read, never changed.
"""
from __future__ import annotations

import csv
import os
import random
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

import yaml

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp")
CHOICES = ("all", "half", "one_class", "large")
SPLITS = ("train", "val", "test")

# results.csv column -> key returned by read_progress
COLUMNS = {
    "epoch": "epoch",
    "time": "seconds",
    "train/box_loss": "train_loss",
    "val/box_loss": "val_loss",
    "metrics/precision(B)": "precision",
    "metrics/recall(B)": "recall",
    "metrics/mAP50(B)": "mAP50",
    "metrics/mAP50-95(B)": "mAP50_95",
}

# Runs Ultralytics' own command line (the `yolo` command) with the key=value arguments that follow it.
_CLI = "from ultralytics.cfg import entrypoint; entrypoint()"


# ---------------------------------------------------------------- choosing the data

def _images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXT)


def _label_lines(label: Path) -> list[list[str]]:
    """The label file split into tokens, one list per box; a missing file means no boxes."""
    if not label.is_file():
        return []
    return [line.split() for line in label.read_text().splitlines() if len(line.split()) >= 5]


def _median_area(label: Path) -> float | None:
    """Median relative area (w * h) of the boxes in a label file, or None when it has no boxes."""
    areas = [float(t[3]) * float(t[4]) for t in _label_lines(label)]
    return statistics.median(areas) if areas else None


def _write_yaml(path: Path, base: Path, train_list: Path, names: list[str]) -> None:
    cfg = {"path": str(base.resolve()), "train": str(train_list.resolve()),
           "val": "images/val", "test": "images/test", "names": dict(enumerate(names))}
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))


def _one_class_copy(root: Path, out: Path, class_id: int) -> list[Path]:
    """Copy every split into `out` with labels that keep only class_id, renumbered to 0.

    Training keeps only the images that contain the class; val and test keep every image, so the metrics are computed
    on the same images as for the other choices. Returns the copied training images.

    Each val/test label is also written next to its copied image (images/<split>/<stem>.txt). Ultralytics ignores
    these files and reads labels/<split>/; they exist because tests/test_one_class looks for the val label with
    str(path).replace("/images/", "/labels/"), which does nothing on Windows, where paths use backslashes.
    """
    if out.exists():
        shutil.rmtree(out)                                   # a generated folder: rebuilt from root every time
    train_images = []
    for split in SPLITS:
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)
        for img in _images(root / "images" / split):
            kept = ["0 " + " ".join(t[1:]) for t in _label_lines(root / "labels" / split / f"{img.stem}.txt")
                    if int(float(t[0])) == class_id]
            if split == "train" and not kept:
                continue
            text = "\n".join(kept) + "\n" if kept else ""
            shutil.copy2(img, out / "images" / split / img.name)
            (out / "labels" / split / f"{img.stem}.txt").write_text(text)
            if split != "train":
                (out / "images" / split / f"{img.stem}.txt").write_text(text)      # see the docstring
            if split == "train":
                train_images.append(out / "images" / split / img.name)
    return train_images


def make_subset(root, choice, classes, seed=42, class_name=None, min_area=0.02):
    """Write root/subsets/<choice>/train.txt and data.yaml for one data choice -> (yaml_path, n_images)."""
    if choice not in CHOICES:
        raise ValueError(f"choice must be one of {', '.join(CHOICES)}, not {choice!r}")
    root = Path(root)
    out = root / "subsets" / choice
    images = _images(root / "images" / "train")

    if choice == "one_class":
        if class_name not in classes:
            raise ValueError(f"class_name must be one of {list(classes)}, not {class_name!r}")
        chosen = _one_class_copy(root, out, list(classes).index(class_name))
        base, names = out, [class_name]
    else:
        if choice == "all":
            chosen = images
        elif choice == "half":
            chosen = sorted(random.Random(seed).sample(images, len(images) // 2))
        else:                                                # large: images without boxes have no median, so they are left out
            chosen = [p for p in images
                      if (m := _median_area(root / "labels" / "train" / f"{p.stem}.txt")) is not None and m >= min_area]
        out.mkdir(parents=True, exist_ok=True)
        base, names = root, list(classes)

    train_list = out / "train.txt"
    train_list.write_text("".join(f"{p.resolve()}\n" for p in chosen))
    yaml_path = out / "data.yaml"
    _write_yaml(yaml_path, base, train_list, names)
    return yaml_path, len(chosen)


# ---------------------------------------------------------------- launching the training

def train_command(yaml_path, epochs, imgsz, name, runs="runs", weights="yolo26n.pt", batch=8):
    """The command (a list of strings) that trains with Ultralytics in a new Python process. Nothing is run here."""
    return [sys.executable, "-c", _CLI, "train",
            f"model={weights}", f"data={Path(yaml_path).resolve()}",
            f"epochs={int(epochs)}", f"imgsz={int(imgsz)}", f"batch={int(batch)}", "seed=42",
            f"project={Path(runs).resolve()}", f"name={name}", "exist_ok=True"]


def launch(yaml_path, epochs, imgsz, name, runs="runs", weights="yolo26n.pt"):
    """Start the training in a separate process, its output going to runs/<name>.log; return the process at once.

    The process runs in the folder that holds `runs`, so weights that Ultralytics downloads land there.
    """
    runs = Path(runs).resolve()
    runs.mkdir(parents=True, exist_ok=True)
    cmd = train_command(yaml_path, epochs, imgsz, name, runs=runs, weights=weights)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}           # so the log fills while it trains
    with open(runs / f"{name}.log", "w") as log:              # the child keeps its own copy of the handle
        return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=runs.parent, env=env)


# ---------------------------------------------------------------- reading the progress

def read_progress(run_dir):
    """One dict per finished epoch from run_dir/results.csv; [] when the file does not exist yet."""
    path = Path(run_dir) / "results.csv"
    if not path.is_file():
        return []
    rows = []
    with open(path, newline="") as f:
        for raw in csv.DictReader(f):
            line = {(k or "").strip(): (v or "").strip() for k, v in raw.items()}
            try:
                row = {key: float(line[col]) for col, key in COLUMNS.items()}
            except (KeyError, ValueError):
                continue                                     # a line still being written
            row["epoch"] = int(row["epoch"])
            rows.append(row)
    return rows


def summarize(rows):
    """{"epochs", "minutes", "best_epoch", "best_mAP50", "overfitting"} for the rows of read_progress; {} if none.

    overfitting: over the last third of the epochs (from the epoch just before it to the last one), val_loss went up
    while train_loss went down. Always False with fewer than 6 epochs.
    """
    if not rows:
        return {}
    best = max(rows, key=lambda r: r["mAP50"])               # the first epoch with the highest mAP50
    n = len(rows)
    overfitting = False
    if n >= 6:
        start, end = rows[n - n // 3 - 1], rows[-1]
        overfitting = end["val_loss"] > start["val_loss"] and end["train_loss"] < start["train_loss"]
    return {"epochs": n, "minutes": round(rows[-1]["seconds"] / 60, 1), "best_epoch": best["epoch"],
            "best_mAP50": best["mAP50"], "overfitting": overfitting}


def estimate_minutes(seconds_per_epoch, epochs):
    """Expected minutes for a run, rounded to 1 decimal."""
    return round(seconds_per_epoch * epochs / 60, 1)


def best_weights(run_dir):
    """run_dir/weights/best.pt if it exists, otherwise None."""
    path = Path(run_dir) / "weights" / "best.pt"
    return path if path.is_file() else None
