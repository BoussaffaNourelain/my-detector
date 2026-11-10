"""Module 2: my dataset as one YOLO folder with a fixed split, and the measurements to check it before trusting it.

Pixel boxes are [x1, y1, x2, y2]; YOLO boxes are (cx, cy, w, h) as fractions of the image. Images are RGB uint8
NumPy arrays (height, width, 3). See docs/data_spec.md.

Safety rules on top of the spec:
- never write into an output folder that already contains files (FileExistsError, nothing is changed);
- two images whose label files would share a name (same stem, e.g. a.jpg and a.png) stop the conversion;
- a box that is invalid or has zero area after clipping is never written; it is counted in
  skipped_objects as "invalid box: <class>".
- format yolo: a polygon (segmentation) label line is replaced by its enclosing box (polygon_to_yolo); every converted
  or dropped line is listed in <yolo folder>/label_conversions.csv and summarised by write_report.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

import cv2
import numpy as np
import yaml

from .classical import load_rgb

SPLITS = ("train", "val", "test")
FORMATS = ("voc", "coco", "yolo")
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp")
REVIEW_KEYS = ("images_without_label", "labels_without_image", "empty_labels", "out_of_range_boxes",
               "unknown_class_ids", "tiny_boxes_lt_0.1pct", "duplicates_across_splits")
PALETTE = [(255, 215, 0), (0, 200, 255), (255, 80, 80), (80, 220, 80), (200, 120, 255),
           (255, 150, 40), (0, 255, 180), (255, 100, 200), (160, 160, 255), (220, 220, 220)]   # RGB, one per class


# ------------------------------------------------------------------ project and raw data
def load_project(path: str | Path = "project.yaml") -> dict:
    """Read project.yaml; ValueError if name, classes, format or source is missing or wrong."""
    with Path(path).open(encoding="utf-8") as f:
        project = yaml.safe_load(f) or {}
    if not isinstance(project, dict):
        raise ValueError(f"{path}: expected a mapping of fields")
    if not project.get("name"):
        raise ValueError(f"{path}: 'name' is missing")
    classes = project.get("classes")
    if not isinstance(classes, list) or not classes:
        raise ValueError(f"{path}: 'classes' must be a non-empty list")
    if len(set(classes)) != len(classes):
        raise ValueError(f"{path}: 'classes' contains the same class twice: {classes}")
    if project.get("format") not in FORMATS:
        raise ValueError(f"{path}: 'format' must be one of {FORMATS}, got {project.get('format')!r}")
    source = project.get("source")
    if not isinstance(source, dict) or not (source.get("kaggle") or source.get("folder")):
        raise ValueError(f"{path}: 'source' needs 'kaggle' or 'folder'")
    return project


def download(project: dict) -> Path:
    """The folder of the raw dataset: the local folder if one is given, otherwise the Kaggle download."""
    source = project.get("source") or {}
    if source.get("folder"):
        folder = Path(source["folder"])
        if not folder.is_dir():
            raise FileNotFoundError(f"Dataset folder not found: {folder}")
        return folder
    import kagglehub                                            # only needed for Kaggle sources
    return Path(kagglehub.dataset_download(source["kaggle"]))


# ------------------------------------------------------------------ boxes
def pixels_to_yolo(box, width: float, height: float) -> tuple[float, float, float, float]:
    """[x1, y1, x2, y2] in pixels -> (cx, cy, w, h) as fractions. The box is clipped to the image first."""
    x1, y1, x2, y2 = (float(v) for v in box)
    x1, x2 = min(max(x1, 0.0), width), min(max(x2, 0.0), width)
    y1, y2 = min(max(y1, 0.0), height), min(max(y2, 0.0), height)
    return (x1 + x2) / 2 / width, (y1 + y2) / 2 / height, (x2 - x1) / width, (y2 - y1) / height


def yolo_to_pixels(cx: float, cy: float, w: float, h: float, width: float, height: float) -> list[float]:
    """(cx, cy, w, h) fractions -> [x1, y1, x2, y2] in pixels."""
    return [(cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height]


def _valid_yolo(cx: float, cy: float, w: float, h: float) -> bool:
    """A box we may write: finite, centre inside the image, width and height in (0, 1] at 6 decimals."""
    if not all(math.isfinite(v) for v in (cx, cy, w, h)):
        return False
    cx, cy, w, h = (round(v, 6) for v in (cx, cy, w, h))
    return 0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1


def polygon_to_yolo(coords) -> tuple[float, float, float, float] | None:
    """Normalised polygon x1 y1 x2 y2 ... (at least 3 points) -> its enclosing box (cx, cy, w, h).

    Points are clipped to the image (0-1) first, like pixels_to_yolo clips boxes. Returns None for an odd number of
    values, fewer than 3 points, a value that is not a finite number, or a box with zero width or height.
    """
    values = list(coords)
    if len(values) < 6 or len(values) % 2:
        return None
    try:
        values = [float(v) for v in values]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in values):
        return None
    xs = [min(max(v, 0.0), 1.0) for v in values[0::2]]
    ys = [min(max(v, 0.0), 1.0) for v in values[1::2]]
    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    box = ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
    return box if _valid_yolo(*box) else None


CONVERSIONS_FILE = "label_conversions.csv"
CONVERSION_FIELDS = ("split", "label_file", "line", "values", "action", "cx", "cy", "w", "h")


def _yolo_label_line(parts: list[str]) -> tuple[str, int | None, tuple | None]:
    """One label line -> (action, class_id, box). action: box | converted | malformed | invalid."""
    try:
        cid_value = float(parts[0])
        values = [float(v) for v in parts[1:]]
    except ValueError:
        return "malformed", None, None                              # a token that is not a number
    if not math.isfinite(cid_value) or cid_value != int(cid_value):
        return "malformed", None, None                              # class id must be a whole number
    cid = int(cid_value)
    if len(values) == 4:
        return ("box", cid, tuple(values)) if _valid_yolo(*values) else ("invalid", cid, None)
    if len(values) < 6 or len(values) % 2:
        return "malformed", cid, None                               # odd count or fewer than 3 points
    box = polygon_to_yolo(values)
    return ("converted", cid, box) if box else ("invalid", cid, None)


# ------------------------------------------------------------------ annotation readers
def _number(text: str | None) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return math.nan                                         # a missing value makes the box invalid later


def read_voc(xml_path: str | Path) -> dict:
    """One PASCAL VOC .xml -> {"filename", "width", "height", "objects": [(class_name, [x1, y1, x2, y2])]}."""
    root = ET.parse(xml_path).getroot()
    filename = Path((root.findtext("filename") or "").strip()).name or Path(xml_path).stem
    size = root.find("size")
    width = int(_number(size.findtext("width"))) if size is not None and math.isfinite(_number(size.findtext("width"))) else 0
    height = int(_number(size.findtext("height"))) if size is not None and math.isfinite(_number(size.findtext("height"))) else 0
    objects = []
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        box = [_number(bb.findtext(k)) if bb is not None else math.nan for k in ("xmin", "ymin", "xmax", "ymax")]
        objects.append(((obj.findtext("name") or "").strip(), box))
    return {"filename": filename, "width": width, "height": height, "objects": objects}


def read_coco(json_path: str | Path) -> list[dict]:
    """A COCO .json -> one record per image, shaped like read_voc. COCO [x, y, w, h] becomes corners."""
    with Path(json_path).open(encoding="utf-8") as f:
        coco = json.load(f)
    names = {c["id"]: c["name"] for c in coco.get("categories", [])}
    records = {img["id"]: {"filename": Path(img["file_name"]).name, "width": img.get("width", 0),
                           "height": img.get("height", 0), "objects": []} for img in coco.get("images", [])}
    for ann in coco.get("annotations", []):
        rec = records.get(ann["image_id"])
        if rec is None:
            continue                                            # annotation of an image the file does not list
        x, y, w, h = (float(v) for v in ann["bbox"])
        rec["objects"].append((names.get(ann["category_id"], str(ann["category_id"])), [x, y, x + w, y + h]))
    return list(records.values())


# ------------------------------------------------------------------ split and conversion
def split_names(names, seed: int = 42, ratios=(0.70, 0.15, 0.15)) -> dict[str, list]:
    """Sort, shuffle with random.Random(seed), cut at int(0.70 n) and int(0.85 n)."""
    names = sorted(names)
    random.Random(seed).shuffle(names)
    n = len(names)
    a = int(ratios[0] * n)
    b = int(round(ratios[0] + ratios[1], 10) * n)              # round: 0.70 + 0.15 must be exactly 0.85
    return {"train": names[:a], "val": names[a:b], "test": names[b:]}


def _check_output_is_free(out: Path) -> None:
    """Refuse to write into a folder that already contains files: nothing is overwritten or deleted."""
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f"{out} already exists and is not empty. Nothing was changed. "
                              "Delete or rename it yourself, or choose another output folder.")


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _images_in(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXT)


def _stem_conflicts(filenames) -> dict[str, list[str]]:
    """Names whose label files would collide: same stem (case-insensitive), or the same name listed twice."""
    by_stem = defaultdict(list)
    for name in filenames:
        by_stem[Path(name).stem.lower()].append(name)
    return {stem: names for stem, names in by_stem.items() if len(names) > 1}


def _image_size(path: Path) -> tuple[int, int]:
    """(width, height) read from the file; (0, 0) if it cannot be read."""
    bgr = cv2.imread(str(path))
    return (0, 0) if bgr is None else (bgr.shape[1], bgr.shape[0])


def convert_to_yolo(records, images_dir: str | Path, out: str | Path, classes: list[str], seed: int = 42) -> dict:
    """Split the records, copy each image and write its YOLO label file, then data.yaml.

    Returns {"images": {split: n}, "skipped_objects": Counter, "missing_images": n}. Objects of other classes are
    counted under their class name; invalid or zero-area boxes under "invalid box: <class>".
    """
    images_dir, out, records = Path(images_dir), Path(out), list(records)
    _check_output_is_free(out)

    conflicts = _stem_conflicts(r["filename"] for r in records)
    if conflicts:
        raise ValueError("These images would share a label file (same stem). Nothing was written: "
                         + "; ".join(f"{s}: {names}" for s, names in sorted(conflicts.items())))

    by_name, by_stem = defaultdict(list), defaultdict(list)    # every image anywhere under images_dir
    for p in sorted(images_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXT:
            by_name[p.name.lower()].append(p)
            by_stem[p.stem.lower()].append(p)

    found, ambiguous, missing = {}, {}, 0
    for rec in records:
        name = rec["filename"]
        candidates = by_name.get(name.lower()) or ([] if Path(name).suffix else by_stem.get(name.lower(), []))
        if not candidates:
            missing += 1
        elif len(candidates) > 1 and len({_md5(p) for p in candidates}) > 1:
            ambiguous[name] = [str(p) for p in candidates]      # different files with the same name: we cannot choose
        else:
            found[name] = candidates[0]                         # one file, or identical copies of it
    if ambiguous:
        raise ValueError("These names match different image files under the dataset folder. Nothing was written: "
                         + "; ".join(f"{n}: {paths}" for n, paths in sorted(ambiguous.items())))

    split = split_names(list(found), seed=seed)
    where = {name: s for s, names in split.items() for name in names}
    for s in SPLITS:
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)

    class_id = {c: i for i, c in enumerate(classes)}
    report = {"images": {s: 0 for s in SPLITS}, "skipped_objects": Counter(), "missing_images": missing}
    for rec in records:
        if rec["filename"] not in found:
            continue
        path, s = found[rec["filename"]], where[rec["filename"]]
        width, height = rec.get("width") or 0, rec.get("height") or 0
        if not (width > 0 and height > 0):
            width, height = _image_size(path)                   # the annotation has no usable size
            if not (width > 0 and height > 0):
                report["missing_images"] += 1                   # unreadable image: not copied
                continue
        lines = []
        for cls, box in rec["objects"]:
            if cls not in class_id:
                report["skipped_objects"][cls] += 1
                continue
            usable = len(box) == 4 and all(math.isfinite(v) for v in box) and box[2] > box[0] and box[3] > box[1]
            yb = pixels_to_yolo(box, width, height) if usable else (math.nan,) * 4
            if not _valid_yolo(*yb):
                report["skipped_objects"][f"invalid box: {cls}"] += 1
                continue
            lines.append(f"{class_id[cls]} " + " ".join(f"{v:.6f}" for v in yb))
        shutil.copy2(path, out / "images" / s / path.name)
        (out / "labels" / s / f"{path.stem}.txt").write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
        report["images"][s] += 1
    write_data_yaml(out, classes)
    return report


def write_data_yaml(root: str | Path, classes: list[str]) -> Path:
    """Write root/data.yaml: absolute path, the three image folders and the class names."""
    root = Path(root)
    content = {"path": str(root.resolve()), "train": "images/train", "val": "images/val", "test": "images/test",
               "names": {i: c for i, c in enumerate(classes)}}
    target = root / "data.yaml"
    target.write_text(yaml.safe_dump(content, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return target


def _copy_yolo(src: Path, out: Path, classes: list[str]) -> None:
    """A dataset already in YOLO format: copy its splits (images/<split> or <split>/images; 'valid' = val)."""
    aliases = {"train": ("train",), "val": ("val", "valid", "validation"), "test": ("test",)}
    layout = {}
    for s, names in aliases.items():
        for a in names:
            for img_dir, lbl_dir in ((src / "images" / a, src / "labels" / a), (src / a / "images", src / a / "labels")):
                if s not in layout and img_dir.is_dir():
                    layout[s] = (img_dir, lbl_dir)
    if not layout:
        raise ValueError(f"No YOLO split found in {src} (expected images/train or train/images)")

    conflicts = {s: c for s, (img_dir, _) in layout.items() if (c := _stem_conflicts(p.name for p in _images_in(img_dir)))}
    if conflicts:
        raise ValueError(f"Images in the same split share a stem, their labels would collide. Nothing was written: {conflicts}")

    src_yaml = src / "data.yaml"
    if src_yaml.exists():
        names = (yaml.safe_load(src_yaml.read_text(encoding="utf-8")) or {}).get("names")
        names = list(names.values()) if isinstance(names, dict) else names
        if names and list(names) != list(classes):
            print(f"WARNING: {src_yaml} names {names}, project.yaml classes {classes}: class ids are copied as they are.")

    log = []                                                         # every converted or dropped line
    for s in SPLITS:
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)
        if s not in layout:
            continue
        img_dir, lbl_dir = layout[s]
        for p in _images_in(img_dir):
            shutil.copy2(p, out / "images" / s / p.name)
        for lbl in sorted(lbl_dir.glob("*.txt")) if lbl_dir.is_dir() else []:
            kept = []
            for number, line in enumerate(lbl.read_text(encoding="utf-8").splitlines(), start=1):
                parts = line.split()
                if not parts:
                    continue
                action, cid, box = _yolo_label_line(parts)
                if action != "box":
                    log.append({"split": s, "label_file": lbl.name, "line": number, "values": len(parts) - 1,
                                "action": action, **dict(zip(("cx", "cy", "w", "h"),
                                                             (f"{v:.6f}" for v in box) if box else ("",) * 4))})
                if box:                                              # a box, or a polygon converted to its box
                    kept.append(f"{cid} " + " ".join(f"{v:.6f}" for v in box))
            (out / "labels" / s / lbl.name).write_text("".join(f"{k}\n" for k in kept), encoding="utf-8")

    with (out / CONVERSIONS_FILE).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CONVERSION_FIELDS)
        writer.writeheader()
        writer.writerows(log)
    actions = Counter(r["action"] for r in log)
    print(f"Label lines: {actions['converted']} polygon(s) converted to boxes; "
          f"{actions['malformed'] + actions['invalid']} dropped ({actions['malformed']} malformed, "
          f"{actions['invalid']} invalid). Details: {out / CONVERSIONS_FILE}")
    write_data_yaml(out, classes)


def prepare(project: dict, src: str | Path, out: str | Path | None = None) -> Path:
    """Raw dataset folder -> data/<name>_yolo (or out). Returns the root of the YOLO folder."""
    src = Path(src)
    out = Path(out) if out is not None else Path("data") / f"{project['name']}_yolo"
    _check_output_is_free(out)
    fmt, classes = project["format"], project["classes"]
    if fmt == "voc":
        xmls = sorted(p for p in src.rglob("*") if p.is_file() and p.suffix.lower() == ".xml")
        if not xmls:
            raise ValueError(f"No .xml annotation found under {src}")
        convert_to_yolo([read_voc(x) for x in xmls], src, out, classes)
    elif fmt == "coco":
        if not project.get("coco_json"):
            raise ValueError("format: coco needs 'coco_json' in project.yaml")
        convert_to_yolo(read_coco(src / project["coco_json"]), src, out, classes)
    elif fmt == "yolo":
        _copy_yolo(src, out, classes)
    else:
        raise ValueError(f"format must be one of {FORMATS}, got {fmt!r}")
    return out


# ------------------------------------------------------------------ measuring the YOLO folder
def read_labels(label_path: str | Path) -> list[tuple]:
    """A YOLO .txt -> [(class_id, cx, cy, w, h)]. An empty file gives []."""
    rows = []
    for number, line in enumerate(Path(label_path).read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(f"{label_path}, line {number}: expected 'class cx cy w h', got {line!r}")
        rows.append((int(float(parts[0])), *(float(v) for v in parts[1:])))
    return rows


def _labels_in(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.txt")) if folder.is_dir() else []


def class_counts(root: str | Path, classes: list[str]) -> dict[str, dict[str, int]]:
    """Boxes per class per split; every class is listed, even with 0 boxes."""
    root = Path(root)
    counts = {s: {c: 0 for c in classes} for s in SPLITS}
    for s in SPLITS:
        for lbl in _labels_in(root / "labels" / s):
            for cid, *_ in read_labels(lbl):
                if 0 <= cid < len(classes):
                    counts[s][classes[cid]] += 1
    return counts


def split_stats(root: str | Path) -> dict[str, dict]:
    """Per split: images, boxes, boxes_per_image, median_box_area_pct, small_boxes_lt_1pct."""
    root = Path(root)
    stats = {}
    for s in SPLITS:
        n_images = len(_images_in(root / "images" / s))
        areas = [w * h for lbl in _labels_in(root / "labels" / s) for _, _, _, w, h in read_labels(lbl)]
        stats[s] = {"images": n_images, "boxes": len(areas),
                    "boxes_per_image": round(len(areas) / n_images, 2) if n_images else 0.0,
                    "median_box_area_pct": round(median(areas) * 100, 2) if areas else 0.0,
                    "small_boxes_lt_1pct": sum(a < 0.01 for a in areas)}
    return stats


def review(root: str | Path, n_classes: int) -> dict:
    """Quality checks -> {"counts": {check: n}, "examples": {check: [file names]}}."""
    root = Path(root)
    examples = {k: [] for k in REVIEW_KEYS}
    seen = defaultdict(list)                                    # md5 -> ["split/name", ...]
    for s in SPLITS:
        images = {p.stem: p for p in _images_in(root / "images" / s)}
        labels = {p.stem: p for p in _labels_in(root / "labels" / s)}
        examples["images_without_label"] += [f"{s}/{images[k].name}" for k in sorted(images.keys() - labels.keys())]
        examples["labels_without_image"] += [f"{s}/{labels[k].name}" for k in sorted(labels.keys() - images.keys())]
        for stem, lbl in sorted(labels.items()):
            rows = read_labels(lbl)
            if not rows:
                examples["empty_labels"].append(f"{s}/{lbl.name}")
            for i, (cid, cx, cy, w, h) in enumerate(rows, start=1):
                where = f"{s}/{lbl.name} (box {i})"
                if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    examples["out_of_range_boxes"].append(where)
                if not 0 <= cid < n_classes:
                    examples["unknown_class_ids"].append(where)
                if w * h < 0.001:
                    examples["tiny_boxes_lt_0.1pct"].append(where)
        for p in images.values():
            seen[_md5(p)].append(f"{s}/{p.name}")
    for files in seen.values():
        if len({f.split("/")[0] for f in files}) > 1:
            examples["duplicates_across_splits"].append(" = ".join(sorted(files)))
    return {"counts": {k: len(v) for k, v in examples.items()}, "examples": examples}


# ------------------------------------------------------------------ my feature (docs/my_feature.md)
UNLABELLED_FIELDS = ("split", "image", "image_path", "label_path")


def unlabelled_images(root: str | Path, out: str | Path = "results/unlabelled_images.csv") -> dict:
    """Images whose YOLO label file exists but has no boxes, listed for review in a CSV. Read-only.

    Empty labels are not automatically errors (they may be intentional negative examples): nothing is changed or
    deleted. An image without any label file is not listed (review() reports it as images_without_label).
    Returns {"counts": {split: n, ..., "total": n}, "images": ["split/file", ...], "csv": Path}.
    """
    root, out = Path(root), Path(out)
    rows = []
    for s in SPLITS:
        for img in _images_in(root / "images" / s):                  # sorted by name
            label = root / "labels" / s / f"{img.stem}.txt"
            if label.is_file() and not read_labels(label):           # blank lines count as empty, as in review()
                rows.append({"split": s, "image": img.name,
                             "image_path": f"images/{s}/{img.name}", "label_path": f"labels/{s}/{label.name}"})
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print(f"Replacing the previous {out}")
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=UNLABELLED_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    counts = {s: sum(r["split"] == s for r in rows) for s in SPLITS}
    counts["total"] = len(rows)
    return {"counts": counts, "images": [f"{r['split']}/{r['image']}" for r in rows], "csv": out}


# ------------------------------------------------------------------ looking at it
def draw_boxes(rgb: np.ndarray, boxes, labels=None, color=(255, 215, 0), thickness: int = 2) -> np.ndarray:
    """Draw pixel boxes (and optional labels) on a copy of an RGB image; the input is never changed."""
    out = np.ascontiguousarray(rgb).copy()
    color = tuple(int(c) for c in color)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        if labels is not None and i < len(labels):
            cv2.putText(out, str(labels[i]), (x1, max(y1 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return out


def _label_path(image_path: Path) -> Path:
    """root/images/<split>/x.jpg -> root/labels/<split>/x.txt"""
    return image_path.parents[2] / "labels" / image_path.parent.name / f"{image_path.stem}.txt"


def _pixel_boxes(image_path: Path, width: int, height: int) -> list[tuple[int, list[float]]]:
    label = _label_path(image_path)
    if not label.exists():
        return []
    return [(cid, yolo_to_pixels(cx, cy, w, h, width, height)) for cid, cx, cy, w, h in read_labels(label)]


def boxes_for(image_path: str | Path) -> list[tuple[int, list[float]]]:
    """Ground truth of one image of the YOLO folder: [(class_id, [x1, y1, x2, y2])] in pixels."""
    image_path = Path(image_path)
    height, width = load_rgb(image_path).shape[:2]
    return _pixel_boxes(image_path, width, height)


def _annotate(rgb: np.ndarray, boxes, classes=None) -> np.ndarray:
    """Draw (class_id, box) pairs with one colour and the name of each class."""
    out = rgb
    for cid, box in boxes:
        name = classes[cid] if classes and 0 <= cid < len(classes) else str(cid)
        out = draw_boxes(out, [box], [name], color=PALETTE[cid % len(PALETTE)])
    return out if out is not rgb else rgb.copy()


def show(image_path: str | Path, classes=None) -> np.ndarray:
    """The image as RGB with its boxes and class names, one colour per class."""
    rgb = load_rgb(image_path)
    return _annotate(rgb, _pixel_boxes(Path(image_path), rgb.shape[1], rgb.shape[0]), classes)


def sample_grid(root: str | Path, split: str = "train", n: int = 20, seed: int = 0, classes=None,
                tile: tuple[int, int] = (256, 192)) -> np.ndarray:
    """n random images with their boxes, resized to tile (width, height), 5 per row, file name in each tile."""
    paths = _images_in(Path(root) / "images" / split)
    if not paths:
        raise FileNotFoundError(f"No images in {Path(root) / 'images' / split}")
    tw, th = tile
    tiles = []
    for p in random.Random(seed).sample(paths, min(n, len(paths))):
        rgb = load_rgb(p)
        sx, sy = tw / rgb.shape[1], th / rgb.shape[0]
        boxes = [(cid, [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]) for cid, b in _pixel_boxes(p, rgb.shape[1], rgb.shape[0])]
        img = _annotate(cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_AREA), boxes, classes)
        (w, h), _ = cv2.getTextSize(p.name, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(img, (0, 0), (w + 6, h + 8), (0, 0, 0), -1)
        cv2.putText(img, p.name, (3, h + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(img)
    rows = math.ceil(len(tiles) / 5)
    tiles += [np.zeros((th, tw, 3), np.uint8)] * (rows * 5 - len(tiles))    # black tiles fill the last row
    return np.vstack([np.hstack(tiles[r * 5:(r + 1) * 5]) for r in range(rows)])


def _conversion_section(root: Path) -> list[str]:
    """Report lines about label_conversions.csv (written by prepare for format yolo); none if the file is absent."""
    path = root / CONVERSIONS_FILE
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    lines = ["", "## Label conversions", "",
             "When the dataset was prepared, each polygon (segmentation) label line was replaced by its enclosing box: "
             "the minimum and maximum x and y of its points, clipped to 0-1. Lines that could not be used were dropped. "
             f"Every such line is listed in `{CONVERSIONS_FILE}` in the dataset folder; the raw dataset is unchanged.", ""]
    if not rows:
        return lines + ["No label lines were converted or dropped."]
    lines += ["| split | converted | dropped: malformed | dropped: invalid |", "|---|---:|---:|---:|"]
    for s in SPLITS:
        c = Counter(r["action"] for r in rows if r["split"] == s)
        lines.append(f"| {s} | {c['converted']} | {c['malformed']} | {c['invalid']} |")
    total = Counter(r["action"] for r in rows)
    lines.append(f"| total | {total['converted']} | {total['malformed']} | {total['invalid']} |")
    lines += ["", "First examples: " + ", ".join(f"{r['split']}/{r['label_file']} line {r['line']} "
                                                f"({r['values']} values, {r['action']})" for r in rows[:5])]
    return lines


def write_report(root: str | Path, classes: list[str], out: str | Path = "results/dataset_review.md") -> Path:
    """Markdown report (boxes per class, split statistics, quality checks) and dataset_sample_grid.jpg next to it."""
    root, out = Path(root), Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    grid_path = out.parent / "dataset_sample_grid.jpg"
    for existing in (out, grid_path):
        if existing.exists():
            print(f"Replacing the previous {existing}")
    cc, st, rv = class_counts(root, classes), split_stats(root), review(root, len(classes))

    lines = ["# Dataset review", "", f"Dataset: `{root}`", "", "## Boxes per class", "",
             "| name | train | val | test | total |", "|---|---:|---:|---:|---:|"]
    lines += [f"| {c} | {cc['train'][c]} | {cc['val'][c]} | {cc['test'][c]} | {sum(cc[s][c] for s in SPLITS)} |"
              for c in classes]
    lines += ["", "## Split statistics", "",
              "| split | images | boxes | boxes per image | median box area (% of image) | boxes < 1% of image |",
              "|---|---:|---:|---:|---:|---:|"]
    lines += [f"| {s} | {st[s]['images']} | {st[s]['boxes']} | {st[s]['boxes_per_image']} | "
              f"{st[s]['median_box_area_pct']} | {st[s]['small_boxes_lt_1pct']} |" for s in SPLITS]
    lines += ["", "## Quality checks", "", "| check | count | first examples |", "|---|---:|---|"]
    lines += [f"| {k} | {rv['counts'][k]} | {', '.join(rv['examples'][k][:5])} |" for k in REVIEW_KEYS]
    lines += _conversion_section(root)

    if _images_in(root / "images" / "train"):
        grid = sample_grid(root, "train", n=20, seed=0, classes=classes)
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError("Could not encode the sample grid as JPEG")
        grid_path.write_bytes(buf.tobytes())
        lines += ["", "## Sample grid", "", "20 training images (seed 0):", "", f"![sample grid]({grid_path.name})"]
    else:
        lines += ["", "## Sample grid", "", "No training images: no grid."]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
