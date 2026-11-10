"""Polygon (segmentation) YOLO label lines become their enclosing box when a YOLO dataset is prepared."""
import csv
import hashlib
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import data as dt

CLASSES = ["plastic-bottle"]


def _approx(box, expected):
    return box is not None and all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip(box, expected))


# ------------------------------------------------------------------ polygon_to_yolo
def test_square_polygon_gives_its_box():
    square = [0.2, 0.3, 0.6, 0.3, 0.6, 0.7, 0.2, 0.7]
    assert _approx(dt.polygon_to_yolo(square), (0.4, 0.5, 0.4, 0.4))


def test_triangle_and_point_order_do_not_matter():
    tri = [(0.1, 0.2), (0.5, 0.9), (0.8, 0.4)]
    expected = (0.45, 0.55, 0.7, 0.7)
    for order in (tri, tri[::-1], tri[1:] + tri[:1]):
        assert _approx(dt.polygon_to_yolo([v for p in order for v in p]), expected)


def test_points_on_the_image_border_are_valid():
    assert _approx(dt.polygon_to_yolo([0, 0, 1, 0, 1, 1, 0, 1]), (0.5, 0.5, 1.0, 1.0))


def test_points_slightly_outside_are_clipped():
    box = dt.polygon_to_yolo([-0.02, 0.1, 1.03, 0.1, 0.5, 0.6])
    assert _approx(box, (0.5, 0.35, 1.0, 0.5))
    cx, cy, w, h = box
    assert 0 <= cx - w / 2 and cx + w / 2 <= 1 and 0 <= cy - h / 2 and cy + h / 2 <= 1


@pytest.mark.parametrize("coords", [
    [0.1, 0.2, 0.3, 0.4, 0.5],                       # odd number of values
    [0.1, 0.2, 0.3, 0.4],                            # only 2 points
    [],                                              # nothing
    [0.1, 0.2, float("nan"), 0.4, 0.5, 0.6],         # not finite
    [0.1, 0.2, float("inf"), 0.4, 0.5, 0.6],
    [0.1, 0.5, 0.4, 0.5, 0.9, 0.5],                  # all on one horizontal line: zero height
    [0.3, 0.1, 0.3, 0.5, 0.3, 0.9],                  # all on one vertical line: zero width
    [1.2, 0.1, 1.5, 0.4, 1.9, 0.8],                  # all outside the image
    ["a", 0.2, 0.3, 0.4, 0.5, 0.6],                  # not a number
])
def test_invalid_polygons_give_none(coords):
    assert dt.polygon_to_yolo(coords) is None


# ------------------------------------------------------------------ prepare (format yolo) and the record
def _yolo_raw(root: Path) -> Path:
    for split, n in (("train", 2), ("valid", 1)):
        for i in range(n):
            img = root / split / "images" / f"{split}{i}.jpg"
            img.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(img), np.full((40, 40, 3), 60 + 40 * i, np.uint8))
            (root / split / "labels").mkdir(parents=True, exist_ok=True)
            (root / split / "labels" / f"{split}{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (root / "train" / "labels" / "train0.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n"                         # line 1: a box, kept as it is
        "0 0.2 0.3 0.6 0.3 0.6 0.7 0.2 0.7\n"        # line 2: a polygon -> converted
        "0 0.1 0.2 0.3\n"                            # line 3: malformed (3 values)
        "0.5 0.5 0.5 0.2 0.2\n")                     # line 4: class id not a whole number -> malformed
    (root / "data.yaml").write_text("names: ['plastic-bottle']\n")
    return root


def _md5_tree(root: Path) -> dict:
    return {str(p.relative_to(root)): hashlib.md5(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file()}


def test_prepare_converts_polygons_and_records_them(tmp_path):
    raw = _yolo_raw(tmp_path / "raw")
    before = _md5_tree(raw)
    root = dt.prepare({"name": "pb", "classes": CLASSES, "format": "yolo", "source": {"folder": str(raw)}},
                      raw, tmp_path / "pb_yolo")
    assert _md5_tree(raw) == before                                        # raw dataset untouched
    assert [len(list((root / "images" / s).glob("*.jpg"))) for s in dt.SPLITS] == [2, 1, 0]   # split kept, valid -> val
    lines = (root / "labels" / "train" / "train0.txt").read_text().splitlines()
    assert lines == ["0 0.500000 0.500000 0.200000 0.200000", "0 0.400000 0.500000 0.400000 0.400000"]
    with (root / dt.CONVERSIONS_FILE).open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert [(r["split"], r["label_file"], r["line"], r["values"], r["action"]) for r in rows] == [
        ("train", "train0.txt", "2", "8", "converted"),
        ("train", "train0.txt", "3", "3", "malformed"),
        ("train", "train0.txt", "4", "4", "malformed")]
    assert rows[0]["cx"] == "0.400000" and rows[1]["cx"] == ""
    assert dt.review(root, 1)["counts"]["out_of_range_boxes"] == 0           # every written line is a valid box


def test_report_has_the_label_conversion_section(tmp_path):
    raw = _yolo_raw(tmp_path / "raw")
    root = dt.prepare({"name": "pb", "classes": CLASSES, "format": "yolo", "source": {"folder": str(raw)}},
                      raw, tmp_path / "pb_yolo")
    text = dt.write_report(root, CLASSES, tmp_path / "results" / "dataset_review.md").read_text(encoding="utf-8")
    assert "## Label conversions" in text and dt.CONVERSIONS_FILE in text
    assert "| train | 1 | 2 | 0 |" in text and "| total | 1 | 2 | 0 |" in text
