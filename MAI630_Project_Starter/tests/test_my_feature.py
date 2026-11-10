"""My feature (docs/my_feature.md): list the images whose YOLO label file is empty, in a CSV, without changing anything."""
import csv
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import data as dt

BOX = "0 0.500000 0.500000 0.200000 0.200000\n"


def _img(path: Path, value=100):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((20, 20, 3), value, np.uint8))


def _label(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _dataset(root: Path) -> Path:
    """train: a (empty), b (whitespace only), d (boxes), e (no label file); val: c (empty); test: f (boxes);
    plus an empty label with no image (train/orphan.txt)."""
    for split, name, value in (("train", "a", 10), ("train", "b", 20), ("train", "d", 30), ("train", "e", 40),
                               ("val", "c", 50), ("test", "f", 60)):
        _img(root / "images" / split / f"{name}.jpg", value)
    _label(root / "labels" / "train" / "a.txt", "")
    _label(root / "labels" / "train" / "b.txt", "  \n\n")
    _label(root / "labels" / "train" / "d.txt", BOX)
    _label(root / "labels" / "val" / "c.txt", "")
    _label(root / "labels" / "test" / "f.txt", BOX)
    _label(root / "labels" / "train" / "orphan.txt", "")
    return root


def _md5_tree(root: Path) -> dict:
    return {str(p.relative_to(root)): hashlib.md5(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file()}


def test_finds_exactly_the_empty_labels(tmp_path):
    rep = dt.unlabelled_images(_dataset(tmp_path / "ds"), tmp_path / "results" / "unlabelled_images.csv")
    assert rep["images"] == ["train/a.jpg", "train/b.jpg", "val/c.jpg"]          # not d (boxes), e (no label), orphan
    assert rep["counts"] == {"train": 2, "val": 1, "test": 0, "total": 3}


def test_csv_has_the_planned_format(tmp_path):
    out = tmp_path / "results" / "unlabelled_images.csv"
    rep = dt.unlabelled_images(_dataset(tmp_path / "ds"), out)
    assert rep["csv"] == out
    assert out.read_text(encoding="utf-8").splitlines()[0] == "split,image,image_path,label_path"
    with out.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows == [
        {"split": "train", "image": "a.jpg", "image_path": "images/train/a.jpg", "label_path": "labels/train/a.txt"},
        {"split": "train", "image": "b.jpg", "image_path": "images/train/b.jpg", "label_path": "labels/train/b.txt"},
        {"split": "val", "image": "c.jpg", "image_path": "images/val/c.jpg", "label_path": "labels/val/c.txt"},
    ]


def test_dataset_is_not_changed(tmp_path):
    root = _dataset(tmp_path / "ds")
    before = _md5_tree(root)
    dt.unlabelled_images(root, tmp_path / "results" / "unlabelled_images.csv")
    assert _md5_tree(root) == before


def test_agrees_with_review_on_a_clean_dataset(tmp_path):
    root = _dataset(tmp_path / "ds")
    (root / "labels" / "train" / "orphan.txt").unlink()                        # every label now has its image
    rep = dt.unlabelled_images(root, tmp_path / "results" / "unlabelled_images.csv")
    assert rep["counts"]["total"] == dt.review(root, 1)["counts"]["empty_labels"] == 3


def test_no_empty_labels_gives_a_header_only_csv(tmp_path):
    root = tmp_path / "ds"
    _img(root / "images" / "train" / "x.jpg")
    _label(root / "labels" / "train" / "x.txt", BOX)
    out = tmp_path / "results" / "unlabelled_images.csv"
    rep = dt.unlabelled_images(root, out)
    assert rep["counts"]["total"] == 0 and rep["images"] == []
    assert out.read_text(encoding="utf-8").splitlines() == ["split,image,image_path,label_path"]


def test_string_paths_and_missing_output_folder(tmp_path):
    out = tmp_path / "new" / "deeper" / "unlabelled_images.csv"
    rep = dt.unlabelled_images(str(_dataset(tmp_path / "ds")), str(out))
    assert out.is_file() and rep["counts"]["total"] == 3
