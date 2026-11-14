"""The data page's image browser: a slider 0 .. n-1 per split, and the events that drive it (docs/data_spec.md)."""
import sys
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import data as dt
from app import app as myapp

CLASSES = ["cat", "dog"]
LABEL = "Image number (0 to n-1)"


def _img(path: Path, w=100, h=50, value=128):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((h, w, 3), value, np.uint8))


@pytest.fixture
def project(tmp_path, monkeypatch):
    """20 VOC images prepared into tmp/data/pets_yolo (split 14 / 3 / 3) with tmp/project.yaml next to it."""
    raw = tmp_path / "raw"
    for i in range(20):
        _img(raw / "images" / f"img{i:02d}.png", value=10 * i)
        objs = "<object><name>cat</name><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>30</xmax><ymax>40</ymax></bndbox></object>"
        if i % 2:
            objs += "<object><name>dog</name><bndbox><xmin>50</xmin><ymin>5</ymin><xmax>90</xmax><ymax>45</ymax></bndbox></object>"
        (raw / "ann").mkdir(exist_ok=True)
        (raw / "ann" / f"img{i:02d}.xml").write_text(f"<annotation><filename>img{i:02d}.png</filename><size><width>100</width>"
                                                    f"<height>50</height></size>{objs}</annotation>")
    root = dt.prepare({"name": "pets", "classes": CLASSES, "format": "voc", "source": {"folder": str(raw)}},
                      raw, tmp_path / "data" / "pets_yolo")
    (tmp_path / "project.yaml").write_text(f"name: pets\nclasses: [cat, dog]\nformat: voc\nsource:\n  folder: {raw.as_posix()}\n")
    monkeypatch.chdir(tmp_path)
    return root


def _components(demo, kind):
    return [b for b in demo.blocks.values() if isinstance(b, kind)]


def test_browser_uses_a_slider_with_the_train_range(project):
    demo = myapp.build_app("project.yaml")
    sliders = [s for s in _components(demo, gr.Slider) if s.label == LABEL]
    assert len(sliders) == 1
    s = sliders[0]
    assert (s.minimum, s.maximum, s.step, s.value, s.interactive) == (0, 13, 1, 0, True)      # 14 train images
    assert [n.label for n in _components(demo, gr.Number)] == ["Seed"]                      # seed box kept, no number box


def _ids_inside(block):
    """The ids of every component inside a layout block (a tab, a row, ...)."""
    ids = set()
    for child in getattr(block, "children", []):
        ids |= {child._id} | _ids_inside(child)
    return ids


def test_events_declare_matching_outputs(project):
    demo = myapp.build_app("project.yaml")
    data_tab = [b for b in demo.blocks.values() if isinstance(b, gr.Tab) and b.label == "1 · Data"]
    assert len(data_tab) == 1
    data_ids = _ids_inside(data_tab[0])
    outputs = []
    for f in demo.fns.values():
        if not any(block_id in data_ids for block_id, _ in f.targets):   # only the events of the tab "1 · Data"
            continue
        outputs.append([getattr(b, "label", None) for b in f.outputs])
        if f.inputs and getattr(f.inputs[0], "label", None) == "Split":   # both browser events (split and slider)
            assert len(f.fn("val", 0)) == len(f.outputs)                  # callback returns as many values as outputs
    assert sorted(outputs, key=len) == [
        ["20 random training images"],
        ["Image with its boxes", "Image and classes"],
        ["Image with its boxes", "Image and classes", LABEL]]


@pytest.mark.parametrize("split, n", [("train", 14), ("val", 3), ("test", 3)])
def test_slider_maximum_follows_the_split(project, split, n):
    rgb, text, slider = myapp.browse_split(project, CLASSES, split, 0)
    assert (slider.minimum, slider.maximum, slider.step, slider.interactive) == (0, n - 1, 1, True)
    assert rgb.shape == (50, 100, 3) and text.startswith(f"{split}/") and f"numbers 0 to {n - 1}" in text


def test_out_of_range_number_is_clamped(project):
    rgb, text, slider = myapp.browse_split(project, CLASSES, "val", 99)
    assert slider.value == 2 and "(image 2, numbers 0 to 2)" in text


def test_image_view_shows_the_image_and_its_classes(project):
    rgb, text, i = myapp.image_view(project, CLASSES, "train", 0)
    path = sorted((project / "images" / "train").glob("*.png"))[0]
    names = [CLASSES[c] for c, _ in dt.boxes_for(path)]
    assert i == 0 and rgb.shape == (50, 100, 3) and rgb.dtype == np.uint8
    assert f"{len(names)} boxes: {', '.join(names)}" in text


def test_empty_split_gives_a_disabled_slider(tmp_path):
    root = tmp_path / "ds"
    _img(root / "images" / "train" / "a.png")
    (root / "labels" / "train").mkdir(parents=True)
    (root / "labels" / "train" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    rgb, text, slider = myapp.browse_split(root, CLASSES, "val", 5)          # val has no images
    assert rgb is None and text == "No images in the val split."
    assert (slider.minimum, slider.maximum, slider.value, slider.interactive) == (0, 1, 0, False)
    one = myapp.browse_split(root, CLASSES, "train", 3)[2]                    # a single image: also disabled at 0
    assert (one.value, one.interactive) == (0, False)
