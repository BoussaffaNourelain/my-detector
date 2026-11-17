"""The page "2 · Training" and train.launch, with fake processes: nothing is trained and no weights are downloaded."""
import subprocess
import sys
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import train as tr
from app import app as myapp

CLASSES = ["cat", "dog"]
HEADER = "epoch, time, train/box_loss, val/box_loss, metrics/precision(B), metrics/recall(B), metrics/mAP50(B), metrics/mAP50-95(B)\n"


class FakeProcess:
    """Stands in for subprocess.Popen: poll() gives None while 'running', then the exit code."""
    def __init__(self, code=None):
        self.code = code

    def poll(self):
        return self.code


@pytest.fixture
def project(tmp_path, monkeypatch):
    """tmp/project.yaml next to tmp/data/pets_yolo with 4 training images (one with a small box) and 1 val image."""
    root = tmp_path / "data" / "pets_yolo"
    rows = {"a": "0 .5 .5 .5 .5", "b": "0 .5 .5 .05 .05", "c": "1 .5 .5 .5 .5", "d": "1 .5 .5 .6 .6"}
    for s in ("train", "val", "test"):
        (root / "images" / s).mkdir(parents=True); (root / "labels" / s).mkdir(parents=True)
    for n, line in rows.items():
        cv2.imwrite(str(root / "images" / "train" / f"{n}.png"), np.zeros((20, 20, 3), np.uint8))
        (root / "labels" / "train" / f"{n}.txt").write_text(line + "\n")
    cv2.imwrite(str(root / "images" / "val" / "v.png"), np.zeros((20, 20, 3), np.uint8))
    (root / "labels" / "val" / "v.txt").write_text("0 .5 .5 .5 .5\n")
    (tmp_path / "project.yaml").write_text("name: pets\nclasses: [cat, dog]\nformat: yolo\nsource:\n  folder: raw\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(myapp, "RUNNING", {})
    return tmp_path, root


@pytest.fixture
def fake_launch(monkeypatch):
    """Replace train.launch: record its arguments and return a FakeProcess that is still running."""
    calls = []

    def launch(yaml_path, epochs, imgsz, name, runs="runs", weights="yolo26n.pt"):
        calls.append({"yaml": Path(yaml_path), "epochs": epochs, "imgsz": imgsz, "name": name, "runs": Path(runs)})
        return FakeProcess(None)
    monkeypatch.setattr(tr, "launch", launch)
    return calls


def _results(run, rows):
    run.mkdir(parents=True, exist_ok=True)
    (run / "results.csv").write_text(HEADER + "\n".join(", ".join(str(x) for x in r) for r in rows) + "\n")


def _components(demo, kind):
    return [b for b in demo.blocks.values() if isinstance(b, kind)]


# ---------------------------------------------------------------- train.launch

def test_launch_starts_a_separate_process_and_returns_at_once(tmp_path, monkeypatch):
    seen = {}

    class Popen:
        def __init__(self, cmd, stdout=None, stderr=None, cwd=None, env=None):
            seen.update(cmd=cmd, stdout=stdout.name, stderr=stderr, cwd=cwd, unbuffered=env.get("PYTHONUNBUFFERED"))

    monkeypatch.setattr(subprocess, "Popen", Popen)
    runs = tmp_path / "runs"
    proc = tr.launch(tmp_path / "d.yaml", 3, 320, "run1", runs=runs)
    assert isinstance(proc, Popen)                                       # the process itself, not its result
    assert seen["cmd"] == tr.train_command(tmp_path / "d.yaml", 3, 320, "run1", runs=runs)
    assert seen["cmd"][0] == sys.executable
    assert Path(seen["stdout"]) == runs / "run1.log" and (runs / "run1.log").exists()
    assert seen["stderr"] == subprocess.STDOUT and seen["unbuffered"] == "1"
    assert Path(seen["cwd"]) == tmp_path                                 # downloads land next to runs/, not in it


def test_train_command_uses_the_spec_values(tmp_path):
    cmd = tr.train_command(tmp_path / "d.yaml", 7, 320, "run1", runs=tmp_path / "runs")
    args = dict(a.split("=", 1) for a in cmd[3:] if "=" in a)
    assert cmd[3] == "train"
    assert args == {"model": "yolo26n.pt", "data": str((tmp_path / "d.yaml").resolve()), "epochs": "7",
                    "imgsz": "320", "batch": "8", "seed": "42", "project": str((tmp_path / "runs").resolve()),
                    "name": "run1", "exist_ok": "True"}


# ---------------------------------------------------------------- the page

def test_page_has_the_controls_with_their_defaults(project):
    demo = myapp.build_app("project.yaml")
    tabs = [t.label for t in _components(demo, gr.Tab)]
    assert tabs[:2] == ["1 · Data", "2 · Training"]
    radio = [r for r in _components(demo, gr.Radio) if r.label == "Training data"][0]
    assert radio.choices == [(c, c) for c in tr.CHOICES] and radio.value == "all"
    sliders = {s.label: s.value for s in _components(demo, gr.Slider)}
    assert sliders["Epochs"] == 30 and sliders["Image size"] == 416
    assert [b.value for b in _components(demo, gr.Button)].count("Start training") == 1
    count = [t for t in _components(demo, gr.Textbox) if t.label == "Training images"][0]
    assert count.value == "4 of 4 training images"
    timers = _components(demo, gr.Timer)
    assert len(timers) == 1 and timers[0].active is False                # nothing polls before a training starts


@pytest.mark.parametrize("choice, text", [("all", "4 of 4"), ("half", "2 of 4"), ("large", "3 of 4")])
def test_count_follows_the_choice(project, choice, text):
    _, root = project
    assert myapp.subset_count(root, CLASSES, choice, "cat") == f"{text} training images"


def test_count_for_one_class(project):
    _, root = project
    assert myapp.subset_count(root, CLASSES, "one_class", "dog") == "2 of 4 training images"


def test_start_saves_the_reason_and_launches(project, fake_launch):
    tmp, root = project
    runs = tmp / "runs"
    status, name, timer, button = myapp.start_training(root, CLASSES, runs, "half", "cat", "  More variety.  ",
                                                       12, 352, "half_e12")
    assert name == "half_e12" and "2 training images" in status
    assert timer.active is True and button.interactive is False
    why = (runs / "half_e12_why.txt").read_text(encoding="utf-8")
    assert "More variety." in why and "data: half" in why and "epochs: 12" in why
    assert fake_launch == [{"yaml": root / "subsets" / "half" / "data.yaml", "epochs": 12, "imgsz": 352,
                            "name": "half_e12", "runs": runs}]
    assert "half_e12" in myapp.RUNNING


def test_start_refuses_an_empty_reason(project, fake_launch):
    tmp, root = project
    with pytest.raises(gr.Error):
        myapp.start_training(root, CLASSES, tmp / "runs", "all", "cat", "   ", 30, 416, "r1")
    assert fake_launch == [] and not (tmp / "runs" / "r1_why.txt").exists()


@pytest.mark.parametrize("bad", ["", "a b", "../up", "x/y"])
def test_start_refuses_a_bad_run_name(project, fake_launch, bad):
    tmp, root = project
    with pytest.raises(gr.Error):
        myapp.start_training(root, CLASSES, tmp / "runs", "all", "cat", "why", 30, 416, bad)
    assert fake_launch == []


def test_start_refuses_an_existing_run(project, fake_launch):
    tmp, root = project
    (tmp / "runs" / "old").mkdir(parents=True)
    with pytest.raises(gr.Error):
        myapp.start_training(root, CLASSES, tmp / "runs", "all", "cat", "why", 30, 416, "old")
    assert fake_launch == []


def test_only_one_training_at_a_time(project, fake_launch):
    tmp, root = project
    myapp.start_training(root, CLASSES, tmp / "runs", "all", "cat", "why", 30, 416, "first")
    with pytest.raises(gr.Error):
        myapp.start_training(root, CLASSES, tmp / "runs", "all", "cat", "why", 30, 416, "second")
    assert [c["name"] for c in fake_launch] == ["first"]


def test_progress_while_running_and_after(project, fake_launch):
    tmp, root = project
    runs = tmp / "runs"
    myapp.start_training(root, CLASSES, runs, "all", "cat", "why", 4, 416, "r1")
    status, table, chart, timer, button = myapp.poll_progress(runs, "r1")        # no results.csv yet
    assert "0 of 4 epochs" in status and table is None and chart is None and timer.active is True

    _results(runs / "r1", [[1, 60, 2.0, 2.1, .1, .2, .3, .1], [2, 120, 1.5, 1.9, .2, .3, .4, .2]])
    status, table, chart, timer, button = myapp.poll_progress(runs, "r1")
    assert "2 of 4 epochs" in status and "About 4.0 minutes in total (2.0 left)" in status
    assert list(table["epoch"]) == [1, 2] and chart is not None
    assert timer.active is True and button.interactive is False

    myapp.RUNNING["r1"]["process"].code = 0                                      # the process ends
    status, table, chart, timer, button = myapp.poll_progress(runs, "r1")
    assert "finished (exit code 0)" in status and "best mAP50 0.400 at epoch 2" in status
    assert timer.active is False and button.interactive is True


def test_progress_reports_a_failed_run(project, fake_launch):
    tmp, root = project
    myapp.start_training(root, CLASSES, tmp / "runs", "all", "cat", "why", 4, 416, "r1")
    myapp.RUNNING["r1"]["process"].code = 1
    status, *_, timer, button = myapp.poll_progress(tmp / "runs", "r1")
    assert "exit code 1" in status and "runs/r1.log" in status
    assert timer.active is False and button.interactive is True


def test_poll_without_a_run(project):
    tmp, _ = project
    status, table, chart, timer, button = myapp.poll_progress(tmp / "runs", "")
    assert status == "No training started yet." and timer.active is False and button.interactive is True


def test_original_dataset_is_not_changed(project, fake_launch):
    tmp, root = project
    before = {p: p.read_bytes() for p in (root / "images").rglob("*") if p.is_file()}
    before |= {p: p.read_bytes() for p in (root / "labels").rglob("*") if p.is_file()}
    for choice in tr.CHOICES:
        myapp.subset_count(root, CLASSES, choice, "dog")
    myapp.start_training(root, CLASSES, tmp / "runs", "one_class", "dog", "why", 3, 320, "oc")
    after = {p: p.read_bytes() for p in (root / "images").rglob("*") if p.is_file()}
    after |= {p: p.read_bytes() for p in (root / "labels").rglob("*") if p.is_file()}
    assert after == before
