"""The app of my detector. Module 2 adds the tab "1 · Data", Module 3 the tab "2 · Training"; Module 4 adds more.

Run it from anywhere: python app/app.py. The dataset folder and runs/ are found next to the project.yaml it reads.
See docs/data_spec.md and docs/train_spec.md.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr                                             # noqa: E402
import pandas as pd                                             # noqa: E402
from matplotlib.figure import Figure                           # noqa: E402

from src import data as dt                                      # noqa: E402
from src import train as tr                                     # noqa: E402

# Trainings started from this app: run name -> {"process": Popen, "epochs": int}. Kept here, not in gr.State,
# because Gradio copies State values and a process cannot be copied.
RUNNING: dict[str, dict] = {}
REFRESH_SECONDS = 5


def dataset_dir(project_path: str | Path, project: dict) -> Path:
    """data/<name>_yolo, relative to the folder that contains project.yaml (not to the working directory)."""
    return Path(project_path).resolve().parent / "data" / f"{project['name']}_yolo"


def _split_images(root: Path, split: str) -> list[Path]:
    folder = root / "images" / split
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in dt.IMAGE_EXT)


def image_view(root: Path, classes: list[str], split: str, number) -> tuple:
    """One image of a split with its boxes -> (rgb or None, text with its classes, the image number used)."""
    paths = _split_images(root, split)
    if not paths:
        return None, f"No images in the {split} split.", 0
    i = min(max(int(number or 0), 0), len(paths) - 1)        # 0-based, kept inside 0 .. n-1
    path = paths[i]
    names = [classes[c] if 0 <= c < len(classes) else f"unknown id {c}" for c, _ in dt.boxes_for(path)]
    text = (f"{split}/{path.name}   (image {i}, numbers 0 to {len(paths) - 1})\n"
            f"{len(names)} boxes: {', '.join(names) if names else 'none'}")
    return dt.show(path, classes), text, i


def image_slider(n_images: int, value: int = 0) -> gr.Slider:
    """The image-number slider for a split of n_images: 0 .. n-1, step 1.

    Gradio refuses a slider whose minimum equals its maximum, so a split with 0 or 1 image gets the range 0 .. 1 with
    the slider disabled at 0 (image_view clamps the number anyway, so index 1 is never shown).
    """
    if n_images < 2:
        return gr.Slider(minimum=0, maximum=1, step=1, value=0, label="Image number (0 to n-1)", interactive=False)
    return gr.Slider(minimum=0, maximum=n_images - 1, step=1, value=min(max(int(value), 0), n_images - 1),
                     label="Image number (0 to n-1)", interactive=True)


def browse_split(root: Path, classes: list[str], split: str, number) -> tuple:
    """A new split was chosen -> (rgb or None, text, slider with this split's range and the clamped number)."""
    rgb, text, i = image_view(root, classes, split, number)
    return rgb, text, image_slider(len(_split_images(root, split)), i)


def grid_view(root: Path, classes: list[str], seed) -> object:
    """The 20-image grid of the training split for this seed."""
    try:
        return dt.sample_grid(root, "train", n=20, seed=int(seed or 0), classes=classes)
    except FileNotFoundError as err:
        raise gr.Error(str(err))


def _data_tab(root: Path, classes: list[str], project_path: Path) -> None:
    if not root.is_dir():
        gr.Markdown(f"The dataset folder `{root}` does not exist yet.\n\n"
                    f"Run `notebooks/02_my_dataset.ipynb` first (it reads `{project_path.name}` and creates it).")
        return

    counts = dt.class_counts(root, classes)
    stats = dt.split_stats(root)
    checks = dt.review(root, len(classes))

    gr.Markdown("### Class balance")
    train_boxes = [counts["train"][c] for c in classes]
    gr.BarPlot(pd.DataFrame({"class": classes, "boxes": train_boxes}),
               x="class", y="boxes", title="Boxes per class in the training split",
               x_title="class", y_title="boxes",
               y_lim=[0, max(train_boxes + [1])])                # start at 0, or small classes look empty

    gr.Markdown("### Split statistics")
    gr.Dataframe(pd.DataFrame([{"split": s, **stats[s]} for s in dt.SPLITS]), interactive=False)

    gr.Markdown("### Quality checks")
    gr.Dataframe(pd.DataFrame([{"check": k, "count": checks["counts"][k],
                                "first examples": ", ".join(checks["examples"][k][:3])}
                               for k in checks["counts"]]), interactive=False)

    gr.Markdown("### Image browser")
    first_image, first_text, _ = image_view(root, classes, "train", 0)
    with gr.Row():
        split = gr.Dropdown(list(dt.SPLITS), value="train", label="Split")
        number = image_slider(len(_split_images(root, "train")))
    image = gr.Image(value=first_image, type="numpy", interactive=False, label="Image with its boxes")
    info = gr.Textbox(value=first_text, label="Image and classes", lines=2, interactive=False)
    # split -> image, text and the slider's range (3 outputs); slider -> image and text only (2 outputs)
    split.change(lambda s, n: browse_split(root, classes, s, n), [split, number], [image, info, number])
    number.change(lambda s, n: image_view(root, classes, s, n)[:2], [split, number], [image, info],
                  trigger_mode="always_last")

    gr.Markdown("### The 20-image grid (training split)")
    with gr.Row():
        seed = gr.Number(value=0, precision=0, label="Seed")
        show_grid = gr.Button("Show the 20-image grid", variant="primary")
    grid = gr.Image(type="numpy", interactive=False, label="20 random training images")
    show_grid.click(lambda s: grid_view(root, classes, s), [seed], [grid])


# ---------------------------------------------------------------- tab "2 · Training" (docs/train_spec.md)

def runs_dir(project_path: str | Path) -> Path:
    """runs/, next to project.yaml (not relative to the working directory)."""
    return Path(project_path).resolve().parent / "runs"


def default_run_name(choice: str, epochs, imgsz) -> str:
    return f"{choice}_e{int(epochs)}_s{int(imgsz)}"


def subset_count(root: Path, classes: list[str], choice: str, class_name: str | None) -> str:
    """How many training images a data choice means (writes root/subsets/<choice>/, as make_subset does)."""
    try:
        _, n = tr.make_subset(root, choice, classes, class_name=class_name)
    except (ValueError, FileNotFoundError) as err:
        return f"Cannot use {choice}: {err}"
    total = len(_split_images(root, "train"))
    return f"{n} of {total} training images"


def _active_run() -> str | None:
    """The name of a training started from this app that is still running, if any."""
    for name, run in RUNNING.items():
        if run["process"].poll() is None:
            return name
    return None


def start_training(root: Path, classes: list[str], runs: Path, choice: str, class_name: str | None,
                   why: str, epochs, imgsz, name: str) -> tuple:
    """Check the inputs, save the reason, start the training in a separate process and return at once.

    -> (status text, run name, timer update, start-button update). Raises gr.Error when the inputs are not valid.
    """
    why, name = (why or "").strip(), (name or "").strip()
    if not why:
        raise gr.Error("Write why you chose this data before you start.")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise gr.Error("The run name may use only letters, digits, '_', '-' and '.'.")
    runs = Path(runs)
    if (runs / name).exists() or (runs / f"{name}.log").exists():
        raise gr.Error(f"runs/{name} already exists: choose another run name.")
    if (busy := _active_run()) is not None:
        raise gr.Error(f"The training {busy} is still running.")
    yaml_path, n = tr.make_subset(root, choice, classes, class_name=class_name)
    if n == 0:
        raise gr.Error(f"The choice {choice} has no training images.")

    runs.mkdir(parents=True, exist_ok=True)
    (runs / f"{name}_why.txt").write_text(
        f"run: {name}\ndata: {choice}" + (f" ({class_name})" if choice == "one_class" else "") +
        f", {n} training images\nepochs: {int(epochs)}\nimage size: {int(imgsz)}\n\nWhy I chose it:\n{why}\n",
        encoding="utf-8")
    process = tr.launch(yaml_path, int(epochs), int(imgsz), name, runs=runs)
    RUNNING[name] = {"process": process, "epochs": int(epochs)}
    status = f"Started {name}: {choice}, {n} training images, {int(epochs)} epochs, image size {int(imgsz)}."
    return status, name, gr.Timer(active=True), gr.Button(interactive=False)


def progress_chart(rows: list[dict]) -> Figure | None:
    """Box losses and mAP50 per epoch."""
    if not rows:
        return None
    epochs = [r["epoch"] for r in rows]
    fig = Figure(figsize=(9, 3.5))
    loss, m = fig.subplots(1, 2)
    loss.plot(epochs, [r["train_loss"] for r in rows], marker="o", label="train box loss")
    loss.plot(epochs, [r["val_loss"] for r in rows], marker="o", label="val box loss")
    loss.set(xlabel="epoch", ylabel="box loss", title="Losses")
    loss.legend()
    m.plot(epochs, [r["mAP50"] for r in rows], marker="o", color="tab:green")
    m.set(xlabel="epoch", ylabel="mAP50", title="mAP50 (validation)", ylim=(0, 1))
    fig.tight_layout()
    return fig


def _status(name: str, rows: list[dict], code: int | None, planned: int | None, runs: Path) -> str:
    lines = []
    if code is None:
        lines.append(f"{name}: training, {len(rows)} of {planned} epochs finished." if planned
                     else f"{name}: training, {len(rows)} epochs finished.")
        if rows and planned:
            per_epoch = rows[-1]["seconds"] / len(rows)
            lines.append(f"About {tr.estimate_minutes(per_epoch, planned)} minutes in total "
                         f"({tr.estimate_minutes(per_epoch, planned - len(rows))} left).")
    else:
        lines.append(f"{name}: finished (exit code {code})." if code == 0 else
                     f"{name}: stopped with exit code {code}. See runs/{name}.log.")
    if (s := tr.summarize(rows)):
        lines.append(f"{s['epochs']} epochs, {s['minutes']} minutes, best mAP50 {s['best_mAP50']:.3f} at epoch "
                     f"{s['best_epoch']}, overfitting: {'yes' if s['overfitting'] else 'no'}.")
    if code is not None and (w := tr.best_weights(runs / name)):
        lines.append(f"Best weights: {w}")
    return "\n".join(lines)


def poll_progress(runs: Path, name: str) -> tuple:
    """Read the run's progress -> (status, table, chart, timer update, start-button update). Called by the timer."""
    if not name:
        return "No training started yet.", None, None, gr.Timer(active=False), gr.Button(interactive=True)
    runs = Path(runs)
    rows = tr.read_progress(runs / name)
    run = RUNNING.get(name)
    code = run["process"].poll() if run else 0                # a run not started here is treated as finished
    status = _status(name, rows, code, run["epochs"] if run else None, runs)
    table = pd.DataFrame(rows) if rows else None
    running = code is None
    return status, table, progress_chart(rows), gr.Timer(active=running), gr.Button(interactive=not running)


def _training_tab(root: Path, classes: list[str], project_path: Path) -> None:
    if not root.is_dir():
        gr.Markdown(f"The dataset folder `{root}` does not exist yet. Prepare it on the tab 1 · Data first.")
        return
    runs = runs_dir(project_path)

    gr.Markdown("### 1. The training data")
    with gr.Row():
        choice = gr.Radio(list(tr.CHOICES), value="all", label="Training data")
        class_name = gr.Dropdown(classes, value=classes[0], label="Class (only for one_class)")
    count = gr.Textbox(value=subset_count(root, classes, "all", classes[0]), label="Training images",
                       interactive=False)
    gr.Markdown("### 2. Why I chose it")
    why = gr.Textbox(label="Why I chose this data", lines=3, placeholder="Saved as runs/<run name>_why.txt")
    gr.Markdown("### 3. The training")
    with gr.Row():
        epochs = gr.Slider(minimum=1, maximum=100, step=1, value=30, label="Epochs")
        imgsz = gr.Slider(minimum=320, maximum=640, step=32, value=416, label="Image size")
        name = gr.Textbox(value=default_run_name("all", 30, 416), label="Run name")
    start = gr.Button("Start training", variant="primary")
    gr.Markdown("### 4. Progress")
    status = gr.Textbox(label="Status", lines=4, interactive=False)
    chart = gr.Plot(label="Losses and mAP50")
    table = gr.Dataframe(label="Progress per epoch", interactive=False)
    run_name = gr.State("")
    timer = gr.Timer(REFRESH_SECONDS, active=False)

    choice.change(lambda c, k: subset_count(root, classes, c, k), [choice, class_name], [count])
    class_name.change(lambda c, k: subset_count(root, classes, c, k) if c == "one_class" else gr.skip(),
                      [choice, class_name], [count])
    for box in (choice, epochs, imgsz):
        box.change(default_run_name, [choice, epochs, imgsz], [name])
    start.click(lambda *a: start_training(root, classes, runs, *a),
                [choice, class_name, why, epochs, imgsz, name], [status, run_name, timer, start])
    timer.tick(lambda n: poll_progress(runs, n), [run_name], [status, table, chart, timer, start])


def build_app(project_path: str | Path = "project.yaml") -> gr.Blocks:
    """The Gradio app for the project described in project_path."""
    project_path = Path(project_path)
    project = dt.load_project(project_path)
    classes = project["classes"]
    root = dataset_dir(project_path, project)
    with gr.Blocks(title=f"My detector: {project['name']}") as demo:
        gr.Markdown(f"# My detector: {project['name']}\nClasses: {', '.join(classes)}")
        with gr.Tab("1 · Data"):
            _data_tab(root, classes, project_path)
        with gr.Tab("2 · Training"):
            _training_tab(root, classes, project_path)
    return demo


if __name__ == "__main__":
    build_app(ROOT / "project.yaml").launch()
