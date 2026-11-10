"""The app of my detector. Module 2 adds the tab "1 · Data"; Modules 3 and 4 add more tabs to this file.

Run it from anywhere: python app/app.py. The dataset folder is found next to the project.yaml it reads.
See docs/data_spec.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr                                             # noqa: E402
import pandas as pd                                             # noqa: E402

from src import data as dt                                      # noqa: E402


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


def build_app(project_path: str | Path = "project.yaml") -> gr.Blocks:
    """The Gradio app for the project described in project_path."""
    project_path = Path(project_path)
    project = dt.load_project(project_path)
    classes = project["classes"]
    with gr.Blocks(title=f"My detector: {project['name']}") as demo:
        gr.Markdown(f"# My detector: {project['name']}\nClasses: {', '.join(classes)}")
        with gr.Tab("1 · Data"):
            _data_tab(dataset_dir(project_path, project), classes, project_path)
    return demo


if __name__ == "__main__":
    build_app(ROOT / "project.yaml").launch()
