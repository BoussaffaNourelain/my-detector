# Dataset source (Module 2, Assignment 2.2)

The plastic-bottle dataset used from Module 2 onwards comes from Roboflow Universe. It is not included in the Git
repository (`data/*` is excluded by `.gitignore`); this file records where it comes from, its licence, and every change
made to it in this project.

## Original dataset (as published)

| Item | Value | Evidence |
|---|---|---|
| Title | Plastic Bottle 2.0 | `README.dataset.txt`, `README.roboflow.txt` |
| Source | https://universe.roboflow.com/somaiya-vlmqe/plastic-bottle-2.0-psuby/dataset/2 | `data.yaml` (`roboflow.url`) |
| Provider | Roboflow Universe workspace `somaiya-vlmqe`; the export names no individual author ("Provided by a Roboflow user") | `data.yaml` (`roboflow.workspace`), `README.dataset.txt` |
| Project / version | `plastic-bottle-2.0-psuby`, version 2 ("v2 2026-05-12 3:07pm") | `data.yaml` (`roboflow.project`, `roboflow.version`), `README.roboflow.txt` |
| Licence | CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/) | `data.yaml` (`roboflow.license`), `README.dataset.txt` |
| Export | Exported via roboflow.com on 12 May 2026 at 9:42 AM GMT | `README.roboflow.txt` |
| Format | YOLOv8 object detection (bounding boxes); one class, `plastic-bottle` (`nc: 1`) | `README.roboflow.txt`, `data.yaml` |
| Images | 3,759 JPEG images, all 640 × 640 pixels | `README.roboflow.txt`; image sizes checked on all 3,759 files |
| Preprocessing (by the publisher) | Auto-orientation of pixel data (with EXIF-orientation stripping); resize to 640 × 640 (Stretch). No image augmentation. | `README.roboflow.txt` |
| Published split | train 1,665 · valid 1,394 · test 700 (one label file per image) | folder counts of the export |
| Download | `Plastic Bottle 2.0.v2i.yolov8.zip` (156,568,346 bytes), downloaded by me on 10 Nov 2026 | file in my Downloads folder |

The export also contains polygon (segmentation) outlines for some objects: 18 of its label lines have more than five
values (train 1, valid 2, test 15). All other label lines are bounding boxes.

## Changes made in this project

The raw export is kept unchanged in `data/raw/plastic-bottle-2.0-v2/` (the zip's `data.yaml`, stored three times with
identical content, was written once). `prepare()` in `src/data.py` (through `notebooks/02_my_dataset.ipynb`, 10 Nov
2026) created the working copy `data/plastic_bottles_yolo/`:

1. **Split kept as published.** The same 1,665 / 1,394 / 700 images, byte-identical to the export; the folder `valid`
   is named `val` in the working copy. No images were added, removed, re-split or modified.
2. **Polygons converted to boxes.** Each of the 18 polygon label lines was replaced by its enclosing bounding box (the
   minimum and maximum x and y of its points, clipped to 0–1). None was dropped. Every conversion is listed in
   `data/plastic_bottles_yolo/label_conversions.csv` and summarised in `results/dataset_review.md`. Total boxes:
   5,682 in the export + 18 converted = 5,700.
3. **Label files rewritten** with six decimals per value. The export uses up to eight decimals, so box values were
   rounded; the largest change is 5 × 10⁻⁷ of the image size (about 0.0003 pixels at 640 × 640).
4. **`data.yaml` rewritten** for this project (absolute `path`, `images/train`, `images/val`, `images/test`,
   `names: {0: plastic-bottle}`).

No annotation was added, removed or edited by hand. Empty label files (417 images with no boxes) were kept as they
are; they are listed for review in `results/unlabelled_images.csv`.

## Attribution (for the assignment report)

> "Plastic Bottle 2.0" (version 2), Roboflow Universe workspace somaiya-vlmqe,
> https://universe.roboflow.com/somaiya-vlmqe/plastic-bottle-2.0-psuby/dataset/2, licensed under CC BY 4.0
> (https://creativecommons.org/licenses/by/4.0/). Changes: 18 polygon annotations were converted to their enclosing
> bounding boxes; the published train/valid/test split and the images were not changed.

Figures that show images from this dataset (for example `results/dataset_sample_grid.jpg`) should carry the same
attribution.
