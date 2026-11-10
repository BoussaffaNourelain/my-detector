# Working log

What I asked the AI assistant, what I checked, and what I changed. A few lines per session.

## Module 1

### Assignment 1.1 (30–31 Oct 2026, Claude Code)

**What I asked**
- Read CLAUDE.md, `docs/classical_spec.md` and `tests/test_classical.py`, verify the environment, and plan the 15
  items of `src/classical.py` before changing anything.
- Implement `src/classical.py` from the approved plan, with `cv2.convertScaleAbs` for `brightness_contrast` as the
  spec says, and run the tests.
- Inspect the notebook and list which Assignment 1.1 parts were complete and which were missing.
- Improve A5: a numerical demo on a flat region and an edge, zoomed crops of bottle details, bottle-related prompts
  instead of the template's examples, and a check of whether `sharpen` counts as a custom kernel.
- Insert my A5 prediction, review A5 before execution, then replace the near-blank hand-vs-OpenCV crop with a bottle
  region and add a bright-to-dark edge so the demo shows negative values.
- Execute the notebook, audit Assignment 1.1, explain each observation cell with a sentence starter, and insert my
  five observation sentences.

**What I checked**
- Environment: the venv is at `my-detector/.venv`; `check_setup.py` printed READY (Python 3.12.1, all pinned
  packages) and `pip check` found no broken requirements.
- Tests: `python -m pytest -q tests` gave 13 passed.
- I reviewed the plan, the A5 cell code and each diff before approving the next step.
- Execution with nbconvert (kernel mai630): 14 code cells, no errors, 30 figures. The A5 demo shows −100 before
  clipping and 0 after, and the hand-vs-OpenCV maximum difference is 0.0.

**What I changed**
- `src/classical.py`: new; written by the assistant from the approved plan.
- Notebook A5, edited by the assistant at my request: numerical demo cell, zoomed-crop cell, bottle-related
  prediction prompts, comparison region moved to the cap of image_1 (`[300:380, 20:100]`), bright-to-dark patch.
- Kept the spec's `sharpen` kernel; the assistant proposed a horizontal-rib kernel and I chose not to use it.
- I provided my A5 prediction (cell 21) and five observation sentences (cells 9, 12, 15, 18, 24), after the assistant
  explained each operation and gave a sentence starter; the assistant inserted my text unchanged.
- While listing the repository, the assistant read the first 20 lines of `reference/classical_reference.py`
  (`load_rgb`). The reference file was not copied or used.
- Final cleanup: the assistant removed the instructional "Things to predict..." paragraph from cell 21. My A5
  prediction was kept exactly, and no code cells, outputs or other notebook sections were modified.
- Not done yet: Assignment 1.2 (B5, B6 and its six observation cells).

### Assignment 1.1: replacing the stock images (31 Oct 2026, Claude Code)

**What I asked**
- Find three plastic-bottle images with explicit reuse licences (crushed; coloured or ribbed; clear with cap and
  neck), check their licences, resize them to about 800 px and save them as `bottle_1.jpg` to `bottle_3.jpg`.
- Move the three stock images out of `data/m1_images/` without deleting them, and record the image sources.
- Propose and apply new A5 crop and comparison coordinates, re-run the notebook, and add image credits.

**What I checked**
- I approved the three Wikimedia Commons images after reviewing their titles, sources and licences.
- The downloaded originals match the SHA-1 checksums published by Wikimedia Commons; the resized files load with
  `cl.load_rgb` as 363 × 800, 344 × 800 and 600 × 800 pixels.
- I approved the proposed coordinates after seeing the exact lines.
- Two re-runs with nbconvert (kernel mai630): 14 code cells, 0 errors, 30 figures, loading `bottle_1.jpg` to
  `bottle_3.jpg`; hand-vs-OpenCV maximum difference 0.0; the second run gave outputs identical to the first.
- Final validation, after the markdown changes: `notebooks/01_classical_cv.ipynb` executed in the venv with
  nbconvert, 14 of 14 code cells completed, 30 figures, 0 errors; `python -m pytest -q tests` gave 13 passed,
  0 failed; `data/m1_images/` holds exactly three inputs, `bottle_1.jpg`, `bottle_2.jpg` and `bottle_3.jpg`. The
  final observations (cells 9, 12, 15, 18, 24) and the prediction (cell 21) were checked and remained unchanged.
- Re-run after the setup-cell code change below: 14 of 14 code cells completed, 30 figures, 0 errors;
  `python -m pytest -q tests` still gave 13 passed.

**What I changed**
- `data/m1_images/` now holds `bottle_1.jpg` (David Monniaux, CC BY-SA 3.0), `bottle_2.jpg` (MdeVicente, CC0) and
  `bottle_3.jpg` (Jiafei Slay Queen, CC0). The stock `image_1.jpg` to `image_3.jpg` were moved unchanged to
  `data/stock_images_backup/`. The three bottle images are included in the repository; the stock backup stays
  excluded (`.gitignore`); sources and licences are documented in `docs/image_sources.md`.
- `docs/image_sources.md`: new; titles, authors, source links, licences, sizes, attribution and checksums.
- Notebook A5, edited by the assistant: the comparison region is now `[30:110, 100:180]` on the red cap of
  `bottle_1.jpg`, and the zoomed crops are `bottle_1.jpg` (170, 50), `bottle_2.jpg` (290, 40) and `bottle_3.jpg`
  (75, 235), given as (top, left).
- Notebook setup cell: an "Image credits" section, taken from `docs/image_sources.md`, with the CC BY-SA attribution
  for `bottle_1.jpg`.
- Observations (cells 9, 12, 15, 18, 24): rewritten to describe the three bottle images and their actual results.
  The assistant reported what each output showed for each image; I provided the new sentences and the assistant
  inserted them unchanged.
- Prediction (cell 21): two phrase substitutions that I specified, removing the references to the old white
  background and to the screw threads, which the caps now hide. The rest of the prediction is unchanged.
- The notebook was not executed between these markdown-only changes and the final validation; the code cells and
  their outputs stayed unchanged until then.
- `.gitignore`: added `!data/m1_images/bottle_[123].jpg`, so only `bottle_1.jpg`, `bottle_2.jpg` and `bottle_3.jpg`
  are included from `data/m1_images/`; `data/stock_images_backup/` and all other data files stay excluded.
- Notebook setup cell (cell 2): `print("Folder:", ROOT)` became `print("Folder:", ROOT.name)`, so the output shows
  only the folder name instead of the machine-specific Windows path.
- LOG.md: corrected two wordings (how I reviewed the images, and when the notebook was last executed).

### Assignment 1.2: boundary detection (31 Oct 2026, Claude Code)

**What I asked**
- Inspect the Part B instructions and plan the missing work (B5, B6 and six written sections).
- Measure candidate regions and design B5 (operators disagree) and B6 (every operator fails), then add the two cells.
- For each written section, report what the saved outputs show; then insert my texts.
- Validate, update this log, export to HTML, and commit and push.

**What I checked**
- The B5 and B6 crops were chosen from measured operator responses and previews; their code was tested outside the
  notebook before insertion, and the notebook printed the same numbers.
- For each written section the assistant reported measurements (for example: Gy stronger than Gx in all three images;
  bottle_1 concrete and bottle edge give Laplacian means 5.3 and 5.2; bottle_2's outline gradient peaks at 66, below
  every Canny high threshold). I reviewed them before writing.
- The assistant flagged that my first cell 33 draft attributed the bottle_2 failure to the image's strongest gradient
  (332, a rib highlight); I corrected it to the outline's own peak (66), and I named the (100, 200) pair in the
  bottle_3 sentence.
- Final validation: the notebook executed with nbconvert (kernel mai630), 16 of 16 code cells, 32 figures, 0 errors;
  `python -m pytest -q tests` gave 13 passed; `check_setup.py` printed READY; Assignment 1.1 cells 0 to 24 have the
  same text and outputs as the Assignment 1.1 commit.

**What I changed**
- B1 to B4 code was not changed; `src/classical.py` and the tests were not changed.
- Notebook B5 (cell 37): `bottle_1.jpg` crop (x1, y1, x2, y2) = (0, 260, 140, 400), concrete texture next to the
  bottle's left edge; colour crop, Sobel, Laplacian (x8 for display only), Canny (30, 90) and (100, 200), plus mean
  responses and Canny percentages for a concrete region and a bottle-edge strip.
- Notebook B6 (cell 40): `bottle_2.jpg` crop (0, 400, 140, 540), the transparent outline on the orange background;
  colour and grey crops, Sobel, Laplacian (x8), Canny (30, 90), the same region numbers, and grey values along row 470.
- Written sections (cells 27, 30, 33, 38, 41, 43: Sobel, Laplacian, Canny, B5, B6 and the final comparison): the
  assistant gave measurements and sentence starters; I provided the texts and the assistant inserted them unchanged.
- HTML export `A1_2_Boussaffa_Nour_El_Ain_MAI630.html` in the project root, for Moodle (not committed).

## Module 2

### Assignment 2.1: a CNN looks at my field (5–9 Nov 2026, Claude Code)

**What I asked**
- Implement `src/cnn.py` from `docs/cnn_spec.md`, following the plan, approval and implementation workflow, and run
  `tests/test_cnn.py`.
- Verify the project read-only (folders, `src/` modules, tests, environment) before continuing Module 2.
- Copy `bottle_1.jpg` to `bottle_3.jpg` from `data/m1_images/` into `data/my_field/` without changing the originals,
  and check their attribution in `docs/image_sources.md`.
- Execute `notebooks/02_my_field.ipynb`, including the one-time ResNet-18 weights download, which I approved; then
  insert my "My reading" text in cell 6.
- Prepare the written review (DOCX and PDF) with verified APA 7 references, then fit it to two A4 pages.

**What I checked**
- Tests: `python -m pytest -q tests/test_cnn.py` gave 10 passed; the full suite gave 37 passed.
- `data/my_field/` holds three byte-identical copies (same MD5) of the Module 1 images: `bottle_1.jpg` (363 × 800),
  `bottle_2.jpg` (344 × 800) and `bottle_3.jpg` (600 × 800); the notebook selected all three, so the course-sample
  fallback was not used.
- Execution with the venv's `jupyter-nbconvert.exe` (kernel mai630; the plain `jupyter` command on PATH belongs to
  another Python): 3 of 3 code cells, 1 figure, 0 errors. The weights file `resnet18-f37072fd.pth` (44.7 MB) was
  downloaded from download.pytorch.org.
- Small CNN trace: 1,400 weights, 1,426 with biases; a 224-pixel photo gives 112 after ResNet-18's first layer;
  ResNet-18 has 11,689,512 parameters.
- Top-3 predictions: `bottle_1.jpg` water bottle 0.89, water jug 0.11, nipple 0.00; `bottle_2.jpg` water bottle
  0.77, perfume 0.11, water jug 0.07; `bottle_3.jpg` water bottle 0.80, pop bottle 0.15, beer bottle 0.02.
- `results/my_field_cnn.png` (1920 × 1368): one row per photo with the photo, one layer1 map (56 × 56), one layer4
  map (7 × 7) and the Grad-CAM overlay for the top-1 class.
- The notebook's code cells are identical to the course copy; only cell 6 (markdown) differs.
- The four references were checked against Crossref, the publishers' pages and PubMed Central before use.
- Final PDF: 2 A4 pages with all paragraphs, both tables, the figure and six links; both pages were rendered and
  inspected.

**What I changed**
- `src/cnn.py`: new; I asked Claude Code to implement it from `docs/cnn_spec.md`, following the plan, approval and
  implementation workflow. It passed all 10 tests in `tests/test_cnn.py`.
- `data/my_field/`: the three copied photos (excluded by `.gitignore`, not committed).
- `notebooks/02_my_field.ipynb`: executed in place, outputs saved; code cells unchanged. Cell 6: I provided my
  "My reading" text and the assistant inserted it unchanged.
- `results/my_field_cnn.png`: generated by the notebook.
- Report `A2_1_Boussaffa_Nour_El_Ain_MAI630.pdf` (2 pages) in the project root, for Moodle (not committed). The
  assistant drafted the introduction, method and source summaries; section 4 is my cell 6 text and I wrote the
  conclusion. It credits `bottle_1.jpg` (CC BY-SA 3.0, the figure shared under the same licence) and `bottle_2.jpg`
  and `bottle_3.jpg` (CC0 1.0).
- Not part of this entry: `src/data.py`, `app/app.py` and the dataset work for Assignment 2.2.

### Assignment 2.2: my dataset and the data page (9–10 Nov 2026, Claude Code)

**What I asked**
- Implement `src/data.py` from `docs/data_spec.md` and then `app/app.py` (tab "1 · Data"), each after a plan that I
  approved, with these safety rules: never overwrite a non-empty output folder, report stem clashes instead of
  overwriting labels, and never write invalid or zero-area boxes.
- Switch `project.yaml` from the potholes example to plastic bottles, then extract my Roboflow download into
  `data/raw/plastic-bottle-2.0-v2/` and check it read-only before preparing anything.
- Keep the published split exactly; convert the 18 polygon label lines into their enclosing boxes (I chose this over
  dropping them), with tests and a record of every conversion.
- Run `notebooks/02_my_dataset.ipynb`, run the review-dataset skill, record my answers to its grid questions, and insert
  my observations in cell 14.
- Implement my own feature from `docs/my_feature.md` (`unlabelled_images()`), with its test, and run it once on the
  dataset.
- Check the data page on the real dataset, replace its image-number box with the slider required by the practice
  guide, and do the CLAUDE.md rule-breaking test.

**What I checked**
- Source: Plastic Bottle 2.0, version 2, Roboflow Universe workspace `somaiya-vlmqe`, CC BY 4.0, exported on 12 May
  2026 (YOLOv8, 3,759 images resized to 640 × 640, no augmentation). Published split: train 1,665, valid 1,394,
  test 700. The export's `data.yaml` names the class `plastic-bottle`. Details are in `docs/dataset_source.md`.
- Raw checks: every image has one label file; all class ids are 0; no box out of range or with zero area; 417 empty
  label files; 18 label lines with more than five values (polygons: train 1, valid 2, test 15).
- Preparation (by the notebook): the same images as the export, byte-identical, in train / val / test; all 18 polygons
  converted, 0 dropped (`label_conversions.csv`); 5,700 boxes (2,515 / 1,970 / 1,215). The export writes up to eight
  decimals and the labels are rewritten with six, so values were rounded by at most 5 × 10⁻⁷ of the image size. The
  raw folder stayed unchanged (fingerprint checked before and after).
- Notebook: 7 of 7 code cells, 0 errors, 2 figure outputs (the class-balance bar chart in cell 7 and the 20-image grid
  in cell 13); it wrote `results/dataset_review.md` and `results/dataset_sample_grid.jpg` (20 training images, seed 0).
- Review-dataset skill: its `write_report` command regenerated both files without changing `data/`. Its evidence: one
  class (train 2,515 boxes), 179 boxes under 1% of the image, 417 empty labels, 4 boxes under 0.1%, 2 duplicates
  across val and test (`plastic353` = `PET1-748`, `plastic393` = `PET1-761`). The assistant asked 12 questions about
  the grid; my answers are recorded unchanged in `results/grid_inspection.md`.
- My feature: `unlabelled_images()` lists 417 images (train 288, val 77, test 52) in `results/unlabelled_images.csv`;
  every row points to an existing image with an existing empty label; both datasets were unchanged afterwards.
- Data page with the real dataset (`python app/app.py`, http://127.0.0.1:7860): bar chart 2,515 training boxes with
  the axis from 0; split statistics and quality checks identical to the report; the slider ranges are 0–1664 (train),
  0–1393 (val) and 0–699 (test); image 530 in train is `pwb1472…` with 12 boxes; 99999 on test is clamped to 699 and
  switching from train 1600 to val is clamped to 1393; the grid with seeds 0 and 7 matches `sample_grid` for the same
  seeds; no errors in the server log or the browser console.
- CLAUDE.md rule test: a request to add boxes by hand to the empty label of `unnamed_jpg.rf.b8034a74…` in
  `data/plastic_bottles_yolo/` was refused with the rule "Never edit files there by hand." The datasets and every
  project file were unchanged afterwards (fingerprints).
- Tests: `python -m pytest -q tests` gave 66 passed, 6 warnings (the 6 warnings are Gradio's NumPy `np.bool8`
  deprecation notice). By file: classical 13, cnn 10, data 14, polygon labels 15, my feature 6, app browser 8.

**What I changed**
- `src/data.py`: new; written by the assistant from `docs/data_spec.md` (14 tests), then extended with
  `polygon_to_yolo()` and the conversion record (`tests/test_polygon_labels.py`, 15 tests) and with my feature
  `unlabelled_images()` (`tests/test_my_feature.py`, 6 tests).
- `app/app.py`: new; the data page. The bar chart's axis starts at 0, and the image-number box was replaced by a
  slider whose range follows the chosen split (`tests/test_app_browser.py`, 8 tests).
- `project.yaml`: plastic-bottle configuration (`plastic_bottles`, `[plastic-bottle]`, `yolo`,
  `data/raw/plastic-bottle-2.0-v2`), with the source URL and licence as a comment.
- `docs/my_feature.md`: my feature, written from the points I gave. `docs/dataset_source.md`: new; source, licence,
  original split and every change made in this project, with an attribution statement for the report.
- `notebooks/02_my_dataset.ipynb`: executed in place, code cells unchanged. Cell 14 ("My observations"): the four
  observations are my own text, which the assistant inserted unchanged under the template's heading and prompts. At
  my request it added only the full file name of the image at row 4, column 4
  (`unnamed_jpg.rf.b8034a74c51fabc3aa02221d9657586a.jpg`); it wrote none of the observations.
- `results/`: `dataset_review.md`, `dataset_sample_grid.jpg`, `unlabelled_images.csv` and `grid_inspection.md`.
- `CLAUDE.md`: its 11-line Module 2 block was already present before this log entry; it was not changed here and is
  not yet committed.
- `data/raw/plastic-bottle-2.0-v2/` and `data/plastic_bottles_yolo/`: on my machine only (excluded by `.gitignore`,
  not committed). No annotation was edited by hand.
- Not done yet: the Assignment 2.2 commit and push, and stopping the data-page server.
