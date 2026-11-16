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

## Module 3

### Assignment 3.1: preparing the training experiment (12 Nov 2026, Claude Code)

**What I asked**
- Append `docs/CLAUDE_module3.md` to the end of CLAUDE.md, check `.gitignore` for `runs/` and `*.pt`, run the tests
  and check `data/plastic_bottles_yolo/`.
- Insert my data choice, reason and prediction into the first three fields of `docs/experiment.md`.
- Challenge my reasoning with real box counts and box sizes from the dataset, in at most three questions, without
  answering them for me.

**What I checked**
- `.gitignore` already contained `runs/` and `*.pt`. `python -m pytest -q tests` stopped at collection because
  `src/train.py` and `src/evaluate.py` do not exist yet; without `tests/test_train.py` and `tests/test_evaluate.py`,
  66 passed.
- Dataset: 640 × 640 images; train 1,665 images / 2,515 boxes, val 1,394 / 1,970, test 700 / 1,215.
- Findings from the box statistics (read-only script, COCO size limits in pixels):
  - Only 2 of the 1,215 test boxes are small (under 32 × 32 pixels).
  - 29.8% of test boxes touch an image edge, compared with 24% in training.
  - Crowded images (more than 5 bottles) are 5.3% of test images, compared with 3.1% in training.
  - A single run cannot prove that "all" performs better than "half" or "large only".
- The three questions the assistant raised:
  1. With only 2 small bottles in the test set, what evidence would show whether the model struggles with small
     bottles?
  2. About 30% of test boxes touch an image edge. How will I look at those boxes separately from the overall mAP?
  3. If more examples help, what result from a single run would show that "all" was better than "half" or "large
     only", and what result would show it was not?

**What I changed**
- `CLAUDE.md`: the Module 3 block from `docs/CLAUDE_module3.md` appended at the end; existing rules unchanged.
- `docs/experiment.md`: the first three fields hold my choice, reason and prediction, inserted unchanged by the
  assistant. The after-training fields are empty.
- My decision after the review: I kept "all" (the complete training dataset).
- Not done yet: `src/train.py`, the "2 · Training" page and the training run.

### Assignment 3.1: `src/train.py` and the page "2 · Training" (12 Nov 2026, Claude Code)

**What I asked**
- Implement `src/train.py` from `docs/train_spec.md` and the page "2 · Training" in `app/app.py`, following the plan I
  approved, with training in a separate process so the page stays responsive.
- Use the Windows workaround for `test_one_class` only if it was required, and document it.
- Keep every Module 2 test; change only the event-count assertion in `tests/test_app_browser.py` so that it checks
  the tab "1 · Data" on its own.
- Create `tests/test_app_training.py` with fake processes. No real training, no weight download, no new packages.

**What I checked**
- Tests: `python -m pytest -q tests/test_train.py` gave 10 passed; `python -m pytest -q tests/test_app_training.py`
  gave 19 passed.
- The full `python -m pytest -q tests` stops at collection because `tests/test_evaluate.py` imports
  `src/evaluate.py`, which does not exist yet. Without that file, 95 passed, 8 warnings (Gradio's NumPy `np.bool8`
  deprecation notice). By file: classical 13, cnn 10, data 14, polygon labels 15, my feature 6, app browser 8,
  train 10, app training 19.
- Windows workaround: with the extra label write switched off, `test_one_class` failed with `FileNotFoundError` on
  `subsets\one_class\images\val\v.txt`; with it, all 10 tests in `tests/test_train.py` passed.
- Before the event-count assertion was changed, adding the tab made only `test_events_declare_matching_outputs` fail
  (7 of 8 passed); after the change, 8 of 8 passed.
- Training tab in the real app (`python app/app.py`, http://127.0.0.1:7860), without pressing Start training: the
  tab loads; "all" showed 1665 of 1665 training images and "large" showed 1350 of 1665; the run name changed to
  `large_e30_s416`; no errors in the browser console or the server log. The generated `train.txt` files have 1,665
  lines (all), 832 (half) and 1,350 (large).
- After this check: no `runs/` folder and no `.pt` file in the project; `subsets/all`, `subsets/half` and
  `subsets/large` were written inside `data/plastic_bottles_yolo/`, as the spec says.

**What I changed**
- `src/train.py`: new; written by the assistant from `docs/train_spec.md`, with the seven functions of the spec.
  `launch()` starts the training with `subprocess.Popen` and returns the process at once; the output goes to
  `runs/<name>.log`.
- Windows workaround in `make_subset(..., "one_class")`: each filtered val and test label is also written next to its
  copied image (`subsets/one_class/images/<split>/<stem>.txt`), because the test finds the label with
  `str(path).replace("/images/", "/labels/")`, which does not match Windows paths with backslashes. Ultralytics
  ignores these files and reads `labels/<split>/`. Documented in the docstring of `_one_class_copy`.
- `app/app.py`: new tab "2 · Training": data choice and its image count, a "why" text box saved as
  `runs/<name>_why.txt` before the training starts, epochs (30) and image size (416) as sliders, a run name, a Start
  training button, and a status, table and chart refreshed every 5 seconds by a `gr.Timer`. Start refuses an empty
  reason, an invalid or existing run name, and a second training while one is running. The tab "1 · Data" is
  unchanged.
- `tests/test_app_browser.py`: only `test_events_declare_matching_outputs` changed: it now keeps the events of the
  tab "1 · Data" and checks them against the same three expected output lists. The other assertions are unchanged.
- `tests/test_app_training.py`: new; 19 tests with fake processes for `launch()`, `train_command()` and the page.
- No packages installed, no training started, no weights downloaded. `docs/experiment.md` was not changed.
- Not done yet: `src/evaluate.py`, the real training run and the commit.

### Assignment 3.1: the training run and its evaluation (14 Nov 2026, Claude Code and Google Colab)

**Work completed**
- Ran a 2-epoch trial from the page "2 · Training" on my laptop CPU (`all_e2_s416_retry1`) to check the page end to
  end. An earlier attempt, `all_e2_s416_test` (12 Nov), was interrupted before it finished: its log ends with a
  `KeyboardInterrupt` traceback and it has no `results.csv`.
- Asked Claude to explain the trial with the `explain-run` skill.
- The local 2-epoch trial took about 21.3 minutes, so I moved the full experiment to Google Colab to use its Tesla T4
  GPU (Plan B in the practice guide). The final run, `all_e30_s416_colab`, completed 30 epochs in 22.2 minutes.
- Asked Claude to plan and then implement `src/evaluate.py` from `docs/eval_spec.md`, and to run the tests and a real
  evaluation on the test split.
- Filled in the second part of `docs/experiment.md`; Claude inserted my figures and my interpretation unchanged.
- Committed the work (`aeb1190b`, then `d0371442`) and prepared the Assignment 3.1 PDF.

**Training configuration and experiment**
- Trial run, local CPU (Intel Core i5-10300H), `all_e2_s416_retry1`: data `all` (1,665 training images), 2 epochs,
  image size 416. It finished with exit code 0 after 21.3 minutes (about 10.6 minutes per epoch); best validation
  mAP50 0.759 at epoch 2. This was a check of the page, not my final model.
- What `explain-run` reported for the trial: validation losses fell at both epochs, mAP50 was still rising at the end,
  `warmup_epochs` (3) was longer than the 2 epochs, and every metric came from the validation split, so the reported
  results must come from the test split. Its question: what in these files could show whether the result was limited
  by the amount of data or by the number of epochs.
- I also asked Claude to run `explain-run` on the final Colab run. The analysis helped me verify the training
  duration (22.2 minutes), the best epoch (29), the validation metrics (best mAP50 92.764%) and the training-loss
  trends.
- Final run, Google Colab, `all_e30_s416_colab`: YOLO26n, data `all` (1,665 training images), image size 416, batch
  size 8, 30 epochs, Tesla T4 GPU. Training time 22.2 minutes.
- Final run results (validation split, from its `results.csv`): best epoch 29, mAP50 0.92764 (92.764%), mAP50-95
  0.80853 (80.853%). Training and validation box losses both peaked in the first epochs (epoch 4 and epoch 3) and
  reached their lowest values at epoch 30 (0.631 and 0.623). Over the last five epochs mAP50 stayed between 0.925
  and 0.928. `summarize()` overfitting flag: False.

**Test-set evaluation**
- Final model on the test split: 700 images, 1,215 ground-truth instances. Test mAP50 0.77842 (77.842%), test
  mAP50-95 0.63546 (63.546%). Sources: `results/final_test_metrics.json` and `results/thresholds.json`.
- Precision and recall at four confidence thresholds (TP / FP / FN out of 1,215 bottles):
  - 0.10: precision 47.85%, recall 79.75% (969 / 1,056 / 246)
  - 0.25: precision 73.82%, recall 74.98% (911 / 323 / 304)
  - 0.50: precision 89.67%, recall 66.42% (807 / 93 / 408)
  - 0.75: precision 96.66%, recall 57.20% (695 / 24 / 520)
- The trade-off in these numbers: raising the threshold keeps only the more confident boxes. From 0.25 to 0.50 the
  false alarms fall from 323 to 93 (230 fewer), but correct detections also fall from 911 to 807 (104 fewer), so
  precision rises and recall drops. mAP50 does not depend on a threshold: it summarises all of them.
- The test mAP50 (77.84%) is lower than the best validation mAP50 (92.76%).

**Verification and project updates**
- `src/evaluate.py`: new; written by Claude from the approved plan. `python -m pytest -q tests/test_evaluate.py`
  gave 10 passed; the full `python -m pytest -q tests` gave 105 passed, 8 warnings (Gradio's NumPy `np.bool8`
  deprecation notice).
- Corrections found during this work: the virtual environment is at `../.venv`, not `.venv`; the test split has 700
  images, not the 350 first estimated in the plan.
- `yolo_metrics()` writes a temporary copy of `data.yaml` under `runs/` when the stored Windows path does not exist
  (Colab or Linux); the original `data.yaml` was checked byte for byte and is unchanged.
- `results/thresholds.json` still held the 2-epoch trial numbers after the Colab results were copied in. It was
  replaced with the four Colab rows from `results/final_test_metrics.json` after checking that TP + FN = 1,215 at
  every threshold and that precision and recall recompute exactly from the counts.
- `results/examples.png` and `results/thresholds.png` are the Colab figures (byte-identical to the downloaded files).
- `evidence/colab_training_evidence.png`: two Colab screenshots taken after training (the GPU, `args.yaml` values and
  30 completed epochs; the run's `results.png`). There are no screenshots taken during the 30-epoch training.
- Commits: `aeb1190b` (`src/evaluate.py` and the first results) and `d0371442` ("Complete Module 3 training and
  evaluation"). `runs/`, `*.pt` and the dataset do not appear in `git status`.

**Lessons learned and reflection**
- From my interpretation in `docs/experiment.md`: my prediction was partly correct. Training with the complete
  dataset helped the model learn useful visual features for detecting plastic bottles. However, the test mAP50 was
  77.84%, compared with 92.76% on the validation set, which suggests that the model performed less effectively on
  unseen images and highlights the challenge of generalization.
- Also from `docs/experiment.md`: the confidence threshold affects the balance between precision and recall. At 0.50,
  precision reached 89.67% while recall was 66.42%: fewer false positives, but more bottles missed. Choosing a
  threshold depends on whether reducing false detections or finding more bottles is the priority.

### Assignment 3.2: evaluation, images and report (16 Nov 2026, Claude Code)

**What I asked**
- Audit the 3.2 deliverables read-only, then run `notebooks/03_my_results.ipynb` on the final Colab model with the
  Colab result files backed up first and restored afterwards.
- Run the `image-auditor` subagent on `results/examples.png`, and make a separate, more readable copy of the figure.
- Inspect every green box in the six example images at full resolution, to find one detection the metric accepts
  that I would not accept.
- Prepare `docs/my_report.md`, help me write my sections, and run the `metrics-reviewer` subagent on it.

**What I checked**
- Tests: `python -m pytest -q tests` gave 105 passed, 8 warnings (Gradio's `np.bool8` notice).
- Notebook: `RUN = "detect/runs/all_e30_s416_colab"` (the only code change). Executed with nbconvert, kernel
  mai630, on CPU: 4 of 4 code cells, 0 errors, 3 min 24 s.
- The local run matches Colab exactly: mAP50 0.77842, mAP50-95 0.63546, and identical TP / FP / FN at all four
  thresholds; `thresholds.json` was byte-identical. `what_changed` from 0.25 to 0.75: 216 correct and 299 wrong
  detections removed. The six example images were the same; the PNGs differed only in font rendering.
- Colab files restored: `thresholds.json`, `thresholds.png`, `examples.png` and `final_test_metrics.json` have their
  original SHA-256 again (backup in `runs/colab_results_backup/`, git-ignored).
- `image-auditor` grouped the mistakes into four types (a red and a yellow box on one object; false alarms on
  possible non-bottles; false alarms on bottle-like objects; partly visible objects at the edges). It undercounted
  the bottom row and marked its bag, cup and can readings as uncertain. The exact per-image counts at confidence 0.25
  are: Correct 1–3 11/0/0, 7/0/0, 6/0/0; Wrong 1–3 10/19/16, 3/21/12, 18/17/12 (TP/FP/FN).
- Box inspection (same model, `predict_images()` and `match()`, read-only script outside the project): the 24 green
  boxes in the three correct images have IoU 0.844–0.988, with no clearly unacceptable box. In Wrong 1, box 6
  (numbered by confidence) has confidence 0.61 and IoU 0.5095; predicted box (431, 345, 623, 515), label
  (441, 372, 565, 506), area ratio 1.96; 51% of the box covers its own label, 77% overlaps the label of a
  neighbouring bottle (475, 329, 635, 627), and 53% of that label lies inside the box. The neighbouring bottle is
  matched separately by box 3 (470, 345, 634, 634; confidence 0.71, IoU 0.897).
- `metrics-reviewer` on `results/thresholds.json` and `docs/my_report.md`: all numbers in Sections 1–4 match the
  result files; the Section 3 differences table and the Section 5 arithmetic are correct. It raised 12 points:
  1. Section 5 numbers were not traceable to any file, and Wrong 1 has two green boxes at 0.61 and two at 0.71.
  2. IoU 0.510 vs 0.509 when recomputed from the rounded coordinates.
  3. "It does not check how much extra area the box includes" is wrong: the extra area is in the IoU union.
  4. The box fails at IoU 0.55–0.95, so mAP50-95 already penalises it.
  5. A false positive at IoU 0.5 can also be a badly placed box on a real bottle, not only a non-bottle.
  6. Section 3 gives no number for what I lose (230 more FPs, precision 89.67% to 73.82%) and does not say why
     not 0.10.
  7. "It could end up in general waste" is an assumption about the use case, not in any file.
  8. The wrong examples are crowded by construction (most FP + FN); 5.3% of test is about 37 images and 3.1% of
     train about 52, so crowding is not shown to cause errors.
  9. The edge-box percentages show a different distribution, not worse performance.
  10. Wrong 1, box 6 is a loose true positive, so it fits proposal 3 (label quality) better than proposal 1.
  11. Not used: 15 of the 18 polygon conversions are in the test split, and 52 test images have empty labels.
  12. "Images used during training": validation selects `best.pt`; 2 shared images out of 700.

**What I changed**
- `notebooks/03_my_results.ipynb`: `RUN` set to the Colab run; executed, outputs saved.
- `results/examples_readable.png`: new; the same six images, predictions and scores as `examples.png`, with short
  titles, exact counts, a legend and labels kept inside the image. `results/examples.png` is unchanged.
- `docs/my_report.md`: new. Sections 1, 2 and 4 and the facts in Sections 5 and 6 were prepared by Claude from
  the verified files. Section 3 is my own text, inserted word for word. The prose of Sections 5 and 6 was drafted
  by Claude from the judgment and the four recommendations I gave, then I edited proposal 1.
- Reviewer points accepted (factual or technical corrections, made by Claude): 1 (Section 5 now says how the values
  were measured, identifies box 6 among the two 0.61 boxes, and gives box 3's coordinates), 2 (the table gives
  0.5095 unrounded and 0.509 from the rounded coordinates), 3 (the sentence now says the extra area lowers the IoU,
  but the metric does not check whether it belongs to another object).
- Reviewer points 4–12: I decided how to address each one and gave the recommendations; Claude applied them:
  - 4: Section 5 now says that mAP50-95 penalises box 6 (a match only at IoU 0.50, a miss at 0.55–0.95), while only
    the image shows that its extra area covers a neighbouring bottle.
  - 5, 6, 7: Section 3 already states that false positives include badly placed boxes on real bottles, gives the
    exact trade-offs (0.10 → 0.25: 733 fewer FPs, 58 more FNs; 0.25 → 0.50: 230 fewer FPs, 104 more FNs), and
    describes the recycling outcome as a possible consequence of my assumed use case. My choice of 0.25 and my
    priority (avoiding missed bottles) are unchanged; Section 3 was not edited.
  - 8: Proposal 1 corrected. Recounted from `data/plastic_bottles_yolo/labels/` (read-only): images with more than
    5 bottles are 37 of 700 test images (5.3%) and 51 of 1,665 training images (3.1%; the reviewer's estimate was
    about 52). The three wrong examples are no longer used as evidence of a general pattern.
  - 9: Proposal 2 now describes the edge-box percentages as a difference in distribution, not worse performance.
  - 10: Wrong 1, box 6 moved from proposal 1 to proposal 3, as a localization and annotation-quality issue.
  - 11: The 15 test polygon conversions were checked against this log (train 1, valid 2, test 15) and
    `label_conversions.csv` (18 rows: test 15, val 2, train 1) before being added to the facts and proposal 3. The 52
    empty test labels are still described as to be checked, not as annotation errors.
  - 12: Proposal 4 now says that validation is used for model selection (`best.pt`), so the 2 shared images weaken
    the test split's independence; it no longer says they were used for training.
- Not done yet: the commit and push, and `A3_2_Boussaffa_Nour_El_Ain_MAI630.pdf`.
