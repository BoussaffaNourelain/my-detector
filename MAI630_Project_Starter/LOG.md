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
