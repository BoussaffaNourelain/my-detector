# My own improvement (Assignment 2.2)
Write this yourself, before asking Claude for anything. Then give it to Claude with docs/data_spec.md and ask for a plan.

**What it does (two or three sentences):**
A function finds the images whose YOLO label files are empty, in every split, and exports their paths to
`results/unlabelled_images.csv`. Empty labels are not automatically errors: the function only identifies these images
for review and never changes or deletes any image or label.

**Why my dataset needs it (what you saw in the data that made you want it):**
The dataset has 417 empty label files: 288 in train, 77 in validation and 52 in test. I want to inspect whether they
are intentional negative examples or potentially missing annotations.

**Where it appears:** (a new function in src/data.py, a new part of the data page, or both)
A new function in `src/data.py`, with a CSV report.

**How I will know it works (the test Claude must write in tests/test_my_feature.py):**
A test creates a small temporary dataset containing both empty and non-empty label files, then checks that the
function identifies exactly the empty ones and writes the correct CSV.

Ideas if you need one: a histogram of box sizes per class on the data page; a check that flags very thin or very
elongated boxes; a button that writes the files with problems to results/problems.csv; showing only the images that
contain a chosen class in the image browser. Your own idea is better.
