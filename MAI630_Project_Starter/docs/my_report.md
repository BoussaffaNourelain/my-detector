# Assignment 3.2 — Reading what my detector produces

Sections marked **✍️ MY TEXT** are mine to write. Everything else is copied from the verified result files listed
under each section.

## 1. Test-set evaluation

- Model: `all_e30_s416_colab` (YOLO26n, data `all`, 30 epochs, image size 416), weights `best.pt`.
- Data: the **test split** only (held out from training): 700 images, 1,215 labelled bottles, one class
  (`plastic-bottle`).

| Metric (test split) | Value |
|---|---|
| mAP50 | 0.77842 (77.84%) |
| mAP50-95 | 0.63546 (63.55%) |
| mAP50, class `plastic-bottle` | 0.77842 (the only class) |

mAP goes through all the confidence values, so it has no threshold of its own. For comparison, the best
**validation** mAP50 during training was 0.92764 (epoch 29).

Sources: `results/final_test_metrics.json`, `notebooks/03_my_results.ipynb` (cell 3).

## 2. Precision and recall at four confidence thresholds

A prediction counts as correct (TP) when it overlaps a labelled bottle with **IoU ≥ 0.5**. Counts are over the
1,215 test bottles.

| Confidence | TP | FP | FN (missed bottles) | Precision | Recall |
|---|---|---|---|---|---|
| 0.10 | 969 | 1,056 | 246 | 47.85% | 79.75% |
| 0.25 | 911 | 323 | 304 | 73.82% | 74.98% |
| 0.50 | 807 | 93 | 408 | 89.67% | 66.42% |
| 0.75 | 695 | 24 | 520 | 96.66% | 57.20% |

Going from 0.25 to 0.75 removes **216 correct** and **299 wrong** detections (`what_changed`).

Sources: `results/thresholds.json`, `results/thresholds.png`, `notebooks/03_my_results.ipynb` (cell 5).

## 3. The threshold trade-off and my choice

What changes between neighbouring thresholds (from the table above):

| Move | TP lost | FP removed | FN added |
|---|---|---|---|
| 0.10 → 0.25 | 58 | 733 | 58 |
| 0.25 → 0.50 | 104 | 230 | 104 |
| 0.50 → 0.75 | 112 | 69 | 112 |

My priority for this project: **avoiding missed bottles**.

**✍️ MY TEXT — what I win and what I lose between two thresholds, which threshold I would use, and why:**

> Missing a bottle is more serious because it could end up in general waste instead of being recycled.
>
> My choice for the confidence threshold is **0.25**.
>
> I would choose 0.25 because finding more real bottles is more important for this recycling task than reducing
> false alarms. At 0.25, the detector misses 104 fewer bottles than at 0.50.
>
> I accept more false alarms because missing recyclable bottles is more serious for this task. Some non-bottle
> objects may be incorrectly flagged, but the detector will identify more real bottles.
>
> The trade-offs in numbers: from 0.10 to 0.25, the detector gives 733 fewer false positives but misses 58 more
> bottles. From 0.25 to 0.50, it gives 230 fewer false positives but misses 104 more bottles; so at 0.25 I accept
> 230 more false positives than at 0.50 in exchange for 104 fewer missed bottles. Between these two thresholds,
> precision falls from 89.67% (0.50) to 73.82% (0.25), and recall rises from 66.42% to 74.98%.
>
> The false positives I accept are not only non-bottle objects: at IoU 0.5 they also include badly placed boxes on
> real bottles, such as a red and a yellow box on the same object (Section 4).
>
> The recycling outcome is a possible consequence under my assumed use case (a recycling task). The result files do
> not describe how the detector would be used.

## 4. Looking at the images

Figures: `results/examples.png` (original Colab figure) and `results/examples_readable.png` (the same six images,
predictions and scores, with short titles, counts and a legend). Confidence ≥ 0.25, IoU 0.5. Green = true
positive, red = false positive, yellow = false negative.

| Panel | Image | TP | FP | FN |
|---|---|---|---|---|
| Correct 1 | `pwb230_jpg.rf.f1beae829a5f35a6b40f01a4a3538fed.jpg` | 11 | 0 | 0 |
| Correct 2 | `-148_jpg.rf.63d6a88a7c03fd7644688f13d4a43bae.jpg` | 7 | 0 | 0 |
| Correct 3 | `-424_jpg.rf.66e32b01c19f91f6ec3d0b01406d6fa3.jpg` | 6 | 0 | 0 |
| Wrong 1 | `WhatsApp-Image-2024-04-07-at-3-07-21-PM_jpeg.rf.e812a733817e88dece1c575ed8f47ba6.jpg` | 10 | 19 | 16 |
| Wrong 2 | `pile-waste-plastic-bottles-aluminium-can-drinking-paper-coffee-cup-plastic-bottles-garbage-waste-recycle-waste-213734970_webp.rf.5fda3e665a806a2f97f173c3411c7b1a.jpg` | 3 | 21 | 12 |
| Wrong 3 | `qbq9x7rw25n5v5cdae79210829_jpg.rf.dc1b4cb3b4d5c7cf0284b523cb9719c3.jpg` | 18 | 17 | 12 |

The six images are chosen by `pick_examples`: the "correct" ones have no FP or FN, the "wrong" ones have the most
FP + FN. They are not a random sample of the test split.

Mistake types reported by the `image-auditor` subagent (its bag, cup and can readings were marked as uncertain):
a red and a yellow box on the same object; false alarms on objects that may not be bottles; false alarms on
objects that look like bottles; small or partly visible objects at the image edges.

## 5. A detection the metric accepts

Wrong 1 (`WhatsApp-Image-2024-04-07-at-3-07-21-PM_jpeg.rf.e812a733817e88dece1c575ed8f47ba6.jpg`), green box 6
(boxes numbered from the highest confidence to the lowest; the figures do not show these numbers). Wrong 1 has two
green boxes at 0.61 (box 6 and box 7, IoU 0.568) and two at 0.71 (box 2 and box 3); box 6 is the one at x 431–623.
These values were measured with the same model, `predict_images()` and `match()` (confidence ≥ 0.25, IoU 0.5) as
the notebook, in a separate read-only script, and are recorded in `LOG.md`. Coordinates are rounded to whole
pixels.

| | Value |
|---|---|
| Confidence | 0.61 |
| IoU with its label | 0.510 (0.5095 from the unrounded coordinates; 0.509 from the rounded ones below. The limit is 0.5) |
| Predicted box (x1, y1, x2, y2, pixels) | 431, 345, 623, 515 (192 × 170) |
| Its label box | 441, 372, 565, 506 (124 × 134) |
| Predicted area / label area | 1.96 |
| Share of the predicted box that covers its own label box | 51% |
| Share of the neighbouring bottle's label box (475, 329, 635, 627) that lies inside the predicted box | 53% |
| Share of the predicted box that overlaps the neighbouring bottle's label box | 77% |

The 51% and the 77% add up to more than 100% because the two label boxes overlap each other: part of the
predicted box lies inside both of them.

The label box lies entirely inside the predicted box. The neighbouring bottle is detected separately (Wrong 1,
box 3 at 470, 345, 634, 634, confidence 0.71, IoU 0.897), so this box does not cause a missed bottle. With a confidence of 0.61, it is also kept at the 0.50
threshold.

**✍️ MY TEXT — whether I would accept this detection for my project, and why:**

> I would not accept this detection. Box 6 has a confidence of 0.61 and an IoU of 0.510 with its label, just above
> the 0.50 evaluation threshold, so the metric counts it as a true positive. However, the predicted box is about
> 1.96 times the area of the labelled box and overlaps the label box of a neighbouring bottle. That neighbouring
> bottle is detected separately, so this box does not cause a missed bottle, but it is still a poor detection
> because it is too large and covers part of another bottle.
>
> This shows a limitation of relying only on IoU-based evaluation. At an IoU threshold of 0.50, the metric only
> checks whether the predicted box overlaps its own label enough; the extra area lowers the IoU, but the metric
> does not check whether that extra area belongs to another object. In crowded scenes, a box that covers parts of two
> bottles can therefore be counted as fully correct. Precision, recall and mAP50 count this detection as a success,
> so inspecting the images is needed to see this kind of error.
>
> mAP50-95 does penalise this box: with an IoU of 0.510 it is a match only at the 0.50 threshold and a miss at all
> nine stricter thresholds (0.55 to 0.95). But no IoU threshold says where the extra area lies; only the image shows
> that it covers a neighbouring bottle.

## 6. What I would change about my data

Facts about the dataset (from the Module 2 dataset review and the Module 3 box statistics in `LOG.md`):

- Training split: 1,665 images, 2,515 boxes. Test split: 700 images, 1,215 boxes.
- Only 2 of the 1,215 test boxes are small (under 32 × 32 pixels).
- 29.8% of test boxes touch an image edge, compared with 24% in training.
- Images with more than 5 bottles: 5.3% of test images (37 of 700), compared with 3.1% of training images
  (51 of 1,665).
- 417 images have empty label files (train 288, val 77, test 52).
- 18 polygon label lines were converted to their enclosing boxes; 15 of them are in the test split (train 1, val 2;
  `data/plastic_bottles_yolo/label_conversions.csv`).
- 2 images appear in both the validation and the test split (2 of the 700 test images).
- The three "wrong" examples hold 26, 15 and 30 labelled bottles; the three "correct" examples hold 11, 7 and 6.

**✍️ MY TEXT — my data-improvement proposal (what I would change about the data, and why):**

> I would change the data in four ways.
>
> 1. **More crowded scenes with overlapping bottles.** Images with more than five bottles are a larger share of the
>    test split (5.3%, 37 images) than of the training split (3.1%), although the training split holds more of them
>    in absolute numbers (51). The three "wrong" examples contain 26, 15 and 30 labelled bottles, but they were
>    selected for having the most FP + FN, so they do not show a general pattern in the test split. Adding more
>    crowded and overlapping bottle scenes to the training split would give the model more examples of these
>    situations.
> 2. **More bottles that touch the image edge or are partly visible.** 29.8% of the test boxes touch an image edge,
>    compared with 24% of the training boxes. This is a difference between the two distributions; the results do not
>    show that the detector performs worse on these boxes. I would add training images with cut-off and partly hidden
>    bottles so that the training data better matches the test data.
> 3. **An annotation-quality review.** I would check the labels in crowded images, where label boxes of
>    neighbouring bottles overlap. Wrong 1, box 6 is an example of both a localization and an annotation-quality
>    issue: the predicted box is loose (IoU 0.510, 1.96 times the label area), and its label box overlaps the label
>    box of the neighbouring bottle. I would also check the 15 test labels converted from polygons, to confirm that
>    each enclosing box fits its bottle, and the 417 images with empty label files (288 train, 77 validation,
>    52 test), to confirm whether each one is a true background image or an image with missing annotations. Errors
>    in either case would affect both training and the test metrics.
> 4. **Checking the validation/test overlap.** Two of the 700 test images also appear in the validation split. The
>    validation split is used for model selection (it chose the best epoch and so `best.pt`), so these two images
>    weaken the independence of the test split. I would remove the duplicates from one of the two splits.
>
> One limitation of the test split itself is that it contains only 2 boxes smaller than 32 × 32 pixels, so these
> results cannot show how well the detector finds small or distant bottles.
