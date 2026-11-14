# My training experiment

Fill this in **before** you train. Do not use Claude to write it. Claude may question it afterwards.

## The data I chose
Which option: all / half / one class / large only: all (the complete training dataset).

## Why I chose it
I believe using all available training images is better because it gives the model more examples to learn from.

## What I expect to happen
My prediction, in one or two sentences: I expect the model to learn useful visual features of plastic bottles from the complete dataset, but I think it may still struggle with small or partially visible bottles.

---
Fill this in **after** training.

## What actually happened
Epochs, minutes and best mAP50: 30 epochs, 22.2 minutes, best validation mAP50 0.92764 (92.764%) at epoch 29.

Training setup:
- Data choice: `all` (1,665 training images)
- Model: YOLO26n
- Image size: 416, batch size: 8
- Hardware: Google Colab, Tesla T4 GPU
- Overfitting flag: False

Test-set evaluation (700 images, 1,215 ground-truth instances):
- Test mAP50: 0.77842 (77.84%)
- Test mAP50-95: 0.63546 (63.55%)

Precision and recall at each confidence threshold (test set):

| Confidence | Precision | Recall |
|---|---|---|
| 0.10 | 47.85% | 79.75% |
| 0.25 | 73.82% | 74.98% |
| 0.50 | 89.67% | 66.42% |
| 0.75 | 96.66% | 57.20% |

## Was my prediction right? What did I learn?
My prediction was partly correct. Training with the complete dataset helped the model learn useful visual features for detecting plastic bottles. However, the test mAP50 was 77.84%, compared with 92.76% on the validation set. This difference suggests that the model performed less effectively on unseen images, highlighting the challenge of generalization.

I also learned that the confidence threshold affects the balance between precision and recall. At a confidence threshold of 0.50, precision reached 89.67%, while recall was 66.42%. This means the model made fewer false-positive detections but missed more actual bottles. Therefore, choosing a confidence threshold depends on whether reducing false detections or finding more bottles is the priority.
