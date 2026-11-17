# My detector (MAI 630)

My own computer-vision project, built module by module.

| Module | What this repository gains |
|---|---|
| 1 | Environment, `src/classical.py`, notebook `01_classical_cv.ipynb` (Assignments 1.1 and 1.2) |
| 2 | My project description and dataset; the data page |
| 3 | Training page and evaluation |
| 4 | The detection page, the full app and this README |

## What it does
It detects plastic bottles (one class, `plastic-bottle`) in photos like those of the Plastic Bottle 2.0 dataset, and
draws a box with a confidence score around each bottle it finds. It is meant for a plastic-bottle recycling task, where
finding the real bottles matters more than avoiding a few false alarms.

## Install
You need Git and **Python 3.12**. The packages come from `requirements.txt` (do not install others).

1. Clone the repository. The project is in the folder `MAI630_Project_Starter` inside it:
   ```bash
   git clone https://github.com/BoussaffaNourelain/my-detector.git
   cd my-detector/MAI630_Project_Starter
   ```
2. Create a virtual environment with Python 3.12 and activate it.

   Windows (PowerShell):
   ```powershell
   py -3.12 -m venv .venv
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .venv\Scripts\Activate.ps1
   ```
   Windows blocks scripts such as `Activate.ps1` by default. The second line allows them only in the current
   PowerShell process: nothing changes for other windows, and the setting ends when you close this one.
   In Command Prompt (cmd), activate with `.venv\Scripts\activate.bat` instead.
   Mac/Linux:
   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   ```
3. Install the packages, register the notebook kernel and check the setup (same commands on every system):
   ```bash
   pip install -r requirements.txt
   python -m ipykernel install --sys-prefix --name mai630 --display-name "Python (mai630)"
   python check_setup.py
   ```
   `check_setup.py` must print `READY`. Every command below is run from `MAI630_Project_Starter`, with the
   environment activated.
4. Optional: check the code with the tests (`python -m pytest -q tests`).

## Get the data
The dataset is not in the repository (`data/` is git-ignored). It is **Plastic Bottle 2.0, version 2**, by
`somaiya-vlmqe` on Roboflow Universe, licence CC BY 4.0:
https://universe.roboflow.com/somaiya-vlmqe/plastic-bottle-2.0-psuby/dataset/2

1. On that page, download the dataset in the **YOLOv8** format as a zip file (Roboflow may ask you to sign in).
2. Extract the zip into `data/raw/plastic-bottle-2.0-v2/`, so that this folder directly holds `train/`, `valid/`,
   `test/` and `data.yaml`. This is the folder written in `project.yaml` (`source: folder:`).
3. Convert it by running the notebook `notebooks/02_my_dataset.ipynb` (kernel "Python (mai630)"), or from the terminal:
   ```bash
   jupyter nbconvert --to notebook --execute --inplace notebooks/02_my_dataset.ipynb
   ```
The converted dataset ends up in `data/plastic_bottles_yolo/` (YOLO format, `images/` and `labels/` with
`train` / `val` / `test`). The published split is kept as downloaded: 1,665 training, 1,394 validation and 700 test
images. The conversion refuses to write into a folder that is not empty, so delete `data/plastic_bottles_yolo/` first
if you want to convert again. Never edit the files in it by hand.

## Train
Start the app with `python app/app.py`, open the address it prints (usually http://127.0.0.1:7860) and go to the
page **2 · Training**:
1. Choose the training data (`all`, `half`, `one_class` or `large`); the page shows how many images that means.
2. Write why you chose this data (saved as `runs/<run name>_why.txt`).
3. Choose the epochs, the image size and a run name, then press **Start training**.

The training runs in a separate process; the page shows the progress, the losses and mAP50 every 5 seconds, and its
output goes to `runs/<run name>.log`. The model is saved in `runs/<run name>/weights/best.pt`.
The first training needs an internet connection: Ultralytics downloads the starting weights `yolo26n.pt` automatically.

How long it took for me, with the data `all` (1,665 training images) and image size 416:
- 2 epochs on my laptop CPU (Intel Core i5-10300H): 21.3 minutes (about 10.6 minutes per epoch).
- 30 epochs on Google Colab with a Tesla T4 GPU: 22.2 minutes. This run, `all_e30_s416_colab`, is my final model.

## Detect
Start the app with `python app/app.py` and go to the page **3 · Detection & results**. The line at the top of the app
says which phases are ready (data, training, results) and what to do when one is missing.
1. Choose a finished run. The page shows its mAP50 and why it was trained. Trained models are not in the repository
   (`runs/` and `*.pt` are git-ignored): train one first (see Train). A run appears here when
   `runs/<run name>/weights/best.pt` exists; restart the app after a training ends.
2. Choose a photo from the test split, or upload your own, and press **Detect**. The model runs once on it.
   The test-split photos are available only after the steps in Get the data; without them, upload your own photo.
3. Move the confidence slider. It starts at my threshold from `project.yaml` (`threshold: 0.25`). The boxes, the
   sentence and the table of boxes per class follow it without running the model again.

At the bottom, the page shows my Module 3 results (`results/thresholds.png`, `results/examples.png` and
`docs/experiment.md`).

## Results
On the test split (700 images, 1,215 labelled bottles), my final model `all_e30_s416_colab` reaches **mAP50 0.778** and
**mAP50-95 0.635**. I chose the confidence threshold **0.25** (precision 73.82%, recall 74.98% at IoU 0.5), because
finding more real bottles is more important for this recycling task than reducing false alarms.

## What doesn't work
The detector does not identify every plastic bottle correctly. Small bottles are underrepresented in the test dataset, and bottles touching image edges may be harder to detect. Some predictions may be missed or may incorrectly identify objects as bottles. The trained model is also not included in the GitHub repository, so users must train their own model before using the detection page.
