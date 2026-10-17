# Rice model training

`train_rice.py` audits the dataset, makes a leakage-safe split, trains
EfficientNetV2-B0 (preprocessing built in, same input contract as the Maize
and Cotton models), evaluates per class and exports `rice_model.tflite` +
`labels.txt`, then verifies the export against the app's requirements.

**Status:** only a synthetic smoke test has run (`tests/test_training.py`).
It has not been trained on real rice data; no accuracy has been measured.

## What is needed

1. **The dataset**: the original images, one folder per class, not split:

   ```
   rice_dataset/
     Rice_Bacterial_Leaf_Blight/  Rice_Brown_Spot/  Rice_Healthy/
     Rice_Leaf_Blast/  Rice_Leaf_scald/  Rice_Sheath_Blight/
     Weed_Broadleaf/  Weed_Grass/
   ```

   (`Rice_Healthy_Rice_Leaf` is accepted for the healthy folder.) The old
   notebook built `/content/agri_split` on Colab's temporary disk, so it has
   to be recreated from the source images.

2. **Optional baseline**: an existing Rice `.tflite` with 8 outputs, to
   evaluate on the same test split before and after.

## Run on Colab (GPU)

```python
from google.colab import drive
drive.mount('/content/drive')
!pip -q install scikit-learn ai-edge-litert
!git clone https://github.com/<you>/agri-doctor-pro.git
%cd agri-doctor-pro/training
!python train_rice.py --audit-only --data /content/drive/MyDrive/rice_dataset --out /content/rice_run
!python train_rice.py --data /content/drive/MyDrive/rice_dataset --out /content/rice_run
!cp /content/rice_run/rice_model.tflite /content/drive/MyDrive/
```

Read `rice_run/data_report.json` first: class counts, imbalance ratio,
near-duplicate groups (kept within one split) and possible label conflicts
(near-identical images with different labels).

`rice_run/metrics.md` has per-class precision, recall, F1 and the confusion
matrix for the exported TFLite model on the held-out test split.

## Install in the app

Share `rice_model.tflite` in Drive as "Anyone with the link → Viewer" and
set `RICE_MODEL_ID` in `app.py` to its file ID. The app validates the file
(TFLite signature, size, 8 outputs) before using it; then
`pytest tests/test_models.py` runs the real-model checks on it.

## Local smoke test

```bash
python3.12 -m venv training/.venv
training/.venv/bin/pip install -r training/requirements-train.txt
training/.venv/bin/python -m pytest tests/test_training.py
```
