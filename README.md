# MRI + Clinical Multimodal pCR Prediction (Cancer-Net BCa)

Seminar project: does adding clinical data to MRI images improve prediction of
**pathological complete response (pCR)** in breast cancer patients?

We compare two models on the same 70/15/15 patient-level split:

| Model | Inputs | Architecture |
|---|---|---|
| `unimodal` (baseline) | MRI slices only | ResNet50 embedding → MLP head |
| `multimodal` | MRI slices + clinical features | ResNet50 embedding ⊕ clinical MLP embedding → MLP head (late concatenation fusion) |

The multimodal model is expected to reach a higher ROC-AUC / F1 than the
MRI-only baseline, demonstrating the value of the clinical modality.

---

## Table of contents

- [Dataset](#dataset)
- [Project layout](#project-layout)
- [Setup](#setup)
- [Configuration](#configuration)
- [How it works](#how-it-works)
- [Usage](#usage)
  - [Train the MRI-only baseline](#1-train-the-mri-only-baseline)
  - [Train the MRI + clinical model](#2-train-the-mri--clinical-model)
  - [Evaluate & compare](#3-evaluate--compare)
  - [Single-patient inference](#4-single-patient-inference)
  - [Explainability (Grad-CAM & SHAP)](#5-explainability-grad-cam--shap)
  - [Finalise deliverables](#6-finalise-deliverables)
  - [Synthetic smoke test](#7-synthetic-smoke-test)
- [Results (real data)](#results-real-data)
- [Outputs](#outputs)
- [Known limitations](#known-limitations)

---

## Dataset

**Cancer-Net BCa** (Kaggle: `amytai/cancernet-bca`, arXiv:2211.05308).

* `CDIs_images_nifti/` — volumetric NIfTI contrast-enhanced MRI per patient.
* `metadata.csv` — patient-level clinical variables and the binary **pCR** label.

Volumes are converted to 2D **axial slices** (axis 2) once, cached as PNGs in
`outputs/slices`, and then fed to ResNet50. Volume filenames in this cohort end
with a `_CDIs_img` suffix which is stripped to match `metadata.csv` patient ids.
Place the Kaggle download under `data/`:

```
data/
├── CDIs_images_nifti/
│   ├── BCA_0001.nii.gz
│   └── ...
└── metadata.csv
```

The clinical feature columns are configured in `configs/multimodal.yaml`
(`age`, `race`, `lesion type`, `HR/HER2`, `MRLD`, `analysis cohort`). `SBR
grade` is excluded by design.

---

## Project layout

```
project/
├── config.py                  # dataclass Config schema + YAML (de)serialisation
├── configs/
│   ├── unimodal.yaml          # MRI-only baseline settings
│   └── multimodal.yaml        # MRI + clinical settings
├── datasets/                  # loading, slicing, preprocessing, split
│   ├── loader.py              # NIfTI discovery + slice extraction
│   ├── manifest.py            # patient-aligned manifest + label mapping
│   ├── preprocess.py          # tabular preprocessor (fit-on-train only)
│   ├── augment.py             # train/val image transforms
│   ├── dataset.py             # PyTorch dataset + collate
│   └── split.py               # patient-level 70/15/15 stratified split
├── models/
│   ├── image_encoder.py       # frozen ResNet50 backbone
│   ├── clinical_mlp.py        # Linear→ReLU→Dropout→Linear
│   ├── components.py          # shared MLP head
│   ├── unimodal.py            # MRI-only classifier
│   └── multimodal.py          # late concat fusion classifier
├── training/
│   ├── feature_cache.py       # cached frozen-backbone image embeddings
│   ├── losses.py              # weighted cross-entropy
│   ├── metrics.py             # accuracy/precision/recall/F1/AUC/CM
│   ├── scheduler.py           # AdamW + cosine schedule
│   ├── trainer.py             # training loop, early stopping, checkpointing
│   └── pipeline.py            # end-to-end experiment orchestrator
├── evaluation/
│   ├── report.py              # confusion matrix + ROC figures, JSON/CSV
│   └── comparison.py          # comparison table (CSV + markdown)
├── explainability/
│   ├── gradcam.py             # manual Grad-CAM (no external dependency)
│   └── shap_explain.py        # SHAP on the clinical modality
├── utils/                     # seeding, logging, IO helpers
├── scripts/synthetic_smoke.py # end-to-end validation on synthetic data
├── train_unimodal.py          # entrypoint: train MRI-only baseline
├── train_multimodal.py        # entrypoint: train MRI + clinical model
├── evaluate.py                # test evaluation + comparison table
├── inference.py               # single-patient prediction
├── explain.py                 # Grad-CAM + SHAP artefacts
├── requirements.txt
└── README.md
```

---

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Python 3.12, CPU-only torch + torchvision were used during development; the
project also runs on GPU if available (`training.device: auto`).

---

## Configuration

All experiment settings live in YAML and load into the nested `Config`
dataclass (`config.py`). Key sections:

* `paths.*` — all input/output directories.
* `data.*` — label column, clinical feature columns, split ratios, `slice_axis`.
* `image.*` — backbone (`resnet50`), `freeze_backbone`, embedding size, feature
  cache.
* `clinical.*` — imputation strategy, standardisation, categorical columns.
* `model.*` — dropout, clinical embedding dim, fusion hidden sizes.
* `training.*` — AdamW lr, epochs, batch size, scheduler, early stopping, seed.

The **clinical preprocessor is fit on the training split only**; validation and
test splits are transformed with those fitted parameters (no leakage). Numeric
columns are median-imputed and standardised; non-numeric columns are
auto-detected and one-hot encoded.

---

## How it works

1. **Manifest** — each patient's NIfTI volume is sliced (axial), cached as PNG,
   and joined with clinical metadata into one row-per-slice DataFrame.
2. **Split** — patients are split 70/15/15 with `StratifiedShuffleSplit`, so a
   patient never appears in two splits.
3. **Clinical pre-processing** — fit on train, transform all.
4. **Image feature cache** — with a frozen backbone, every slice's 512-d ResNet50
   embedding is computed once and cached (`outputs/feature_cache/`), so training
   is fast even on CPU.
5. **Model** — `UnimodalModel` (`image_feat → head`) or `MultimodalModel`
   (`concat(image_feat, clinical_emb) → head`).
6. **Training** — AdamW (lr 1e-4), cosine LR, balanced class-weighted
   cross-entropy (pCR ≈ 2× weight), early stopping with best-state checkpointing
   (best model selected by **validation ROC-AUC**, more robust under class
   imbalance than accuracy).
7. **Evaluation** — metrics on the held-out test split, confusion matrix + ROC +
   precision-recall figures, predictions CSV, and a comparison table of both
   models.
8. **Finalise** — `scripts/finalize.py` aggregates everything into
   `outputs/final_results.md`, tables and combined figures.

---

## Usage

All commands run from the `project/` directory. The active virtualenv must have
the real Cancer-Net BCa data under `data/` (or `--config` pointing elsewhere).

### 1. Train the MRI-only baseline

```bash
python train_unimodal.py --config configs/unimodal.yaml
```

### 2. Train the MRI + clinical model

```bash
python train_multimodal.py --config configs/multimodal.yaml
```

Each run writes a checkpoint (`checkpoints/<experiment>/best_model.pt`), test
metrics JSON, predictions CSV, and confusion-matrix / ROC figures.

### 3. Evaluate & compare

```bash
# Evaluate a single experiment on the held-out test split
python evaluate.py --config configs/unimodal.yaml --experiment unimodal
python evaluate.py --config configs/multimodal.yaml --experiment multimodal

# Build the comparison table from the saved metrics JSON files
python evaluate.py --compare
```

`--compare` writes `outputs/comparison/comparison_table.{csv,md}` summarising
accuracy / precision / recall / F1 / ROC-AUC for both models.

### 4. Single-patient inference

```bash
python inference.py \
  --config configs/multimodal.yaml \
  --checkpoint checkpoints/multimodal/best_model.pt \
  --image data/CDIs_images_nifti/BCA_0001.nii.gz \
  --clinical-csv sample_clinical.csv
```

`--image` accepts a NIfTI volume or a cached PNG slice; `--clinical-csv` is a
one-row CSV whose columns match the configured clinical features (required for
the multimodal model, ignored by the unimodal model).

### 5. Explainability (Grad-CAM & SHAP)

```bash
python explain.py --config configs/multimodal.yaml
```

* **Grad-CAM** overlays on sample test MRI slices → `outputs/figures/gradcam/`.
  The implementation uses forward/backward hooks on the last bottleneck block
  (manual, no external dependency).
* **SHAP** beeswarm + feature-importance plots for the clinical modality →
  `outputs/shap/`. Clinical features are perturbed while each patient's image
  embedding is kept fixed, so attributions reflect the clinical contribution on
  top of the MRI evidence.

### 6. Finalise deliverables

Assemble publication-ready outputs from the trained runs (combined ROC / PR
curves, training curves, tables, checkpoint copies and the auto-generated
`outputs/final_results.md` report):

```bash
python scripts/finalize.py
```

### 7. Synthetic smoke test

Validates the whole pipeline end-to-end without the real dataset by generating
synthetic NIfTI volumes + `metadata.csv`, training both models with a
randomly-initialised backbone, and checking all artefacts:

```bash
python scripts/synthetic_smoke.py --config configs/multimodal.yaml --outdir data_synthetic
```

---

## Results (real data)

Ran on the full Cancer-Net BCa cohort (253 patients: 171 no-pCR / 82 pCR),
single axial slice per patient, 70/15/15 patient-level split
(177 train / 38 val / 38 test).

Test-split comparison:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| MRI-only (`unimodal`) | 0.526 | 0.517 | 0.519 | 0.504 | 0.500 |
| MRI + clinical (`multimodal`) | 0.658 | 0.584 | 0.571 | 0.572 | **0.696** |

Interpretation:

* Adding clinical data lifts **ROC-AUC from 0.500 → 0.696** (+0.196) and **F1
  from 0.504 → 0.572**, clearly demonstrating the value of the clinical modality
  over MRI alone.
* The MRI-only model is close to chance: a single mid-axial slice encoded by a
  **frozen ImageNet backbone** carries little pCR signal in this cohort.
* SHAP (clinical modality, image embedding held fixed) ranks **age**, **lesion
  type (Single mass)**, **analysis cohort**, **HR/HER2 (TN)**, **MRLD** and
  **race** as the top drivers of the multimodal prediction.

All metrics, per-slice predictions, confusion matrices and ROC curves are under
`outputs/` (see [Outputs](#outputs)).

---

## Outputs

```
checkpoints/<experiment>/best_model.pt   # best-state checkpoints (per run)
outputs/
├── final_results.md                     # auto-generated experiment report
├── <experiment>_metrics.json            # test metrics
├── <experiment>_test_predictions.csv    # per-slice predictions + probabilities
├── figures/
│   ├── confusion_matrix_<experiment>.png
│   ├── roc_curve.png                    # both models overlaid
│   ├── precision_recall_curve.png       # both models overlaid
│   ├── train_loss.png / val_loss.png    # training curves (both models)
│   ├── train_accuracy.png / val_accuracy.png
│   ├── shap_summary.png                 # SHAP summary (pCR class)
│   └── gradcam/                         # Grad-CAM overlays
├── tables/
│   ├── metrics.csv                      # aggregate metrics per model
│   ├── comparison_table.csv             # unimodal vs multimodal
│   ├── classification_report.csv        # per-class precision/recall/f1/support
│   ├── confusion_matrix_<experiment>.csv
│   └── shap_feature_importance.csv      # ranked clinical variables
├── checkpoints/                         # final deliverable checkpoints
│   ├── unimodal_best.pth
│   └── multimodal_best.pth
├── shap/                                # raw SHAP figures + importance CSV
├── slices/                              # cached 2D MRI slices
├── feature_cache/                       # cached ResNet50 embeddings
├── logs/                                # histories, configs, test predictions
└── comparison/                          # comparison table (csv + markdown)
```

---

## Known limitations

* **CPU-only**: the ResNet50 backbone is frozen and image features are cached,
  which is the intended fast workflow. Full fine-tuning is possible by setting
  `image.freeze_backbone: false`.
* **Frozen-backbone MRI features**: with a frozen ImageNet backbone the
  MRI-only model is near chance (ROC-AUC ≈ 0.50); a fine-tuned or
  MRI-specific encoder would be needed to extract more image signal.
* **Single slice per volume** is the default (`n_slices_per_volume: 1`); the
  loader supports several slices per patient for slice-level augmentations.
* SHAP's `PermutationExplainer` is slow on large clinical matrices; `n_explain`
  can be reduced in `explain.py`.
* The volume filenames in this cohort include a `_CDIs_img` suffix
  (`ACRIN-6698-XXXXXX_CDIs_img.nii`); `datasets/loader.py` strips known suffixes
  so volumes match `metadata.csv` patient ids.
