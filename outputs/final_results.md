# Experiment Summary

## Dataset Summary

- Number of patients: **253** (253 image slices).
- Class distribution: no pCR = 171, pCR = 82.
- Patient-level split (train/val/test): 177 / 38 / 38.

## Model Configurations

### unimodal

| Component | Setting |
|---|---|
| Image backbone | `resnet50` (frozen=True) |
| Image embedding dim | 512 |
| Clinical embedding dim | n/a (image-only) |
| Fusion hidden | [128] |
| Dropout | 0.3 |
| Optimizer | adamw (lr=0.0001) |
| Batch size | 8 |
| Max epochs | 30 (early stop after 8) |
| Scheduler | cosine |
| Loss | CrossEntropyLoss (class weighting = balanced) |

### multimodal

| Component | Setting |
|---|---|
| Image backbone | `resnet50` (frozen=True) |
| Image embedding dim | 512 |
| Clinical embedding dim | 32 |
| Fusion hidden | [128] |
| Dropout | 0.3 |
| Optimizer | adamw (lr=0.0001) |
| Batch size | 8 |
| Max epochs | 30 (early stop after 8) |
| Scheduler | cosine |
| Loss | CrossEntropyLoss (class weighting = balanced) |

## Results Table (held-out test split)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| unimodal | 0.5263 | 0.5167 | 0.5192 | 0.5043 | 0.5000 |
| multimodal | 0.6579 | 0.5843 | 0.5705 | 0.5723 | 0.6955 |

## Discussion

- **Which model performed better?** The **multimodal** model achieved the highest ROC-AUC (0.696 vs 0.500) and F1 (0.572 vs 0.504).
- **How much improvement was observed?** Adding clinical data improved ROC-AUC by **+0.196** and F1 by **+0.068** on the held-out test split.
- **Did clinical data improve prediction?** Yes. The multimodal model, which fuses the MRI embedding with clinical features (age, lesion type, HR/HER2, MRLD, analysis cohort, race), consistently outperformed the MRI-only baseline. SHAP analysis ranks age, lesion type (Single mass), analysis cohort, HR/HER2 (triple-negative) and MRLD among the most influential clinical predictors.
- **Limitations.** The MRI-only model relies on a single mid-axial slice encoded by a frozen ImageNet-pretrained ResNet50; its near-chance performance (ROC-AUC ≈ 0.50) suggests limited image signal with this representation. Training used class-weighted cross-entropy on 177 train patients with early stopping, so the gains should be interpreted for this small cohort rather than as a generalisation guarantee.
