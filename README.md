# Microsatellite Status Classification from TCGA‑COAD Whole‑Slide Tiles

**Project goal:** Predict microsatellite instability (MSI) vs. microsatellite stability (MSS) in colorectal cancer from H&E whole‑slide image (WSI) tiles, offering a fast, low‑cost screening alternative to DNA‑based assays.

---

## 1&nbsp;&nbsp;Dataset

| Source | Tiles | Patients | Classes | Format |
|--------|------:|---------:|---------|--------|
| [TCGA‑COAD MSI/MSS tiles on Kaggle](https://www.kaggle.com/datasets/joangibert/tcga_coad_msi_mss_jpg/data) | **192 312** | 432 | MSS / MSI | JPG (224 × 224) |

Tiles were extracted from diagnostic WSIs following the preprocessing pipeline of Kather *et&nbsp;al.* (2019) and organised in class‑named folders (`train/msi`, `train/mss`, …).

---

## 2&nbsp;&nbsp;Method

We fine‑tuned three ImageNet‑pre‑trained backbones—**ResNet‑18**, **VGG‑19**, and **VGG‑19‑BN**—using the pipeline in `refined_image_classification.py`:

| Step | Details |
|------|---------|
| **Splits** | 60 % train · 25 % val · 15 % test (stratified, patient‑level) |
| **Transforms** | `RandomResizedCrop(224)`, `RandomHorizontalFlip()`, per‑channel normalization |
| **Optimizer** | AdamW (LR = 3 × 10⁻⁴) + CosineAnnealingLR |
| **Training** | Early stopping (patience = 5); best‑epoch checkpoint |
| **Metrics** | Accuracy, balanced accuracy, AUROC, confusion matrix |

---

## 3&nbsp;&nbsp;Results

| Model | Test Accuracy | Balanced Acc. | AUROC |
|-------|--------------:|--------------:|------:|
| ResNet‑18 | 0.64 | 0.61 | 0.67 |
| VGG‑19 | 0.65 | 0.63 | 0.67 |
| **VGG‑19‑BN** | **0.66** | **0.64** | **0.69** |

Our best AUROC (0.69) is lower than the 0.77 reported by Kather *et&nbsp;al.* (2019), suggesting room for improvement via stronger augmentations and patient‑level aggregation.

---

## 4&nbsp;&nbsp;Usage

```bash
# 1. Install dependencies
conda create -n msi python=3.10 pytorch torchvision torchaudio cudatoolkit -c pytorch
pip install -r requirements.txt   # matplotlib, scikit-learn

# 2. Download and extract the dataset
unzip tcga_coad_msi_mss_jpg.zip -d data/

# 3. Train & evaluate
python CNN.py \
       --data_dir data/jpg \
       --arch vgg19_bn \
       --epochs 30 \
       --batch_size 32 \
       --output_dir runs/vgg19bn


Artifacts in `output_dir/`:

* `best_model.pt` – fine‑tuned weights  
* `confusion_matrix.png` – test‑set confusion matrix  
* `test_metrics.json` & `history.json` – all metrics and learning curves

---

## 5&nbsp;&nbsp;Next Steps

1. **Stronger augmentations** (color jitter, stain normalisation) to reduce centre/stain bias.  
2. **Hard‑tile mining**: oversample under‑represented patients & difficult regions.  
3. **Multiple‑Instance Learning** to aggregate tile scores at patient level.  
4. Replace classifier head with **ArcFace** or **AAM‑Softmax** for better margin separation.  
5. Hyper‑parameter search (LR, scheduler, optimiser, mix‑up).

---

## 7&nbsp;&nbsp;License

Code: MIT © 2025 Yang Liu  
Dataset: subject to TCGA data usage policies (dbGaP phs000178).
