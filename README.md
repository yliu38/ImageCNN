Microsatellite Status Classification from TCGA-COAD Whole-Slide Tiles
Project goal: predict microsatellite instability (MSI) vs. microsatellite stability (MSS) in colorectal cancer (COAD) from H&E whole-slide image (WSI) tiles, enabling a fast, low-cost screening alternative to DNA sequencing–based assays.

1 Dataset
Source	Tiles	Patients	Classes	Format
TCGA-COAD MSI/MSS tiles on Kaggle	192 312 JPG tiles (224 × 224)	432	MSS / MSI	JPG

Tiles were extracted from diagnostic WSIs following the preprocessing pipeline of Kather et al. (2019) and are organised in class-named folders (train/msi, train/mss, …).

2 Method
We fine-tuned three ImageNet-pre-trained backbones—ResNet-18, VGG-19, and VGG-19-BN—with the following pipeline (see refined_image_classification.py):

Step	Details
Splits	60 % train · 25 % val · 15 % test (stratified, patient-level)
Transforms	RandomResizedCrop(224), RandomHorizontalFlip(), per-channel normalization
Optimiser	AdamW (LR = 3 × 10⁻⁴), CosineAnnealingLR
Training	Early stopping (patience = 5); best epoch checkpoint
Metrics	Accuracy, balanced accuracy, AUROC, confusion matrix

3 Results
Model	Test Accuracy	Balanced Acc.	AUROC
ResNet-18	0.64	0.61	0.67
VGG-19	0.65	0.63	0.67
VGG-19-BN	0.66	0.64	0.69

Our best AUROC (0.69) falls short of the 0.77 reported by Kather et al. (2019) despite similar data volume and augmentation. 

4 Usage
bash
Copy
Edit
# 1. Install deps
conda create -n msi python=3.10 pytorch torchvision torchaudio cudatoolkit -c pytorch
pip install -r requirements.txt   # matplotlib, scikit-learn

# 2. Download / extract the Kaggle dataset
unzip tcga_coad_msi_mss_jpg.zip -d data/

# 3. Train & evaluate
python refined_image_classification.py \
    --data_dir data/jpg \
    --arch vgg19_bn \
    --epochs 30 \
    --batch_size 32 \
    --output_dir runs/vgg19bn

# 4. Inference on new tiles
python predict.py --weights runs/vgg19bn/best_model.pt --img img.jpg
Artifacts written to output_dir/:

best_model.pt – fine-tuned weights

confusion_matrix.png – test-set confusion matrix

test_metrics.json & history.json – metrics and learning curves

5 Next Steps
Stronger augmentations (color jitter, stain-normalisation) to reduce stain/centre bias.

Hard-tile mining: sample under-represented patients & difficult regions.

Multiple-Instance Learning to aggregate tile scores at patient level.

Replace classifier head with ArcFace or AAM-Softmax for better margin separation.

Hyper-parameter search (learning rate, LR scheduler, optimiser, mix-up).
