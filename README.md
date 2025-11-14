/*
# Longitudinal Brain MRI Augmentation & Segmentation Pipeline

> A Two-Generator PGGAN pipeline for synthetic longitudinal MRI generation, followed by 3D segmentation and RANO classification.

---

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [Dataset Format](#dataset-format)
- [Installation](#installation)
- [Usage: End-to-End Pipeline](#usage-end-to-end-pipeline)
- [Configuration](#configuration)
- [Output Directory Guide](#output-directory-guide)
- [Tips & Notes](#tips--notes)
- [Contributing](#contributing)
- [License & Citation](#license--citation)
- [Contact](#contact)

---

## Overview

This project provides an end-to-end pipeline for augmenting **longitudinal brain MRI data** using a **Two-Generator Progressive GAN (PGGAN)**.

The system generates anatomically consistent **baseline and follow-up** image pairs. This synthetic data is used to augment real, often limited, datasets to improve the performance of **3D segmentation** models and subsequent **RANO classification** models.

### End-to-End Flow

1.  **Train GAN:** The Two-Generator PGGAN is trained on preprocessed real data (from `CACHED128`).
2.  **Generate Data:** The trained GAN generates synthetic `.npz` files (baseline + follow-up pairs) and saves them (to `generated1070`).
3.  **Train Segmentation:** A segmentation model (e.g., `CoTrSeg`, `nnU-Net`, or `DynUNet`) is trained on the real data.
4.  **Run Inference:** The trained segmentation model is used to create segmentation masks for the newly generated synthetic data (results in `visualisation1DigitalImageAug`).
5.  **Train Classifier:** A RANO classifier is trained on a combined dataset of both real and synthetic images (with their corresponding masks), often using SMOTE and radiomics features.

---

## Key Features

* **Dual Generators:**
    * `G_B`: Baseline generator (creates a baseline image from a latent vector `z`).
    * `G_F`: Follow-up generator (creates a follow-up image from `z` + `baseline`). It is conditioned on the baseline image features at every scale to ensure consistency.

* **Dual Discriminators:**
    * `D_B` and `D_F` are independent discriminators for baseline and follow-up images, trained with **WGAN-GP** for stability.

* **Hybrid Temporal Loss:**
    * The follow-up generator uses a combined loss (`Adversarial + λ₁·L1 + λ₂·SSIM`) to enforce anatomical consistency between the baseline and follow-up images.

* **3D Volume Support:**
    * The entire pipeline is designed to work with 3D MRI volumes, specifically `128x128x128`.

* **Full Segmentation & Classification Pipeline:**
    * Includes scripts for training transformer-based (CoTrSeg) or U-Net-style segmentation models.
    * Includes a final classification script using an XGBoost classifier with radiomics features and SMOTE for class imbalance.

---

## Repository Structure

```bash
pggan/
├── config.py                 # Main configuration file for the GAN
├── dataset.py                # Dataloader for GAN training
├── networks.py               # PGGAN generator and discriminator models
├── train.py                  # Main training script for the GAN
├── generate.py               # Script to generate synthetic images from a checkpoint
├── utils.py                  # Utility functions (losses, etc.)
│
├── saved256/                 # Checkpoints and image samples from PGGAN training
│   ├── samples/
│   └── checkpoint_XXXk.pth
│
├── generated1070/            # Output directory for generated synthetic .npz files
│   ├── 0/
│   ├── 1/
│   └── ...
│
├── CACHED128/                # Preprocessed real dataset (10-channel .npz)
│   ├── 0/
│   ├── 1/
│   └── ...
│
models_cotrseg/               # Saved weights from trained segmentation models
visualisation1DigitalImageAug/          # Segmentation masks inferred on synthetic data
│
├── train_cotrseg.py          # Script to train the segmentation model
├── dataset_seg_cotrseg.py    # Dataloader for segmentation
├── model_cotrseg.py          # Segmentation model architecture
├── train_classifier_radiomics.py # Script to train the final RANO classifier
│
├── requirements.txt
└── README.md

## 📂 Repository & Dataset Layout
# 🧾 10-Channel .npz Mapping
Each .npz contains a NumPy array 'vol' of shape (10, 128, 128, 128).

| Index | Channel | Description |
|:---|:---|:---|
| 0 | Baseline T1 | |
| 1 | Baseline T1ce | Used by GAN & segmentation |
| 2 | Baseline T2 | |
| 3 | Baseline FLAIR | |
| 4 | Follow-Up T1 | |
| 5 | Follow-Up T1ce | Used by GAN & segmentation |
| 6 | Follow-Up T2 | |
| 7 | Follow-Up FLAIR | |
| 8 | Baseline Mask | 0: bg, 1: edema, 2: tumor |
| 9 | Follow-Up Mask | 0: bg, 1: edema, 2: tumor |

---

## 🚀 Quick Start

### 1️⃣ Clone the Repository

```bash
git clone <your-repo-url>
cd <your-repo>

python -m venv venv
source venv/bin/activate # (Linux / macOS)
# .\venv\Scripts\activate # (Windows)

pip install -r requirements.txt

python pggan/train.py --config pggan/config.py \
--data_root pggan/CACHED128 \
--result_dir pggan/saved256 \
--image_size 128 \
--batch_size 4 \
--epochs 200000

Common Requirements
•	python >= 3.8
•	torch >= 1.10
•	torchvision
•	numpy
•	scipy
•	scikit-image
•	tqdm
•	pillow
•	pyradiomics
•	scikit-learn
•	imbalanced-learn
•	nibabel
________________________________________
🧪 Pipeline Commands
🔹 Train the Two-Generator PGGAN
Bash
python pggan/train.py --config pggan/config.py \
--data_root pggan/CACHED128 \
--result_dir pggan/saved256 \
--image_size 128 \
--batch_size 4 \
--epochs 200000
🔹 Generate Synthetic Dataset
Bash
python pggan/generate.py \
--checkpoint pggan/saved256/checkpoint_XXXk.pth \
--out_dir pggan/generated1070 \
--n_samples 5000 \
--seed 42
Each generated .npz contains (baseline_T1ce, followup_T1ce).
🔹 Train Segmentation Model (CoTrSeg / nnU-Net style)
Bash
python train_cotrseg.py \
--data_root pggan/CACHED128 \
--out_dir models_cotrseg \
--folds 3 \
--epochs 400 \
--batch_size 1 \
--in_channels 4
🔹 Inference on Synthetic Images
Bash
python model_infer.py \
--model models_cotrseg/best_fold.pth \
--input_dir pggan/generated1070 \
--out_dir visualisation128aug \
--sliding_window True
🔹 Train Final Classifier (RANO)
Bash
python train_classifier_radiomics.py \
--data_dirs pggan/CACHED128 pggan/generated1070 \
--masks_dir visualisation128aug \
--out_model ranoclass_xgb.pkl
________________________________________
⚙️ Configuration
Example from pggan/config.py:
Python
DATA_ROOT = 'pggan/CACHED128' # Real data (10-channel .npz)
RESULT_DIR = 'pggan/saved256' # Checkpoints & samples
GENERATED_DIR = 'pggan/generated1070'
IMAGE_SIZE = 128
NUM_CHANNELS = 1 # Each generator uses 1 channel (T1ce)
LATENT_SIZE = 256
BATCH_SIZE = 4
LR = 1e-4
LAMBDA_L1 = 10.0
LAMBDA_SSIM = 1.0
GP_LAMBDA = 10.0 # Gradient penalty (WGAN-GP)
________________________________________
📤 Output Folders Explained
Folder	Description
pggan/saved256/	Model checkpoints and sample PNGs
pggan/generated1070/	Synthetic .npz (2-channel) files
models_cotrseg/	Saved segmentation model weights
visualisation128aug/	Segmentation outputs for synthetic images
________________________________________
💡 Tips & Notes
•	⚠️ Naming mismatch: Folder saved256 contains 128×128×128 data (legacy name).
•	🧹 Dataset sanity: dataset_seg_cotrseg.py filters empty masks to prevent NaN losses.
•	⚙️ 3D SSIM: Ensure your SSIM function supports 3D tensors or compute slice-wise.
•	🔁 Reproducibility: Set torch.manual_seed() and log all config values.
•	⚡ GPU memory: Use batch_size=1 or mixed precision for 3D models.
________________________________________
🤝 Contributing
1.	Fork this repository.
2.	Create your feature branch:
Bash
git checkout -b feature/your-feature
3.	Commit your changes.
4.	Push and open a Pull Request 🚀


