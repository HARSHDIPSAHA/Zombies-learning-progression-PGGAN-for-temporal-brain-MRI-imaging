# 🧠 Longitudinal Brain MRI Augmentation & Segmentation Pipeline

> **Two-Generator PGGAN → Synthetic Longitudinal MRI (Baseline + Follow-Up) → Segmentation → RANO Classification**

---

## 📘 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Repository & Dataset Layout](#repository--dataset-layout)
- [10-Channel `.npz` Mapping](#10-channel-npz-mapping)
- [Quick Start](#quick-start)
- [Pipeline Commands](#pipeline-commands)
- [Configuration](#configuration)
- [Output Folders Explained](#output-folders-explained)
- [Tips & Notes](#tips--notes)
- [Contributing](#contributing)
- [License & Citation](#license--citation)
- [Contact](#contact)

---

## 🧩 Overview

This project implements an end-to-end pipeline for augmenting **longitudinal brain MRI data** using a **Two-Generator Progressive GAN (PGGAN)**.  
The system generates anatomically consistent **baseline and follow-up** image pairs that augment real data to improve **3D segmentation** and **RANO classification** performance.

### 🧠 End-to-End Flow

1. **Preprocessed real data (`CACHED128`)** → train Two-Generator PGGAN  
2. **Generate synthetic `.npz`** (baseline + follow-up) → stored in `generated1070`  
3. **Train segmentation model** (`CoTrSeg`, `nnU-Net`, or `DynUNet`)  
4. **Run inference on synthetic data** → create segmentation masks (`visualisation128aug`)  
5. **Train RANO classifier** on combined real + synthetic dataset (with SMOTE + radiomics)

---

## ⚙️ Key Features

- 🧬 **Two Generators**  
  - `G_B`: Baseline generator (z → baseline image)  
  - `G_F`: Follow-up generator (z + baseline → follow-up), conditioned on baseline at every scale  

- 🧩 **Two Discriminators:** `D_B` and `D_F` trained with **WGAN-GP**

- 🎯 **Hybrid Loss for Follow-Up Generator:**  
  `Adversarial + λ₁·L1 + λ₂·SSIM` → enforces anatomical consistency over time

- 🧠 **Supports 3D MRI volumes** (`128×128×128`)

- 🧾 **Segmentation + Classification Pipeline:**  
  Uses transformer-based CoTrSeg or nnU-Net style networks, then XGBoost classifier with radiomics & SMOTE.
  ## 📂 Repository Structure

```ruby
pggan/
├── config.py
├── dataset.py
├── networks.py
├── train.py
├── generate.py
├── utils.py
│
├── saved256/                  # PGGAN checkpoints & sample images
│   ├── samples/
│   └── checkpoint_XXXk.pth
│
├── generated1070/             # GAN-generated 2-channel .npz data
│   ├── 0/
│   ├── 1/
│   └── ...
│
├── CACHED128/                 # Real dataset (10-channel .npz)
│   ├── 0/
│   ├── 1/
│   └── ...
│
models_cotrseg/                # Trained segmentation weights
visualisation128aug/           # Synthetic inference results
train_cotrseg.py
dataset_seg_cotrseg.py
model_cotrseg.py
train_classifier_radiomics.py
requirements.txt
README.md

---

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


