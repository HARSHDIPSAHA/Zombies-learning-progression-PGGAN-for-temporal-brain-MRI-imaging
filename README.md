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

---

## 📂 Repository & Dataset Layout

