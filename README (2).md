# 🧠 Brain Tumor Segmentation Using U-Net with PSO Optimization

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.0-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A deep learning pipeline for automatic brain tumor segmentation in MRI images using U-Net architecture, enhanced with classical image processing techniques and Particle Swarm Optimization (PSO) for hyperparameter tuning.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Model Architecture](#model-architecture)
- [Optimization](#optimization)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)

---

## 🔍 Overview

This project builds a medical image segmentation system that automatically detects and delineates brain tumors in MRI scans. The system combines:

- **Deep Learning** → U-Net for pixel-wise segmentation
- **Classical Image Processing** → Bilateral filtering, CLAHE, morphological operations
- **Metaheuristic Optimization** → PSO for automatic hyperparameter tuning

---

## 📂 Dataset

**LGG MRI Segmentation** — available on [Kaggle](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)

| Property | Value |
|----------|-------|
| Total Images | 7,858 |
| Used Samples | 4,500 |
| Image Size | 256 × 256 |
| Train Split | 3,600 (80%) |
| Validation Split | 900 (20%) |
| Mask Values | Binary (0 = background, 255 = tumor) |

---

## ⚙️ Pipeline

```
Raw MRI Images
      │
      ▼
1. Image Acquisition      → Load & pair images with masks
      │
      ▼
2. Image Enhancement      → Bilateral Filter + CLAHE + Unsharp Masking
      │
      ▼
3. Data Augmentation      → 18 geometric & photometric techniques
      │
      ▼
4. PSO Optimization       → Tune LR, Batch Size, CLAHE Clip
      │
      ▼
5. U-Net Training         → Combined BCE + Dice Loss
      │
      ▼
6. Post-Processing        → Morphological Operations
      │
      ▼
Final Segmentation Masks
```

### 🎨 Image Enhancement

| Technique | Purpose |
|-----------|---------|
| Bilateral Filtering | Noise reduction while preserving edges |
| CLAHE | Local contrast enhancement |
| Unsharp Masking | Sharpening fine details |

### 🔧 Data Augmentation

**Strong (Training):** HorizontalFlip, VerticalFlip, Rotation, Perspective, Affine, ElasticTransform, GridDistortion, CoarseDropout, RandomBrightnessContrast, RandomGamma, CLAHE, GaussNoise, ISONoise, MultiplicativeNoise, GaussianBlur, MotionBlur, Sharpen, RandomScale

**Weak (Validation):** HorizontalFlip, VerticalFlip, Rotation, RandomBrightnessContrast

---

## 🏗️ Model Architecture

**U-Net** with 5 encoder levels and symmetric decoder with skip connections.

```
Input: (256, 256, 3)
│
├── Encoder
│   ├── Block 1: Conv2D(64)  → MaxPool → (128, 128, 64)
│   ├── Block 2: Conv2D(128) → MaxPool → (64, 64, 128)
│   ├── Block 3: Conv2D(256) → MaxPool → (32, 32, 256)
│   └── Block 4: Conv2D(512) → MaxPool → (16, 16, 512)
│
├── Bottleneck: Conv2D(1024) → (16, 16, 1024)
│
└── Decoder (with Skip Connections)
    ├── Block 6: UpSample + Concat(c4) → Conv2D(512)
    ├── Block 7: UpSample + Concat(c3) → Conv2D(256)
    ├── Block 8: UpSample + Concat(c2) → Conv2D(128)
    └── Block 9: UpSample + Concat(c1) → Conv2D(64)
│
Output: Conv2D(1, activation='sigmoid') → (256, 256, 1)
```

| Property | Value |
|----------|-------|
| Total Parameters | 31,402,497 |
| Trainable Parameters | 31,390,721 |
| Optimizer | Adam |
| Loss | 0.5 × BCE + 0.5 × Dice |
| Metrics | Dice Coefficient, IoU, Binary Accuracy |

---

## ⚡ Optimization

**Particle Swarm Optimization (PSO)** automatically searches for optimal hyperparameters.

| Parameter | Search Range | Optimal Value |
|-----------|-------------|---------------|
| Learning Rate | 1e-5 → 1e-3 | 0.000098 |
| Batch Size | 4 → 16 | 10 |
| CLAHE Clip Limit | 1.0 → 5.0 | 5.00 |

PSO Settings: **8 particles**, **3 iterations**

---

## 📊 Results

### Training Configuration
| Setting | Value |
|---------|-------|
| Epochs | 50 |
| Best Epoch | 44 |
| Final Train Loss | 0.2173 |
| Final Val Loss | 0.1825 |
| Best Val Dice | 0.6641 |

### Segmentation Performance

| Metric | Without Morphology | With Morphology |
|--------|--------------------|-----------------|
| Mean Dice | 0.8120 ± 0.3325 | **0.8142 ± 0.3314** |
| Mean IoU | 0.7789 ± 0.3455 | **0.7815 ± 0.3441** |
| Median Dice | 1.0000 | 1.0000 |
| Median IoU | 1.0000 | 1.0000 |

### Case Difficulty Analysis
| Category | Count | Percentage |
|----------|-------|------------|
| Easy Cases (Dice ≥ 0.70) | 725 | 80.6% |
| Difficult Cases (Dice < 0.70) | 175 | 19.4% |

### Overall Assessment: ✅ **VERY GOOD — High Quality Segmentation**

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/Brain-Tumor-Segmentation-U-Net-PSO.git
cd Brain-Tumor-Segmentation-U-Net-PSO

# Install dependencies
pip install tensorflow numpy opencv-python matplotlib scikit-learn \
            albumentations scikit-image scipy kagglehub tqdm
```

---

## 🚀 Usage

```bash
# Run the notebook
jupyter notebook brain_tumor_segmentation.ipynb
```

Or run the script directly:

```python
# The pipeline runs end-to-end:
# 1. Downloads dataset automatically via kagglehub
# 2. Preprocesses and augments images
# 3. Runs PSO to find optimal hyperparameters
# 4. Trains U-Net model for 50 epochs
# 5. Evaluates and visualizes results
```

---

## 📁 Project Structure

```
Brain-Tumor-Segmentation-U-Net-PSO/
│
├── brain_tumor_segmentation.ipynb   # Main notebook
├── README.md                         # This file
│
├── outputs/
│   ├── best_model.h5                 # Best checkpoint (by val Dice)
│   ├── final_model.h5                # Final model
│   ├── final_model.keras             # Final model (Keras format)
│   ├── training_log.csv              # Epoch-by-epoch metrics
│   ├── training_curves.png           # Loss & metric plots
│   ├── segmentation_results.png      # Sample predictions
│   ├── overlay_visualization.png     # GT vs Prediction overlays
│   ├── metrics_comparison.png        # Boxplot comparisons
│   └── preprocessing_effect.png      # Enhancement visualization
```

---

## 📚 Key Concepts

| Term | Definition |
|------|-----------|
| **Dice Score** | Overlap metric between prediction and ground truth (1 = perfect) |
| **IoU** | Intersection over Union — area overlap ratio |
| **U-Net** | Encoder-decoder CNN with skip connections for segmentation |
| **CLAHE** | Adaptive histogram equalization for local contrast |
| **PSO** | Bio-inspired optimization mimicking bird flocking behavior |
| **Morphological Operations** | Post-processing to clean and refine predicted masks |

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgements

- Dataset: [LGG MRI Segmentation — Mateusz Buda, Kaggle](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)
- U-Net Paper: Ronneberger et al., 2015
- TensorFlow / Keras
- Albumentations Library
