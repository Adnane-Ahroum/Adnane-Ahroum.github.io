---
layout: default
title: Brain Tumor Segmentation & Classification
---

# 🧠 Deep Learning-Driven MRI Analysis for Brain Tumor Detection, Segmentation & Classification  

This project implements a **two-stage AI pipeline** for brain tumor analysis using the **BraTS dataset (University of Pennsylvania)**.  
It combines **segmentation (U-Net)** and **classification (DenseNet-121)**, optimized for clinical interpretability and reproducibility.  

---

## 📊 Capstone Poster
![Capstone Poster](CapstonePoster.png)

---

## 🔬 Pipeline Architecture
![Pipeline Architecture](architecture.png)

**Stage 1 — Segmentation (3D U-Net):**
- Inputs: multi-modal MRI scans (T1, T1ce, T2, FLAIR)  
- Outputs: WT (Whole Tumor), TC (Tumor Core), ET (Enhancing Tumor) masks  
- Loss: Dice + Cross-Entropy  

**Stage 2 — Classification (DenseNet-121):**
- Inputs: segmented tumor region  
- Outputs: Tumor Type → Glioma, Meningioma, or Pituitary  
- Conditional check: if segmentation mask = empty, classification ignored  

---

## 📈 Results & Visualizations

### Segmentation Performance
- Dice Loss converged to **0.18** → strong overlap between predicted vs. ground truth masks  
- Visual comparison:

| Learning Rate | Validation Dice | Dice Tumor Core |
|---------------|-----------------|-----------------|
| <img src="./SEGlearningrate.png" width="300"/> | <img src="./SEGmeandice.png" width="300"/> | <img src="./SEGmeandicetumorcore.png" width="300"/> |

| Whole Tumor Dice | Training Loss | Enhancing Tumor Dice |
|------------------|---------------|----------------------|
| <img src="./SEGmeandicewholetumor.png" width="300"/> | <img src="./SEGmeantrainloss.png" width="300"/> | <img src="./VALIDATION EHABNED TUMOR .png" width="300"/> |

---

### Classification Results
- Glioma: **97% accuracy**  
- Meningioma: **78% accuracy**  
- Pituitary: **85% accuracy**  

Visualization (blue = ground truth, red = prediction):  
![Classification MRI](/assets/images/classification-mri.png)

---

## ⚙️ Code Implementation

```python
# Training 3D U-Net for segmentation
import torch
from monai.networks.nets import UNet

model = UNet(
    dimensions=3,
    in_channels=4,
    out_channels=4,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2
).to(device)

# Dice + Cross Entropy Loss
from monai.losses import DiceCELoss
loss_function = DiceCELoss(include_background=True, to_onehot_y=True, softmax=True)
