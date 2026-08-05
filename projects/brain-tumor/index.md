---
layout: page
title: Brain Tumor Pipeline
---

<img class="project-cover" src="{{ '/projects/brain-tumor/cover.png' | relative_url }}" alt="Segmentation and classification results" />

**Stack:** Python, PyTorch, MONAI  
**Repo:** [BrainTumorPipeline](https://github.com/Adnane-Ahroum/BrainTumorPipeline)

End-to-end deep learning pipeline on the **BraTS** dataset combining 3D segmentation and tumor-type classification.

## Pipeline architecture

<img class="project-cover" src="{{ '/projects/brain-tumor/architecture.png' | relative_url }}" alt="Two-stage pipeline architecture" />

**Stage 1 — Segmentation (3D U-Net)**  
Multi-modal MRI inputs (T1, T1ce, T2, FLAIR) → WT / TC / ET masks.

**Stage 2 — Classification (DenseNet-121)**  
Segmented region → Glioma, Meningioma, or Pituitary.

## Capstone poster

<img class="project-cover" src="{{ '/projects/brain-tumor/CapstonePoster.png' | relative_url }}" alt="Capstone poster" />

[← Back to Home]({{ '/' | relative_url }})
