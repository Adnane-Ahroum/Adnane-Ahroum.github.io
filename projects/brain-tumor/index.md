---
layout: default
title: Brain Tumor Segmentation & Classification
---

# 🧠 Brain Tumor Segmentation & Classification

**Stack:** Python, PyTorch, MONAI, Apache Airflow, 3D Slicer, Hugging Face, Weights & Biases  
**Data:** BraTS (University of Pennsylvania)  
**Repo:** <https://github.com/Adnane-Ahroum/BrainTumorPipeline>

## Problem
Automate tumor **segmentation** and patient **classification** from MRI scans to support clinical workflows.

## Approach
- Preprocessing and augmentation pipeline for multi‑modal MRI volumes.  
- **U‑Net** (segmentation) and **DenseNet121** (classification).  
- Orchestrated with **Apache Airflow** for reproducible stages (ingest → preprocess → train → eval).

## Results (example placeholders)
- Dice score (WT): 0.89 · (TC): 0.84 · (ET): 0.81  
- Classification accuracy: 91% on held‑out set  
_Add your real numbers/screenshots._

## What I learned
Efficient handling of 3D medical data, evaluation pitfalls (class imbalance), and experiment tracking.

## Artifacts
- Model weights, training logs, sample predictions (see repo).  
- Demo notebook and pipeline DAG.

[← Back to Home](/)