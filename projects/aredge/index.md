---
layout: page
title: Medical Video Segmentation
---

<img class="project-cover" src="{{ '/projects/aredge/cover.png' | relative_url }}" alt="Medical video segmentation" />

**Stack:** Python, PyTorch, SAM2, OpenCV  
**Repo:** [AREdge-Based-Data-Science-Final-Project](https://github.com/Adnane-Ahroum/AREdge-Based-Data-Science-Final-Project)

Comparative study of **supervised** (EchoNet-Dynamic / DeepLabV3) vs **zero-shot** (SAM2) approaches for cardiac and surgical video segmentation.

## Highlights

- EchoNet-Dynamic on 10k+ echocardiography videos with EF metrics
- SAM2 zero-shot masks on SurgiS4K surgical frames (~87% success at 480p)
- Trade-off analysis: annotation cost vs generalization

## Sample frame

<img class="project-cover" src="{{ '/projects/aredge/frame.png' | relative_url }}" alt="Surgical frame sample" />

[← Back to Home]({{ '/' | relative_url }})
