# DGNet-Endoscopic-Segmentation

This repository contains a PyTorch implementation of the **Dual-Guided Network (DGNet)** for endoscopic image segmentation with region and boundary cues, as proposed in the research paper:

> **Dual-guided network for endoscopic image segmentation with region and boundary cues**
> [Dongzhi He, Yunyu Li, Liule Chen, Xingmei Xiao, Yongle Xue, Zhiqiang Wang, Yunqi Li]
> [Paper Link](https://www.sciencedirect.com/science/article/pii/S1746809424001174)

---

## Paper Summary:
Endoscopic images are inherently more challenging than standard RGB images due to weaker contrast and unclear lesion boundaries, which often leads to inaccurate segmentation results, especially around boundary regions. The DGNet framework is proposed to address the challenges of low contrast and blurred lesion boundaries in endoscopic image segmentation which is an essential step for early gastrointestinal tumor diagnosis.

The Dual-Guided Network (DGNet) architecture is comprised of two branches:
* *Bilateral Attention Branch* which is composed of a mask decoder named Progressive Partial Decoder (PPD) and a module named Full-context Bilateral Relation (FBR). The primary objective of this branch is to focus attention on the ambiguous boundaries of lesion regions by augmenting the correlation between foreground and background cues in the images.
* *Boundary Aggregation Branch* which is composed of a boundary decoder named Boundary-Aware Extraction (BAE) and a module named Boundary-guided Feature Aggregation (BFA). This branch utilizes additional boundary semantic cues to generate features that accentuate the structural aspects of lesion regions.

