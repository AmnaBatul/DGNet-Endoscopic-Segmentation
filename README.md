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

## Implementation Details:

### Dataset
The implementation was performed on [Kvasir-SEG dataset](https://www.kaggle.com/datasets/debeshjha1/kvasirseg) - a benchmark dataset for polyp segmentation in endoscopic images.

While the original paper uses image resolution of 512x512 for Kvasir-SEG and batch size of 16, this implementation uses image resolution of 256x256 and batch size of 8. These adjusments were made to accommodate hardware resource constraints (limited GPU memory). As a result, performance metrics such as Dice and IoU scores are sightly lower than those reported in the paper.
Despite this, the architecture remains faithful to the original.

## Main Architecture
The architecture is built around a shared encoder and two parallel, yet complementary branches: *Bilateral Attention Branch* and *Boundary Aggregation Branch* to improve lesion localization and boundary precision.

<img width="927" height="670" alt="image" src="https://github.com/user-attachments/assets/8aae2da6-0b34-4622-bae8-68a483bd68c6" />


* Res2Net50 Backbone:
  * This encoder captures captures multiscale features by splitting the convolutional channels into several smaller groups processed in parallel with different receptive fields.
  * It outputs 5 stages of hierarchical feature maps: {X0, X1, X2, X3, X4}. Low level features {X0, X1} retain spatial details while high level features {X2, X3, X4} capture semantic context.
  * For segmentation, only {X1, X2, X3, X4} are used; X0 is ignored due to redundant noise.
    
