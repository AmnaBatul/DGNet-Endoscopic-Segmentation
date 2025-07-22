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

### Main Architecture
The architecture is built around a shared encoder and two parallel, yet complementary branches: *Bilateral Attention Branch* and *Boundary Aggregation Branch* to improve lesion localization and boundary precision.

<img width="927" height="670" alt="image" src="https://github.com/user-attachments/assets/8aae2da6-0b34-4622-bae8-68a483bd68c6" />


* Res2Net50 Backbone:
  * This encoder captures captures multiscale features by splitting the convolutional channels into several smaller groups processed in parallel with different receptive fields.
  * It outputs 5 stages of hierarchical feature maps: {X0, X1, X2, X3, X4}. *Low level features* {X0, X1} retain spatial details while *high level features* {X2, X3, X4} capture semantic context.
  * For segmentation, only {X1, X2, X3, X4} are used; X0 is ignored due to redundant noise.
    
* Expand Receptive Field (ERF) Block:
  The ERF block is used throughout both branches to enhance contextual understanding.
  * It applies *strip + dilated* convolutions to expand receptive fields.
  * Channels are concatenated, fused using a 1x1 conv, and modulated via an attention mechanism with residual connections.
  * ERF increases sensitivity to lesion edges and small spatial variations.
  <img width="1194" height="452" alt="image" src="https://github.com/user-attachments/assets/2244ad7d-9f9c-4bb0-94da-cc24191808ef" />
  

* Bilateral Attention Branch:
  This branch focuses on localizing ambiguous lesion regions, especially small or low-contrast areas.
  1. Progressive Partial Decoder (PPD):
     * Combines X1-1, X2-1, X3-1, X4-1 features (processed via ERF).
     * Upsamples and processes each level via 3x3 convolutions.
     * Performs matrix multiplication between layers to reduce semantic gaps.
     * Aggregates coarse lesion mask D0, with intermediate predictions {D1, D2, D3} for deep supervision.
  <img width="1005" height="514" alt="image" src="https://github.com/user-attachments/assets/60399761-c223-4eab-adbb-fa064049f5fd" />
  
  
  2. Full-context Bilateral Relation (FBR):
     * Applies Global Attention Block (self-attention along H, W, and C).
     * Splits predictions into foreground and background maps: Df = X' * D', Db = X' * (1 - D')
     * Combines both maps with 3x3 convs to enhance representation.
  <img width="653" height="328" alt="image" src="https://github.com/user-attachments/assets/ad7f669c-a3bb-4bd5-ad88-f2bc95d8bb3d" />


 * Boundary Aggregation Branch:
   This branch is designed to sharpen boundary details by fusing edge-aware and semantic features.
   1. Boundary-Aware Extraction (BAE):
      * Takes low-level X1-2 and high-level X4-2.
      * Applies convolutions on X4-2 to get an attention map Xa_4-2.
      * Enhances X1-2 with boundary weights and predicts M0 using sigmoid.
   <img width="702" height="242" alt="image" src="https://github.com/user-attachments/assets/baf4cd5b-f1b5-494d-a634-a8c3b685ba78" />

   2. Boundary-guided Feature Aggregation (BFA):
      * Incorporates both Boundary Extraction Block (BEB) and Feature Aggregation Block (FAB).
      * BEB: Multiplies features {Xi-2} with M0, passes through spatial attention.
      * FAB: Splits features into 4 groups, applies dilated convs with dilation rates {1, 2, 3, 4}.
      * Combines features via addition and concatenation, enhancing boundary-aware representations.
    <img width="775" height="616" alt="image" src="https://github.com/user-attachments/assets/a5c6b9b8-0021-423b-a9ca-f7c986dc5897" />

        
 * Fusion:
   * Final prediction is obtained by combining outputs from both branches.
   * Deep supervision is applied at each stage (D1–D3, M1–M3) to guide training.
     
 ### Loss Function:
   Total loss is a weighted combination of:
   * Dice Loss — Supervises boundary prediction (M0).
   * BCE + IoU Loss — Supervises coarse lesion mask (D0).
   * Intermediate BCE + IoU — Supervises {D1+M1, D2+M2, D3+M3}.
  
 ### Evaluation Metrics:
   To assess the segmentation performance, the following metrics are calculated:
   * Accuracy (Acc): Proportion of correctly predicted pixels.
   * IoU (Intersection over Union): Measures overlap between prediction and ground truth.
   * Dice Score: Harmonic mean of precision and recall, suitable for imbalanced datasets.
   * Precision & Recall: Focused on positive class detection.
   * G-score: Geometric mean of precision and recall.
   * Kappa Score: Adjusted agreement score accounting for chance.
   * Matthews Correlation Coefficient (MCC): Balanced metric accounting for all confusion matrix elements.

 ## Results
| Metric     | DGNet (Paper) | This Implementation |
|------------|---------------|---------------------|
| Accuracy   | 96.48%        | 95.85%              |
| IoU        | 86.17%        | 77.41%              |
| Dice       | 92.69%        | 87.06%              |
| Recall     | 92.33%        | 86.09%              |
| Precision  | 93.34%        | 88.48%              |
| G-Score    | 92.83%        | 87.17%              |
| Kappa      | 84.77%        | 84.55%              |
| MCC        | 85.82%        | 84.72%              |

**Note**: Paper results are based on 512×512 input images with a batch size of 16.  
This implementation uses 256×256 resolution and batch size 8 due to GPU constraints.
