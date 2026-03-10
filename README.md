
# Cross-Attention Enables Context-Aware Multimodal Skin Lesion Diagnosis

Official repository for the paper:

**Cross-Attention Enables Context-Aware Multimodal Skin Lesion Diagnosis**  
**Krishna Mridha**, **Humayera Islam**  
Case Western Reserve University, University of Chicago

---

## Abstract

Clinical diagnosis of skin lesions is inherently context-aware: dermatologists interpret lesion morphology together with patient-specific factors such as age, anatomical location, lesion diameter, and skin phenotype. However, many deep learning systems for skin lesion classification rely on images alone and do not explicitly model structured clinical metadata.

In this work, we propose a **context-aware multimodal deep learning framework** that integrates dermoscopic images with structured patient metadata using a **metadata-guided cross-attention mechanism**. Rather than appending metadata only at the final prediction stage, the proposed model allows metadata tokens to guide attention over spatial image features extracted by a Vision Transformer. We compare this approach against metadata-only, image-only, and late-fusion baselines. Results on the **PAD-UFES-20** dataset show that cross-attention yields the strongest overall performance and improved calibration, highlighting that **how metadata is integrated matters as much as whether it is included**.

---

## Overview

This repository contains code for:

- metadata-only baseline
- image-only baseline
- multimodal late-fusion baseline
- proposed **cross-attention multimodal model**
- training and evaluation scripts
- bootstrap comparison analysis
- ablation studies on metadata features
- attention visualization and interpretability analysis

---

## Figure 1: Multimodal Framework


<p align="center">
  <img src="assets/skin_disease_four_architectures.png" width="950">
</p>


**Figure 1.** Overview of the multimodal framework for context-aware skin lesion classification. The model integrates dermoscopic images with structured clinical metadata (age, sex, Fitzpatrick skin type, anatomical site, and lesion diameter). Four modeling strategies are evaluated:  
(1) metadata-only logistic regression,  
(2) image-only convolutional neural network (ResNet18),  
(3) multimodal late fusion through feature concatenation, and  
(4) the proposed **cross-attention multimodal architecture**.  
In the proposed model, metadata tokens attend to visual tokens extracted by a Vision Transformer, enabling metadata-guided interpretation of lesion morphology prior to classification.

---

## Proposed Model Architecture

The proposed model performs **structured multimodal fusion** by allowing clinical metadata to interact directly with image tokens.

### Input modalities

#### Dermoscopic image
A dermoscopic image is resized to `224 × 224` and processed through a pretrained **Vision Transformer (ViT-B/16)** to obtain spatial image tokens.

#### Structured metadata
Clinical metadata include:

- age
- sex
- Fitzpatrick skin type
- anatomical site
- lesion diameter

Categorical features are embedded into learned metadata tokens, while numerical features are normalized and projected into the same latent space.

---

### Cross-attention mechanism

Let:

- `H_img ∈ R^(T_img × d)` be the image token sequence from ViT
- `H_meta ∈ R^(T_meta × d)` be the metadata token sequence

Cross-attention is computed as:

```math
H'_{meta} = \text{softmax}\left(\frac{(H_{meta}W_Q)(H_{img}W_K)^T}{\sqrt{d_k}}\right)(H_{img}W_V)
