# multimodal-skin-lesion-cross-attention


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
