# Cross-Attention Enables Context-Aware Multimodal Skin Lesion Diagnosis

Official repository for the paper:

**Cross-Attention Enables Context-Aware Multimodal Skin Lesion Diagnosis**  
**Krishna Mridha**, **Humayera Islam**  
Case Western Reserve University, University of Chicago


## 📋 Abstract

Clinical diagnosis of skin lesions is inherently context-aware: dermatologists interpret lesion morphology together with patient-specific factors such as age, anatomical location, lesion diameter, and skin phenotype. However, many deep learning systems for skin lesion classification rely on images alone and do not explicitly model structured clinical metadata.

In this work, we propose a **context-aware multimodal deep learning framework** that integrates dermoscopic images with structured patient metadata using a **metadata-guided cross-attention mechanism**. Rather than appending metadata only at the final prediction stage, the proposed model allows metadata tokens to guide attention over spatial image features extracted by a Vision Transformer. We compare this approach against metadata-only, image-only, and late-fusion baselines. Results on the **PAD-UFES-20** dataset show that cross-attention yields the strongest overall performance (AUC 0.9818) and improved calibration (ECE 0.0379), highlighting that **how metadata is integrated matters as much as whether it is included**.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Key Contributions](#-key-contributions)
- [Dataset](#-dataset)
- [Model Architectures](#-model-architectures)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Interpretability](#-interpretability)
- [Citation](#-citation)
- [License](#-license)

---

## 🔍 Overview

This repository contains code for:

- Metadata-only baseline (logistic regression)
- Image-only baseline (ResNet18)
- Multimodal late-fusion baseline (feature concatenation)
- Proposed **cross-attention multimodal model** (ViT + metadata-guided attention)
- Training and evaluation pipelines
- Bootstrap comparison analysis
- Ablation studies on metadata features
- Attention visualization and interpretability analysis

---

## 🎯 Key Contributions

1. **Context-aware architecture**: Novel multimodal framework integrating dermoscopic images with structured clinical metadata through metadata-guided cross-attention
2. **Systematic comparison**: Evaluation across four modeling strategies to quantify the benefit of explicit cross-modal interaction
3. **Tokenized metadata representation**: Handles heterogeneous categorical and numerical variables while explicitly modeling missingness
4. **Interpretability analyses**: Feature ablation, cross-attention visualization, metadata perturbation, and case-based examination

---

## 📊 Dataset

We use the **PAD-UFES-20** dataset, a clinically annotated collection of smartphone-acquired dermoscopic images from Brazilian dermatology clinics.

### Dataset Statistics

| Attribute | Value |
|-----------|-------|
| Total lesions | 1,568 |
| Malignant | 1,089 (69%) |
| Benign | 479 (31%) |
| Patient-level split | 80% train / 20% test |

### Clinical Metadata

| Variable | Type | Categories/Range |
|----------|------|-------------------|
| Age | Numerical (continuous) | 18-95 years |
| Sex | Categorical (binary) | Male, Female |
| Fitzpatrick skin type | Categorical | I-VI |
| Anatomical site | Categorical | 5 locations |
| Lesion diameter | Numerical (continuous) | 2-50 mm |

---

## 🏗 Model Architectures

### Figure 1: Multimodal Framework

<p align="center">
  <img src="assets/skin_disease_four_architectures.png" width="950" alt="Four architectural approaches: metadata-only, image-only, late fusion, and cross-attention fusion">
</p>

**Figure 1.** Overview of the multimodal framework for context-aware skin lesion classification. The model integrates dermoscopic images with structured clinical metadata (age, sex, Fitzpatrick skin type, anatomical site, and lesion diameter). Four modeling strategies are evaluated:  
(1) metadata-only logistic regression,  
(2) image-only convolutional neural network (ResNet18),  
(3) multimodal late fusion through feature concatenation, and  
(4) the proposed **cross-attention multimodal architecture**. In the proposed model, metadata tokens attend to visual tokens extracted by a Vision Transformer, enabling metadata-guided interpretation of lesion morphology prior to classification.

---

### 1. Metadata-Only Model (Baseline)
Predicts malignancy risk using structured clinical variables without image information.

- **Input:** Standardized numerical variables (age, lesion diameter) and one-hot encoded categorical variables (sex, Fitzpatrick skin type, anatomical site)
- **Feature Vector:** $x_{meta} = [\tilde{x}_{num}; \psi_{sex}; \psi_{skin}; \psi_{site}] \in \mathbb{R}^{15}$
- **Architecture:** Logistic Regression
- **Output:** $P(y=1|x_{meta}) = \sigma(w^T x_{meta} + b)$

---

### 2. Image-Only Model (Baseline)
Predicts malignancy directly from dermoscopic images using a convolutional neural network.

- **Input:** Dermoscopic images resized to $224 \times 224$ pixels, normalized with ImageNet statistics
- **Architecture:** ResNet18 (excluding final classification layer)
- **Feature Extraction:** $h_{img} = \phi_{ResNet}(resize(I)) \in \mathbb{R}^{512}$
- **Residual Block Update:** $h^{(l+1)} = h^{(l)} + \mathcal{F}(h^{(l)}; W^{(l)})$
- **Output:** $P(y=1|I) = \sigma(w_{img}^T h_{img} + b_{img})$

---

### 3. Late Fusion Multimodal Model
Integrates image features and clinical metadata through feature-level concatenation.

- **Image Features:** $h_{img} \in \mathbb{R}^{512}$ (from ResNet18 encoder)
- **Metadata Features:** $x_{meta} \in \mathbb{R}^{15}$ (standardized numerical + one-hot categorical)
- **Fusion:** $h_{fused} = [h_{img}; x_{meta}] \in \mathbb{R}^{527}$
- **Classifier:** Logistic layer on fused representation
- **Limitation:** Interactions between clinical variables and spatial image features are modeled only implicitly through the final classifier

---

### 4. Proposed Cross-Attention Multimodal Model
Integrates visual and metadata representations through metadata-guided cross-attention.

#### Image Encoder
- Pretrained Vision Transformer (ViT-B/16)
- Retains full sequence of transformer tokens (class + patch tokens) to preserve spatial information
- Output: $H_{img} \in \mathbb{R}^{T_{img} \times d}$

#### Metadata Encoder
- Categorical variables (sex, Fitzpatrick type, anatomical site): dedicated embedding layers → projected to latent space $d$
- Numerical variables (age, lesion diameter): normalized + binary missingness indicators → projected to metadata token
- Output: $H_{meta} \in \mathbb{R}^{T_{meta} \times d}$

#### Cross-Attention Fusion
Metadata tokens act as queries; visual tokens serve as keys and values:

$$
H'_{meta} = \text{softmax}\left(\frac{(H_{meta}W_Q)(H_{img}W_K)^T}{\sqrt{d_k}}\right)(H_{img}W_V)
$$

- Enables each metadata token to selectively attend to spatial regions of the lesion representation
- Patient-specific clinical information dynamically guides which visual patterns are emphasized

#### Prediction Head
1. Residual update + layer normalization: $H'_{meta} \leftarrow \text{LayerNorm}(H_{meta} + H'_{meta})$
2. Feed-forward refinement: $H''_{meta} \leftarrow \text{FFN}(H'_{meta})$
3. Mean pooling: $h_{meta} = \text{MeanPool}(H''_{meta})$
4. Concatenate with CLS token: $h_{fused} = [h_{cls}; h_{meta}]$
5. Final prediction: $\hat{y} = \sigma(\text{MLP}(h_{fused}))$

---

### Algorithm 1: Forward Pass of Metadata-Guided Cross-Attention Model


Require: Image I, numerical metadata x_num, missingness mask x_miss, categorical metadata x_cat
1:  Extract visual tokens using Vision Transformer
2:  H_img ← ϕ_ViT(I)
3:  Construct metadata tokens
4:  H_meta ← MetadataTokenizer(x_num, x_miss, x_cat)
5:  Compute cross-attention
6:  H_attn ← MHA(Q=H_meta, K=H_img, V=H_img)
7:  Residual update
8:  H'_meta ← LayerNorm(H_meta + H_attn)
9:  Feed-forward refinement
10: H''_meta ← FFN(H'_meta)
11: Aggregate metadata tokens
12: h_meta ← MeanPool(H''_meta)
13: Concatenate with CLS token
14: h_fused ← [h_cls; h_meta]
15: Predict malignancy probability
16: ŷ ← σ(MLP(h_fused))

