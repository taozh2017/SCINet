# Spectral-attentive Contextual Interaction Network for Surgical Instrument Segmentation

## 1. Preface
- This repository provides code for "Spectral-attentive Contextual Interaction Network for Surgical Instrument Segmentation"

## 2.Overview

### 2.1 Introduction
 Surgical instrument segmentation plays an important role in aiding visual perception and precise operation of robotic surgical systems. However, the complex background interference, diverse instrument morphologies, and low contrast between instruments and background in surgical scenes make current segmentation models still face significant challenges in accuracy and robustness. Despite significant advances in deep learning based approaches, existing models still fall short in capturing the fine edges and global contextual relationships of instruments. To address these issues, we propose a Spectral-attentive Contextual Interaction Network (SCI-Net) for surgical instrument segmentation. Specifically, we present a Global Context Aggregation Module (GCAM) to integrate high-level features, which is used to produce a global map for the coarse localization of the segmented target. Then, a Spectral-enhanced Feature Module (SFM) is proposed to enhance the expression of features in the form of frequency-domain attention by transforming features from the spatial domain to the frequency domain. In addition, we design the Scale-aware Dilation Module (SADM) in the decoder to further adaptively integrate the augmented features through multi-scale dilation convolution combined with a dynamic fusion mechanism, which improves the segmentation performance on the fine boundaries of instruments.We have extensively validated SCI-Net on multiple publicly available surgical instrument segmentation datasets, and the experimental results show that SCI-Net significantly outperforms other state-of-the-art segmentation methods. We also construct a benchmark for surgical instrument segmentation.

### 2.2 Dataset Overview

<div align="center">
  <img src="imgs\datasets.jpg" alt="datasets">
  <p><b>Fig. 1</b>: Some sampling examples from four datasets, i.e., RoboTool, Kvasir-Instrument, Endovis2017, and Endovis2018. These datasets highlight several challenging factors for surgical instrument segmentation, including incomplete target display, blurriness, poor angles, low image quality, and high instrument similarity.</p>
</div>

<div align="center">
  <img src="imgs\results_ev1718.jpg" alt="results_ev1718">
  <p><b>Fig. 1</b>: Qualitative comparisons of SCI-Net with other methods on the EndoVis 2017 and EndoVis 2018 datasets.</p>
</div>

<div align="center">
  <img src="imgs\results_kvrt.jpg" alt="results_kvrt" >
  <p><b>Fig. 1</b>: Qualitative comparisons of SCI-Net with other SOTA methods on the Kvasir-Instrument and RoboTool datasets.Image GT CTNetEMCADNet SCI-Net.</p>
</div>

