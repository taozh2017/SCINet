# Spectral-attentive Contextual Interaction Network for Surgical Instrument Segmentation

## 1. Preface
- This repository provides the official implementation of the paper "Spectral-attentive Contextual Interaction Network for Surgical Instrument Segmentation".
- The code is fully reproducible and supports training/evaluation on four public surgical instrument segmentation datasets (RoboTool, Kvasir-Instrument, Endovis2017, Endovis2018).

## 2. Overview

### 2.1 Introduction
Surgical instrument segmentation plays an important role in aiding visual perception and precise operation of robotic surgical systems. However, the complex background interference, diverse instrument morphologies, and low contrast between instruments and background in surgical scenes make current segmentation models still face significant challenges in accuracy and robustness. Despite significant advances in deep learning based approaches, existing models still fall short in capturing the fine edges and global contextual relationships of instruments. To address these issues, we propose a Spectral-attentive Contextual Interaction Network (SCI-Net) for surgical instrument segmentation. Specifically, we present a Global Context Aggregation Module (GCAM) to integrate high-level features, which is used to produce a global map for the coarse localization of the segmented target. Then, a Spectral-enhanced Feature Module (SFM) is proposed to enhance the expression of features in the form of frequency-domain attention by transforming features from the spatial domain to the frequency domain. In addition, we design the Scale-aware Dilation Module (SDM) in the decoder to further adaptively integrate the augmented features through multi-scale dilation convolution combined with a dynamic fusion mechanism, which improves the segmentation performance on the fine boundaries of instruments.We have extensively validated SCI-Net on multiple publicly available surgical instrument segmentation datasets, and the experimental results show that SCI-Net significantly outperforms other state-of-the-art segmentation methods. We also construct a benchmark for surgical instrument segmentation.

### 2.2 Dataset Overview

<div align="center">
  <img src="imgs/datasets.png" alt="datasets">
  <p><b>Fig. 1</b>: Some sampling examples from four datasets, i.e., RoboTool, Kvasir-Instrument, Endovis2017, and Endovis2018. These datasets highlight several challenging factors for surgical instrument segmentation, including incomplete target display, blurriness, poor angles, low image quality, and high instrument similarity.</p>
</div>

<div align="center">
  <img src="imgs/results_ev1718.png" alt="results_ev1718">
  <p><b>Fig. 2</b>: Qualitative comparisons of SCI-Net with other methods on the EndoVis 2017 and EndoVis 2018 datasets.</p>
</div>

<div align="center">
  <img src="imgs/results_kvrt.png" alt="results_kvrt" >
  <p><b>Fig. 3</b>: Qualitative comparisons of SCI-Net with other SOTA methods on the Kvasir-Instrument and RoboTool datasets (Image, GT, CTNet, EMCADNet, SCI-Net).</p>
</div>

## 3. Dependencies
### 3.1 Required Packages
The code is developed based on Python 3.7+ and PyTorch 1.9.0+. The following packages are required:
```
torch >= 1.9.0
torchvision
numpy
opencv-python (cv2)
tqdm
PyYAML
matplotlib
h5py
torchcam
scikit-learn  # for evaluation metrics
albumentations  # for data augmentation
pillow
scipy
ml_collections
```

### 3.2 Installation
Install the dependencies via pip:
```bash
# Install PyTorch (match your CUDA version, e.g., CUDA 11.1)
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# Install other dependencies
pip install numpy opencv-python tqdm PyYAML matplotlib h5py torchcam scikit-learn albumentations pillow scipy
```

## 4. Dataset Preparation
### 4.1 Supported Datasets
- **EndoVis 2017**: Surgical instrument segmentation challenge dataset (8 instrument classes)
- **EndoVis 2018**: Robotic instrument segmentation dataset (8 instrument classes)
<!-- - **Kvasir-Instrument**: Gastrointestinal endoscopy instrument dataset (1 instrument class)
- **RoboTool**: Robotic surgical tool segmentation dataset (1 instrument classes) -->

### 4.2 Dataset Download Links
- EndoVis 2017/2018: [MICCAI EndoVis Challenge](https://endovissub2017-roboticinstrumentsegmentation.grand-challenge.org/)
<!-- - Kvasir-Instrument: [Kvasir Dataset](https://datasets.simula.no/kvasir-instrument/)
- RoboTool: [RoboTool Dataset](https://github.com/MehmetAygun/robotic-tool-segmentation) -->

### 4.3 Dataset Structure
Organize all datasets in the following unified structure (critical for reproducibility):
```
data/
├── endovis18/
│   ├── train/
│   │   ├── images/          # Training images (JPG/PNG)
│   │   └── annotations/     # Training masks (grayscale PNG, pixel value = class ID)
│   ├── val/
│   │   ├── images/          
│   │   └── annotations/     
│   └── test/
│       ├── images/          
│       └── annotations/     # Optional (for test set evaluation)
├── endovis17/       # Four-fold cross-processing
│   ├── fold0/                   # Cross-validation fold 0 (training/validation split)
│   │   ├── annotations/         # Multi-class segmentation masks (grayscale PNG, pixel = class ID)
│   │   │   └── *.png
│   │   ├── binary_annotations/  # Binary segmentation masks (instrument = foreground, background = 0)
│   │   │   └── *.png
│   │   └── images/              # Raw surgical RGB images
│   │       └── *.png/*.jpg
│   ├── fold1/                   # Cross-validation fold 1 (same structure as fold0)
│   │   ├── annotations/
│   │   ├── binary_annotations/
│   │   └── images/
│   ├── fold2/                   # Cross-validation fold 2 (same structure as fold0)
│   │   ├── annotations/
│   │   ├── binary_annotations/
│   │   └── images/
│   ├── fold3/                   # Cross-validation fold 3 (same structure as fold0)
│   │   ├── annotations/
│   │   ├── binary_annotations/
│   │   └── images/
│   └── test/                    # Independent test set
│       ├── annotations/         # Test set ground truth masks
│       │   └── *.png
│       └── images/              # Raw test images
│           └── *.png/*.jpg
```

### 4.4 Data Preprocessing
- All images are resized to 512×512 (configurable) and converted to RGB format
- Masks are loaded as grayscale images, with pixel values corresponding to class IDs (0 = background, 1~N = instrument classes)
- Training data uses augmentation (random flip, rotation, scaling, brightness/contrast adjustment) via `albumentations`
- Validation/test data only uses resizing and normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Masks are one-hot encoded during training for multi-class segmentation


## 5. Training
### 5.1 Training Configuration
The training process uses distributed data parallel (DDP) for multi-GPU training. Key hyperparameters are configurable via command line arguments.

You need to replace the **dataset storage path** with your own in advance in `dataloader/dataset.py` (where the dataset loading is defined); since `pvt_v2` is adopted as the backbone network in `models/nets/SCINet.py`, you also need to modify the **pre-trained weight path** to match your local file location.

### 5.2 Command Line Arguments
## Command Line Arguments
| Argument         | Default Value                                  | Description |
|------------------|------------------------------------------------|-------------|
| `--tag`          | scinet                                      | Experiment tag for distinguishing different training tasks |
| `--data_dir`     | endovis18                                      | Dataset directory or name (supports `endovis17` / `endovis18`) |
| `--save_root_path` | /opt/data/private/save/lightmodel             | Root path to save model checkpoints and training logs |
| `--local_rank`   | -1                                             | Local rank for distributed training (-1 for single-GPU training) |
| `--num_classes`  | 8                                              | Number of segmentation classes |
| `--epoch_max`    | 200                                            | Maximum training epochs |
| `--image_size`   | 512                                            | Input image resolution (512×512) |

### 6.3 Training Command Examples
#### Single GPU Training
```bash
# EndoVis17 训练命令
python train.py --tag scinet_endovis17 --data_dir endovis17 --num_classes 8 --epoch_max 200 --image_size 512

# EndoVis18 训练命令
python train.py --tag scinet_endovis18 --data_dir endovis18 --num_classes 8 --epoch_max 200 --image_size 512
```


## 7. Inference

### 7.1 Command Line Arguments
| Argument         | Default Value                                  | Description |
|------------------|------------------------------------------------|-------------|
| `--data_path`    | /opt/data/private/MGX/data                     | Root directory of the test dataset |
| `--dataset`      | /endovis18                                     | Test dataset name (`endovis17` / `endovis18`) |
| `--num_classes`  | 8                                              | Number of segmentation classes (consistent with training) |
| `--image_size`   | 512                                            | Input image resolution (consistent with training) |
| `--model`        | /opt/data/private/save/lightmodel/mynetSCTNet.pth | Path to the trained model checkpoint |
| `--save_preds`   | /opt/data/private/save/lightmodel/preds        | Directory to save prediction masks and visualization results |
| `--device`       | cuda                                           | Inference device (`cuda` for GPU, `cpu` for CPU) |

### 7.2 Run Inference
Execute the following command in the terminal to start model inference and evaluation:
```bash

python prediction.py --dataset /endovis17 --num_classes 8 --model /path/to/your/best_model.pth --save_preds /path/to/save/results
```


