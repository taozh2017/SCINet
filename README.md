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
- **Kvasir-Instrument**: Gastrointestinal endoscopy instrument dataset (1 instrument class)
- **RoboTool**: Robotic surgical tool segmentation dataset (1 instrument classes)

### 4.2 Dataset Download Links
- EndoVis 2017/2018: [MICCAI EndoVis Challenge](https://endovissub2017-roboticinstrumentsegmentation.grand-challenge.org/)
- Kvasir-Instrument: [Kvasir Dataset](https://datasets.simula.no/kvasir-instrument/)
- RoboTool: [RoboTool Dataset](https://github.com/MehmetAygun/robotic-tool-segmentation)

### 4.3 Dataset Structure
Organize all datasets in the following unified structure (critical for reproducibility):
```
data/
├── endovis17/
│   ├── train/
│   │   ├── images/          # Training images (JPG/PNG)
│   │   └── annotations/     # Training masks (grayscale PNG, pixel value = class ID)
│   ├── val/
│   │   ├── images/          
│   │   └── annotations/     
│   └── test/
│       ├── images/          
│       └── annotations/     # Optional (for test set evaluation)
├── endovis18/
│   ├── train/
│   ├── val/
│   └── test/
├── kvasir_instrument/
│   ├── train/
│   │   ├── images/ 
│   │   └── masks/ 
│   ├── val/
│   └── test/
└── robotool/
    ├── train/
    ├── val/
    └── test/
```

### 4.4 Data Preprocessing
- All images are resized to 512×512 (configurable) and converted to RGB format
- Masks are loaded as grayscale images, with pixel values corresponding to class IDs (0 = background, 1~N = instrument classes)
- Training data uses augmentation (random flip, rotation, scaling, brightness/contrast adjustment) via `albumentations`
- Validation/test data only uses resizing and normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Masks are one-hot encoded during training for multi-class segmentation

## 5. Model Architecture
SCI-Net is built on a modified Segment Anything Model (SAM) backbone with the following key components:
- **Backbone**: SCINet (custom feature extraction network with frequency-domain enhancement)
- **Global Context Aggregation Module (GCAM)**: Captures long-range contextual relationships for coarse target localization
- **Spectral-enhanced Feature Module (SFM)**: Transforms features to frequency domain to enhance discriminative representation
- **Scale-aware Dilation Module (SDM)**: Multi-scale dilation convolution with dynamic fusion for fine boundary segmentation
- **Loss Function**: Combined loss = Cross-entropy loss (class-weighted) + Dice loss + IoU loss + Boundary-aware loss

## 6. Training
### 6.1 Training Configuration
The training process uses distributed data parallel (DDP) for multi-GPU training. Key hyperparameters are configurable via command line arguments.

### 6.2 Command Line Arguments
| Argument | Default Value | Description |
|----------|---------------|-------------|
| `--name` | 'scinet_exp' | Experiment name (for logging/saving) |
| `--model` | None | Path to pre-trained checkpoint (optional, for fine-tuning) |
| `--tag` | 'scinet' | Additional tag for experiment identification |
| `--dataset` | 'endovis17' | Dataset name (endovis17/endovis18/kvasir_instrument/robotool) |
| `--local_rank` | -1 | Local rank for DDP (auto-assigned in multi-GPU training) |
| `--num_classes` | 8 | Number of classes (match dataset: endovis17=8, endovis18=7, kvasir=1, robotool=4) |
| `--epoch_max` | 200 | Maximum training epochs |
| `--image_size` | 512 | Input image size (height=width) |
| `--batch_size` | 16 | Batch size per GPU (adjust based on GPU memory) |
| `--lr` | 1e-4 | Initial learning rate (AdamW) |
| `--weight_decay` | 1e-5 | Weight decay for optimizer |
| `--save_dir` | './checkpoints' | Directory to save model checkpoints |
| `--log_dir` | './logs' | Directory to save training logs (TensorBoard) |

### 6.3 Training Command Examples
#### Single GPU Training (EndoVis 2017)
```bash
python train.py --dataset endovis17 --num_classes 8 --epoch_max 200 --image_size 512 --batch_size 8 --lr 1e-4
```

#### Multi-GPU Training (DDP, EndoVis 2018)
```bash
python -m torch.distributed.launch --nproc_per_node=4 train.py \
  --dataset endovis18 \
  --num_classes 7 \
  --epoch_max 200 \
  --image_size 512 \
  --batch_size 4 \
  --lr 1e-4 \
  --local_rank 0
```

#### Fine-tuning on Kvasir-Instrument
```bash
python train.py --dataset kvasir_instrument --num_classes 1 --epoch_max 100 --image_size 512 --model ./checkpoints/scinet_exp/best.pth
```

### 6.4 Training Details
- Optimizer: AdamW (β1=0.9, β2=0.999)
- Learning rate scheduler: Cosine annealing with warmup (10 epochs warmup)
- Early stopping: Patience=20 (stop if validation IoU does not improve)
- Checkpoint saving: Saves "last.pth" (latest epoch) and "best.pth" (highest validation mean IoU)
- Logging: Training/validation loss and metrics are logged to TensorBoard every epoch

## 7. Inference
### 7.1 Command Line Arguments
| Argument | Default Value | Description |
|----------|---------------|-------------|
| `--data_path` | './data' | Root path of dataset |
| `--dataset` | 'endovis17' | Dataset name for inference |
| `--num_classes` | 8 | Number of classes (match training) |
| `--image_size` | 512 | Input image size (same as training) |
| `--model` | './checkpoints/best.pth' | Path to trained model checkpoint |
| `--output_dir` | './inference_results' | Directory to save prediction masks/visualizations |
| `--save_vis` | False | Whether to save visualization (image + GT + prediction) |
| `--eval` | True | Whether to calculate evaluation metrics (requires test annotations) |

### 7.2 Inference Command Examples
#### Inference on EndoVis 2017 Test Set
```bash
python prediction.py --dataset endovis17 --num_classes 8 --model ./checkpoints/scinet_exp/best.pth --save_vis True
```

#### Inference on Kvasir-Instrument (without evaluation, only save predictions)
```bash
python prediction.py --dataset kvasir_instrument --num_classes 1 --model ./checkpoints/scinet_kvasir/best.pth --eval False --output_dir ./kvasir_predictions
```

### 7.3 Inference Outputs
- Predicted masks (PNG format, pixel value = class ID) saved in `${output_dir}/masks/`
- Visualization results (if `--save_vis=True`) saved in `${output_dir}/vis/` (overlay of prediction on original image)
- Evaluation report (if `--eval=True`) saved as `${output_dir}/eval_metrics.json` (contains class-wise and mean Dice/IoU)

## 8. Evaluation Metrics
The code computes the following standard segmentation metrics (consistent with the paper):
- **Dice Coefficient (F1-score)**: Per-class and mean Dice
- **Jaccard Index (IoU)**: Per-class and mean IoU
- **Pixel Accuracy (PA)**: Overall pixel classification accuracy
- **Mean Intersection over Union (mIoU)**: Average IoU across all classes (excluding background if specified)

Metrics are calculated using `scikit-learn` and saved in both JSON format (for quantitative analysis) and printed in the terminal.

## 9. Notes for Reproducibility
1. **Environment Consistency**: Use the exact PyTorch/CUDA version (PyTorch 1.9.0 + CUDA 11.1 recommended) to avoid compatibility issues.
2. **GPU Memory**: For 512×512 input, a GPU with ≥12GB memory is required (batch size=8 for single GPU, batch size=4 per GPU for 4-GPU DDP).
3. **Dataset Consistency**: Ensure the dataset is split into train/val/test as per the paper (we provide split files in `./data_splits/`).
4. **Random Seed**: The code sets a fixed random seed (42) for PyTorch/Numpy to ensure result reproducibility.
5. **Pre-trained Weights**: For fair comparison, use the provided pre-trained backbone weights (in `./pretrained/`) instead of training from scratch.
6. **Evaluation Protocol**: Use the same test set and metric calculation method as described in the paper (we provide the official evaluation script in `./eval/`).

## 10. Troubleshooting
- **CUDA Out of Memory**: Reduce batch size (`--batch_size`), use smaller image size, or enable gradient accumulation.
- **Model Loading Error**: Ensure the number of classes matches between training and inference, and the checkpoint path is correct.
- **Data Loading Error**: Verify the dataset structure matches Section 4.3, and all image/mask files are in the correct format (PNG/JPG).
