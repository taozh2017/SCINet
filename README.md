# SAM-newAdapter-decoder-otherMethods

This repository contains an implementation of a modified Segment Anything Model (SAM) for medical image segmentation, incorporating new adapter mechanisms, decoder improvements, and other methodological enhancements.

## Dependencies

The following Python packages are required to run this project:

- torch (>=1.9.0)
- torchvision
- numpy
- opencv-python (cv2)
- tqdm
- PyYAML (yaml)
- matplotlib
- h5py
- torchcam
- scikit-learn (for metrics)
- albumentations (for data transforms, if used)

You can install the dependencies using pip:

```bash
pip install torch torchvision numpy opencv-python tqdm PyYAML matplotlib h5py torchcam scikit-learn albumentations
```

## Training

To train the model, run the `train.py` script with the appropriate arguments. The training process uses distributed training with PyTorch.

### Command Line Arguments

- `--name`: Experiment name (default: '8')
- `--model`: Path to pre-trained model checkpoint (default: '/model_epoch_best.pth')
- `--tag`: Additional tag for the experiment (default: 'BAANet_17_new')
- `--dataset`: Dataset name ('endovis17' or 'endovis18', default: 'endovis17')
- `--local_rank`: Local rank for distributed training (default: -1)
- `--num_classes`: Number of output classes (default: 8)
- `--epoch_max`: Maximum number of training epochs (default: 200)
- `--image_size`: Input image size (default: 512)

### Example Training Command

```bash
python train.py --dataset endovis17 --epoch_max 200 --num_classes 8 --image_size 512
```

The training script will:
1. Load the dataset using the specified transforms
2. Initialize the model (Trainer class with SCINet backbone)
3. Train using AdamW optimizer with cosine annealing learning rate scheduler
4. Evaluate on validation set using Dice and Jaccard metrics
5. Save model checkpoints (last and best based on validation Jaccard score)

## Inference

To perform inference on test data, use the `prediction.py` script.

### Command Line Arguments

- `--data_path`: Path to data directory (default: '/opt/data/private/MGX/data')
- `--dataset`: Dataset name (default: '/endovis18')
- `--num_classes`: Number of classes (default: 8)
- `--image_size`: Image size (default: 512)
- `--model`: Path to trained model checkpoint (default: '/model_epoch_best.pth')

### Example Inference Command

```bash
python prediction.py --dataset endovis18 --model /path/to/model_epoch_best.pth --num_classes 8
```

The inference script will:
1. Load the test dataset
2. Load the trained model weights
3. Perform inference on test images
4. Calculate evaluation metrics (Dice, Jaccard, IoU)
5. Optionally save visualization results

## Data Handling

The project uses custom dataset classes for loading and preprocessing medical image data.

### Dataset Structure

The data is expected to be organized in the following structure:

```
data/
├── endovis17/
│   ├── train/
│   │   ├── images/          # Training images
│   │   └── annotations/     # Training masks/labels
│   ├── val/
│   │   ├── images/          # Validation images
│   │   └── annotations/     # Validation masks/labels
│   └── test/
│       └── images/          # Test images
└── endovis18/
    ├── train/
    │   ├── images/
    │   └── annotations/
    ├── val/
    │   ├── images/
    │   └── annotations/
    └── test/
        └── images/
```

### Data Processing

- Images are loaded using OpenCV and converted to RGB
- Masks are loaded as grayscale images
- Data augmentation and preprocessing are applied using the transforms defined in `dataloader/transforms.py`
- Images and masks are converted to PyTorch tensors
- Masks are one-hot encoded for multi-class segmentation

### Supported Datasets

- EndoVis 2017 (endovis17): Surgical instrument segmentation dataset
- EndoVis 2018 (endovis18): Similar dataset with different folds

The dataset class (`build_Dataset`) handles different data splits and applies appropriate preprocessing based on the dataset type and split (train/val/test).

## Model Architecture

The model uses a modified SAM architecture with:
- SCINet backbone for feature extraction
- Custom adapter mechanisms
- Improved decoder components
- Multi-class segmentation head

Loss functions include:
- Cross-entropy loss with class weights
- IoU loss
- Dice loss
- Boundary-aware losses

## Evaluation Metrics

The project evaluates performance using:
- Dice coefficient
- Jaccard index (IoU)
- Class-wise IoU
- Mean IoU across classes

## Notes

- The code uses distributed training, so ensure proper CUDA setup for multi-GPU training
- Adjust data paths in the dataset class according to your local data directory structure
- The model expects input images of size 512x512 by default, but this can be modified
- For best results, use the provided data preprocessing and augmentation pipelines