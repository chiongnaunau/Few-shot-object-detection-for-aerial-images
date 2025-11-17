# Few-Shot Object Detection for Aerial Images

Implementation of few-shot object detection on the NWPU VHR-10 aerial imagery dataset using contrastive learning and DINOv2 features.

## Overview

This project implements a two-stage few-shot object detection approach for detecting novel object classes in aerial imagery with limited training examples. The system combines a pretrained COCO Faster R-CNN RPN for region proposals with DINOv2 visual features and learnable prototypes for few-shot classification.

**Dataset**: NWPU VHR-10 (1,172 images: 1,044 train, 86 validation, 42 test)  
**Novel Classes**: airplane, baseball diamond, tennis court (3 classes)  
**K-shot**: 20 examples per novel class

## Installation

```bash
pip install -r requirements.txt
```

**Key Dependencies**:
- PyTorch 2.9.1+cpu
- torchvision
- DINOv2 (facebookresearch)
- albumentations
- pycocotools
- numpy<2.0 (compatibility requirement)
- pandas, opencv-python, matplotlib

## Dataset Structure

The project expects the following data organization:

```
data/
├── train/
│   ├── *.jpg
│   └── _annotations.csv
├── valid/
│   ├── *.jpg
│   └── _annotations.csv
└── test/
    ├── *.jpg
    └── _annotations.csv
```

Annotation CSV format: `filename,width,height,class,xmin,ymin,xmax,ymax`

## Usage

### Training

Train the prototype classifier with contrastive learning:

```bash
cd MyFsDet
python src/training.py
```

**Key Parameters**:
- `--k_shot`: Number of examples per novel class (default: 20)
- `--epochs`: Training epochs (default: 30)
- `--lr`: Learning rate (default: 5e-4)
- `--temperature`: Temperature for contrastive loss (default: 0.05)
- `--contrastive_weight`: Weight for contrastive loss (default: 0.1)
- `--resume`: Resume from latest checkpoint

**Resume Training**:
If model need more training after validation
```bash
python src/training.py --resume
```

### Validation

Evaluate on validation set (86 images):

```bash
python utils/run_validation.py
```

Outputs:
- `results_validation/predictions_valid.json`: Detection predictions
- `results_validation/metrics_map50.json`: mAP@50 metrics
- `results_validation/metrics_map75.json`: mAP@75 metrics

### Inference and Visualization

Run inference on test set with visualization:

```bash
python src/inference.py --num_visualize 42
```

**Parameters**:
- `--num_visualize`: Number of images to visualize (default: 10)
- `--confidence`: Detection confidence threshold (default: 0.3)
- `--nms_threshold`: NMS IoU threshold (default: 0.5)

Outputs:
- `results/predictions.json`: COCO format predictions
- `results/result_*.jpg`: Visualization images with bounding boxes

## Methodology

### Architecture

**Two-Stage Detection Pipeline**:

1. **Region Proposal Network (RPN)**
   - COCO pretrained Faster R-CNN ResNet-50 FPN
   - Generates ~200 region proposals per image
   - Provides class-agnostic object candidates

2. **Few-Shot Classification**
   - DINOv2 ViT-L/14 feature extractor (frozen)
   - Extracts 1024-dimensional visual features
   - Learnable prototype classifier (3×1024)
   - Temperature-scaled cosine similarity matching

### Training Strategy

**Contrastive Learning Approach**:
- Combined loss: L_total = L_CE + λ×L_contrastive
- Cross-entropy for class prediction
- Contrastive loss for feature discrimination
- Temperature scaling (τ=0.05) for similarity scores

**Hyperparameters**:
- Learning rate: 5e-4
- Optimizer: Adam
- Contrastive weight: 0.1
- Temperature: 0.05
- K-shot: 20 examples per class
- Training epochs: 30
- Hardware: CPU (i5-12500H, 8GB RAM)

### Data Augmentation

Geometric transforms:
- Random horizontal/vertical flips
- Random rotation (±15°)
- Random crops and resizing

Photometric transforms:
- Color jittering
- Gaussian noise
- Brightness/contrast adjustment

## Performance Results

### Validation Set (86 images)

**Overall Metrics** (30 epochs):
- **mAP@50**: 0.39%
- **mAP@75**: 0.15%
- **Mean Precision**: 3.39%
- **Mean Recall**: 2.82%

**Per-Class Results**:

| Class | Precision | Recall | AP@50 | AP@75 | TP | FP | FN |
|-------|-----------|--------|-------|-------|----|----|-----|
| Airplane | 6.45% | 6.82% | 0.76% | 0.30% | 6 | 87 | 82 |
| Baseball Diamond | 0% | 0% | 0% | 0% | 0 | 5 | 43 |
| Tennis Court | 3.70% | 1.79% | 0.43% | 0.16% | 1 | 26 | 55 |
| **Mean** | **3.39%** | **2.82%** | **0.39%** | **0.15%** | **7** | **118** | **174** |

**Detection Summary**:
- Total ground truth objects: 181
- True positives: 7 (3.9%)
- False positives: 118
- False negatives: 174 (96.1%)

### Analysis

**Critical Issues**:
1. **Severe underperformance** compared to baseline methods (~9x worse)
2. **Baseball diamond complete failure** (0% recall)
3. **High false positive rate** (94.4% of detections are incorrect)
4. **Training instability** (loss fluctuations in billions range)

**Root Causes**:
- Frozen DINOv2 features may not be optimal for aerial imagery
- Limited training data (20-shot may be insufficient)
- CPU-only training limiting optimization convergence

## Project Structure

```
MyFsDet/
├── src/
│   ├── training.py              # Prototype training with contrastive learning
│   └── inference.py             # Multi-scale inference and visualization
├── utils/
│   ├── run_validation.py        # Validation pipeline with evaluation
│   └── run_test.py              # Test set evaluation
├── data/
│   ├── train/                   # Training images and annotations (1,044)
│   ├── valid/                   # Validation images and annotations (86)
│   └── test/                    # Test images and annotations (42)
├── saved_models/
│   ├── prototypes.pth           # Final trained prototypes (3×1024)
│   └── checkpoint_latest.pth    # Latest training checkpoint
├── results/                     # Inference visualizations (42 images)
├── results_validation/          # Validation predictions and metrics
└── requirements.txt             # Python dependencies
```


## Class Configuration

- **Base classes** (7): ship, storage tank, basketball court, ground track field, harbor, bridge, vehicle
- **Novel classes** (3): airplane, baseball diamond, tennis court

## Computational Performance

- **Training time**: ~13 hours (30 epochs on CPU)
- **Inference time**: ~10.2 seconds per image (CPU)
- **Validation time**: ~15 minutes (86 images)
- **Hardware**: Intel i5-12500H (CPU only), 8GB RAM

## References

1. Ren et al., "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks", NeurIPS 2015
2. Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision", CVPR 2023
3. Cheng et al., "Learning Rotation-Invariant Convolutional Neural Networks for Object Detection in VHR Optical Remote Sensing Images", IEEE TGRS 2016
4. Kang et al., "Few-Shot Object Detection via Feature Reweighting", ICCV 2019
5. Xiao et al., "Few-Shot Object Detection and Viewpoint Estimation for Objects in the Wild", ECCV 2020

