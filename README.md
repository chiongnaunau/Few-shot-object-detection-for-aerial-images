# Few-Shot Object Detection for Aerial Images

Implementation of few-shot object detection on the NWPU VHR-10 aerial imagery dataset.

## Overview

This project implements a two-stage few-shot object detection approach for detecting novel object classes in aerial imagery with limited training examples.

## Installation

```bash
pip install -r requirements.txt
```

Required dependencies include PyTorch, torchvision, Albumentations, pycocotools, and standard scientific Python packages.

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

### Step 1: Dataset Analysis

```bash
python src/data_analysis.py
```

Generates dataset statistics, class distributions, and few-shot support sets (5, 10, 20, 30-shot configurations).

### Step 2: Training

```bash
python src/improved_training.py --k_shot 20 --proto_epochs 50 --device cuda
```

Trains prototypes using contrastive learning and data augmentation. Key parameters:
- `--k_shot`: Number of examples per novel class (default: 20)
- `--proto_epochs`: Training epochs for prototypes (default: 50)
- `--device`: cuda or cpu

### Step 3: Inference and Evaluation

```bash
python src/improved_inference.py --multi_scale --confidence 0.3
```

Runs multi-scale inference and computes COCO evaluation metrics. Parameters:
- `--multi_scale`: Enable multi-scale testing
- `--confidence`: Detection confidence threshold (default: 0.3)

## Methodology

### Architecture

**Stage 1: Region Proposal Network**
- COCO pretrained Faster R-CNN for region proposals
- Generates object proposal candidates

**Stage 2: Prototype Classification**
- DINOv2 ViT-L/14 feature extractor (frozen, 1024-dim features)
- Learnable prototypes for novel classes
- Temperature-scaled cosine similarity matching

### Key Components

1. Region Proposal Network
   - Pretrained Faster R-CNN from COCO dataset
   - Generates high-quality region proposals
   
2. Prototype Learning
   - Cross-entropy and contrastive loss combination
   - Feature-based prototype matching
   
3. Data Augmentation
   - Geometric transforms (crops, flips, rotations)
   - Photometric transforms (color jittering, noise)
   
4. Multi-Scale Inference
   - Testing at multiple scales (0.8x, 1.0x, 1.2x)
   - Non-maximum suppression per class
   
5. Prototype Ensembling
   - Averaged prototypes from multiple training epochs
   - Improved generalization

### Class Configuration

- **Base classes** (7): ship, storage tank, basketball court, ground track field, harbor, bridge, vehicle
- **Novel classes** (3): airplane, baseball diamond, tennis court

## Project Structure

```
src/
├── data_analysis.py         # Dataset analysis and support set generation
├── training.py              # Prototype training implementation
└── inference.py             # Multi-scale inference and evaluation

data/                        # Dataset images and annotations
saved_model/                 # Trained model checkpoints (generated)
requirements.txt             # Python dependencies
```

## Hyperparameters

Training:
- Optimizer: AdamW (lr=5e-4, weight_decay=1e-4)
- Scheduler: CosineAnnealingLR
- Temperature: 0.05
- Batch size: 8

Inference:
- Confidence threshold: 0.3
- NMS threshold: 0.5
- Multi-scale factors: [0.8, 1.0, 1.2]

## References

1. Ren et al., "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks"
2. Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision"
3. Cheng et al., "Learning Rotation-Invariant Convolutional Neural Networks for Object Detection in VHR Optical Remote Sensing Images"
4. Kang et al., "Few-Shot Object Detection via Feature Reweighting"
