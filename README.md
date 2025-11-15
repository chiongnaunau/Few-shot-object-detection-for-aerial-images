# Few-Shot Object Detection for Aerial Images

An improved implementation of few-shot object detection on the NWPU VHR-10 aerial imagery dataset.

## Overview

This project implements a two-stage few-shot object detection approach with multiple improvements over baseline methods. The system achieves approximately 2x performance improvement (mAP@50: 1.9% to 3.7%) through key architectural and training enhancements.

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

Trains prototypes using contrastive learning with advanced data augmentation. Key parameters:
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
- COCO pretrained Faster R-CNN (key improvement over NWPU-trained RPN)
- Generates high-quality object proposals

**Stage 2: Prototype Classification**
- DINOv2 ViT-L/14 feature extractor (frozen, 1024-dim features)
- Learnable prototypes for novel classes
- Temperature-scaled cosine similarity matching

### Key Improvements

1. COCO Pretrained RPN
   - Transfer learning from large-scale dataset
   - Superior proposal quality compared to domain-specific training
   
2. Contrastive Prototype Learning
   - Combined cross-entropy and contrastive loss
   - Enhanced feature discrimination
   
3. Advanced Data Augmentation
   - Albumentations pipeline with geometric and photometric transforms
   - Random crops, rotations, color jittering, noise injection
   
4. Multi-Scale Inference
   - Testing at multiple scales (0.8x, 1.0x, 1.2x)
   - Per-class non-maximum suppression
   
5. Prototype Ensembling
   - Averaged prototypes from final 10 training epochs
   - Improved stability and generalization

### Class Configuration

- **Base classes** (7): ship, storage tank, basketball court, ground track field, harbor, bridge, vehicle
- **Novel classes** (3): airplane, baseball diamond, tennis court

## Expected Performance

Performance on NWPU VHR-10 test set:

| Approach | mAP@50 | Total Detections |
|----------|--------|------------------|
| Baseline (NWPU RPN) | 1.9% | 817 |
| Improved (COCO RPN) | 3.7% | 1,060 |

Per-class AP@50 for novel classes: airplane (3.7%), baseball diamond (2.4%), tennis court (1.0%).

Note: Actual results may vary based on random initialization and support set selection.

## Project Structure

```
src/
├── data_analysis.py         # Dataset analysis and support set generation
├── improved_training.py     # Prototype training with improvements
└── improved_inference.py    # Multi-scale inference and COCO evaluation

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
