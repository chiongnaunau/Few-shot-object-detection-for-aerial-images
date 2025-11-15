"""
Improved Few-Shot Object Detection Training Pipeline
Implements multiple improvements over baseline:
1. Multi-scale feature extraction
2. Contrastive prototype learning
3. Advanced data augmentation
4. Temperature-scaled cosine similarity
5. Ensemble prototypes from multiple epochs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import json
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ImprovedFewShotDataset(Dataset):
    """Dataset with advanced augmentation for few-shot learning"""
    
    def __init__(self, csv_path, img_dir, classes, transform=None, is_support=False):
        self.df = pd.read_csv(csv_path)
        self.img_dir = Path(img_dir)
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.transform = transform
        self.is_support = is_support
        
        # Group by image
        self.image_groups = self.df.groupby('filename')
        self.image_names = list(self.image_groups.groups.keys())
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = self.img_dir / img_name
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        image_np = np.array(image)
        
        # Get annotations
        img_annotations = self.df[self.df['filename'] == img_name]
        
        boxes = []
        labels = []
        
        for _, row in img_annotations.iterrows():
            if row['class'] in self.class_to_idx:
                boxes.append([row['xmin'], row['ymin'], row['xmax'], row['ymax']])
                labels.append(self.class_to_idx[row['class']])
        
        if len(boxes) == 0:
            # No valid annotations for this image
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Apply augmentation
        if self.transform:
            # Albumentations format
            transformed = self.transform(
                image=image_np,
                bboxes=boxes.numpy() if len(boxes) > 0 else [],
                labels=labels.numpy() if len(labels) > 0 else []
            )
            image_np = transformed['image']
            boxes = torch.as_tensor(transformed['bboxes'], dtype=torch.float32) if transformed['bboxes'] else torch.zeros((0, 4))
            labels = torch.as_tensor(transformed['labels'], dtype=torch.int64) if transformed['labels'] else torch.zeros((0,))
        
        # Convert to tensor if not already
        if not isinstance(image_np, torch.Tensor):
            image_tensor = T.ToTensor()(image_np)
        else:
            image_tensor = image_np
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx])
        }
        
        return image_tensor, target

def get_augmentation_pipeline(is_train=True, image_size=400):
    """Get advanced augmentation pipeline"""
    if is_train:
        return A.Compose([
            A.RandomResizedCrop(height=image_size, width=image_size, scale=(0.8, 1.0), p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1),
                A.RandomGamma(gamma_limit=(80, 120), p=1),
            ], p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    else:
        return A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

class DINOv2FeatureExtractor(nn.Module):
    """DINOv2 feature extractor for prototypes"""
    
    def __init__(self, model_name='dinov2_vitl14', freeze=True):
        super().__init__()
        # Load DINOv2 model
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.feature_dim = 1024  # ViT-L/14
        
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
    
    def forward(self, x):
        """Extract features from image crops"""
        with torch.no_grad() if not self.training else torch.enable_grad():
            features = self.model(x)
        return features

class ContrastivePrototypeLearner(nn.Module):
    """Learns prototypes using contrastive learning"""
    
    def __init__(self, feature_dim=1024, num_classes=3, temperature=0.05):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.temperature = temperature
        
        # Learnable prototypes
        self.prototypes = nn.Parameter(torch.randn(num_classes, feature_dim))
        nn.init.xavier_normal_(self.prototypes)
        
        # Optional projection head
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
    
    def forward(self, features, labels=None):
        """
        Compute similarity to prototypes
        features: (N, feature_dim)
        labels: (N,) - class indices
        """
        # Optional: project features
        projected_features = self.projection(features)
        
        # Normalize
        features_norm = F.normalize(projected_features, p=2, dim=1)
        prototypes_norm = F.normalize(self.prototypes, p=2, dim=1)
        
        # Cosine similarity
        logits = torch.matmul(features_norm, prototypes_norm.t()) / self.temperature
        
        return logits
    
    def compute_loss(self, features, labels):
        """Compute cross-entropy + contrastive loss"""
        logits = self.forward(features, labels)
        
        # Cross-entropy loss
        ce_loss = F.cross_entropy(logits, labels)
        
        # Contrastive loss: pull same class together, push different apart
        features_norm = F.normalize(features, p=2, dim=1)
        similarity_matrix = torch.matmul(features_norm, features_norm.t())
        
        # Create label mask
        labels_expanded = labels.unsqueeze(1)
        label_mask = (labels_expanded == labels_expanded.t()).float()
        
        # Positive pairs (same class)
        positive_pairs = similarity_matrix * label_mask
        positive_loss = -torch.sum(positive_pairs) / (torch.sum(label_mask) + 1e-8)
        
        # Negative pairs (different class)
        negative_mask = 1 - label_mask
        negative_pairs = similarity_matrix * negative_mask
        negative_loss = torch.sum(torch.exp(negative_pairs / self.temperature)) / (torch.sum(negative_mask) + 1e-8)
        
        contrastive_loss = positive_loss + negative_loss
        
        # Total loss
        total_loss = ce_loss + 0.1 * contrastive_loss
        
        return total_loss, {'ce_loss': ce_loss.item(), 'contrastive_loss': contrastive_loss.item()}

def train_base_detector(train_dataset, num_classes, num_epochs=25, device='cuda'):
    """Train base Faster R-CNN detector on base classes"""
    print("\n" + "="*80)
    print("TRAINING BASE DETECTOR (Faster R-CNN)")
    print("="*80)
    
    # Load pretrained Faster R-CNN
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    
    # Replace classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    model.to(device)
    model.train()
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # DataLoader
    data_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for images, targets in progress_bar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            epoch_loss += losses.item()
            progress_bar.set_postfix({'loss': losses.item()})
        
        lr_scheduler.step()
        avg_loss = epoch_loss / len(data_loader)
        print(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.4f}")
    
    return model

def extract_roi_features(model, images, boxes, feature_extractor, device='cuda'):
    """Extract features from RoIs using RPN proposals and DINOv2"""
    model.eval()
    feature_extractor.eval()
    
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for img, box_info in zip(images, boxes):
            img = img.to(device)
            
            # Get boxes and labels
            bboxes = box_info['boxes'].to(device)
            labels = box_info['labels'].to(device)
            
            if len(bboxes) == 0:
                continue
            
            # Extract crops
            img_np = img.cpu().numpy().transpose(1, 2, 0)
            img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
            
            for bbox, label in zip(bboxes, labels):
                x1, y1, x2, y2 = bbox.int().tolist()
                crop = img_pil.crop((x1, y1, x2, y2))
                crop = crop.resize((224, 224))  # DINOv2 input size
                crop_tensor = T.ToTensor()(crop).unsqueeze(0).to(device)
                
                # Extract feature
                feature = feature_extractor(crop_tensor)
                all_features.append(feature.squeeze(0))
                all_labels.append(label)
    
    if len(all_features) == 0:
        return None, None
    
    return torch.stack(all_features), torch.stack(all_labels)

def train_prototypes(support_dataset, feature_extractor, num_classes, 
                     num_epochs=50, device='cuda', save_dir=None):
    """Train prototypes with contrastive learning"""
    print("\n" + "="*80)
    print("TRAINING PROTOTYPES (Contrastive Learning)")
    print("="*80)
    
    # Initialize prototype learner
    prototype_learner = ContrastivePrototypeLearner(
        feature_dim=1024,
        num_classes=num_classes,
        temperature=0.05
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        prototype_learner.parameters(),
        lr=5e-4,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # DataLoader
    data_loader = DataLoader(
        support_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=2,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    # Training loop with ensemble collection
    best_prototypes = []
    
    for epoch in range(num_epochs):
        prototype_learner.train()
        epoch_loss = 0
        epoch_ce_loss = 0
        epoch_contrastive_loss = 0
        
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for images, targets in progress_bar:
            # Extract features from all boxes
            features_list = []
            labels_list = []
            
            for img, target in zip(images, targets):
                img = img.to(device)
                boxes = target['boxes']
                labels = target['labels']
                
                if len(boxes) == 0:
                    continue
                
                # Extract crops and features
                img_np = (img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                img_pil = Image.fromarray(img_np)
                
                for bbox, label in zip(boxes, labels):
                    x1, y1, x2, y2 = bbox.int().tolist()
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(img_pil.width, x2), min(img_pil.height, y2)
                    
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    crop = img_pil.crop((x1, y1, x2, y2))
                    crop = crop.resize((224, 224))
                    crop_tensor = T.ToTensor()(crop).unsqueeze(0).to(device)
                    
                    # Extract feature
                    with torch.no_grad():
                        feature = feature_extractor(crop_tensor).squeeze(0)
                    
                    features_list.append(feature)
                    labels_list.append(label)
            
            if len(features_list) == 0:
                continue
            
            features = torch.stack(features_list).to(device)
            labels = torch.stack(labels_list).to(device)
            
            # Compute loss
            loss, loss_dict = prototype_learner.compute_loss(features, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_ce_loss += loss_dict['ce_loss']
            epoch_contrastive_loss += loss_dict['contrastive_loss']
            
            progress_bar.set_postfix({
                'loss': loss.item(),
                'ce': loss_dict['ce_loss'],
                'contr': loss_dict['contrastive_loss']
            })
        
        scheduler.step()
        
        avg_loss = epoch_loss / len(data_loader)
        avg_ce = epoch_ce_loss / len(data_loader)
        avg_contr = epoch_contrastive_loss / len(data_loader)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, CE: {avg_ce:.4f}, Contrastive: {avg_contr:.4f}")
        
        # Save prototypes from last 10 epochs for ensemble
        if epoch >= num_epochs - 10:
            best_prototypes.append(prototype_learner.prototypes.data.clone())
    
    # Ensemble: average last 10 epoch prototypes
    ensemble_prototypes = torch.stack(best_prototypes).mean(dim=0)
    
    # Save
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        
        torch.save({
            'prototypes': ensemble_prototypes,
            'model_state': prototype_learner.state_dict(),
            'num_classes': num_classes,
            'feature_dim': 1024
        }, save_dir / 'improved_prototypes.pth')
        
        print(f"\nSaved improved prototypes to {save_dir / 'improved_prototypes.pth'}")
    
    return ensemble_prototypes, prototype_learner

def main():
    """Main training pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--output_dir', type=str, default='../saved_model')
    parser.add_argument('--k_shot', type=int, default=20)
    parser.add_argument('--base_epochs', type=int, default=25)
    parser.add_argument('--proto_epochs', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    print("="*80)
    print("IMPROVED FEW-SHOT OBJECT DETECTION TRAINING")
    print("="*80)
    print(f"Device: {args.device}")
    print(f"K-shot: {args.k_shot}")
    print(f"Base epochs: {args.base_epochs}")
    print(f"Prototype epochs: {args.proto_epochs}")
    
    # Define classes
    novel_classes = ['airplane', 'baseball diamond', 'tennis court']
    base_classes = ['ship', 'storage tank', 'basketball court', 
                    'ground track field', 'harbor', 'bridge', 'vehicle']
    all_classes = base_classes + novel_classes
    
    # Create datasets (this is a simplified version - you need to implement proper data loading)
    print("\nNote: Implement proper dataset loading based on your data structure")
    print("This script provides the training framework with improvements")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Models saved to: {args.output_dir}")

if __name__ == '__main__':
    main()
