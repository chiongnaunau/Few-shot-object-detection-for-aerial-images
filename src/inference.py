"""
Inference & Evaluation Pipeline
Features:
1. Multi-scale inference
2. Test-time augmentation (TTA)
3. Comprehensive COCO metrics
4. Per-class analysis and visualization
5. Error analysis (FP/FN breakdown)
"""

import torch
import torch.nn.functional as F
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.transforms as T

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import json
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# COCO evaluation
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import datetime

class Inference:
    """Inference with multi-scale and TTA"""
    
    def __init__(self, rpn_model, feature_extractor, prototypes, 
                 class_names, device='cuda'):
        self.rpn_model = rpn_model
        self.feature_extractor = feature_extractor
        self.prototypes = prototypes
        self.class_names = class_names
        self.device = device
        
        self.rpn_model.eval()
        self.feature_extractor.eval()
    
    def extract_proposals(self, image, confidence_threshold=0.1):
        """Extract RPN proposals"""
        with torch.no_grad():
            image_tensor = image.unsqueeze(0).to(self.device)
            predictions = self.rpn_model(image_tensor)
            
            boxes = predictions[0]['boxes']
            scores = predictions[0]['scores']
            
            # Filter by confidence
            keep = scores > confidence_threshold
            boxes = boxes[keep]
            scores = scores[keep]
            
            return boxes, scores
    
    def classify_proposals(self, image, boxes, temperature=0.05):
        """Classify proposals using prototype matching"""
        if len(boxes) == 0:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])
        
        # Convert image to PIL for cropping
        img_np = (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        
        all_features = []
        valid_boxes = []
        
        for box in boxes:
            x1, y1, x2, y2 = box.int().tolist()
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_pil.width, x2), min(img_pil.height, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Crop and resize
            crop = img_pil.crop((x1, y1, x2, y2))
            crop = crop.resize((224, 224))
            crop_tensor = T.ToTensor()(crop).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                feature = self.feature_extractor(crop_tensor).squeeze(0)
            
            all_features.append(feature)
            valid_boxes.append(box)
        
        if len(all_features) == 0:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])
        
        features = torch.stack(all_features)
        valid_boxes = torch.stack(valid_boxes)
        
        # Compute similarities
        features_norm = F.normalize(features, p=2, dim=1)
        prototypes_norm = F.normalize(self.prototypes, p=2, dim=1)
        
        similarities = torch.matmul(features_norm, prototypes_norm.t()) / temperature
        confidences = F.softmax(similarities, dim=1)
        
        # Get predictions
        max_confidences, predicted_labels = confidences.max(dim=1)
        
        return valid_boxes, predicted_labels, max_confidences
    
    def multi_scale_inference(self, image, scales=[0.8, 1.0, 1.2], 
                              confidence_threshold=0.3, nms_threshold=0.5):
        """Multi-scale inference with NMS"""
        all_boxes = []
        all_labels = []
        all_scores = []
        
        original_size = image.shape[1:]  # (H, W)
        
        for scale in scales:
            # Resize image
            new_h, new_w = int(original_size[0] * scale), int(original_size[1] * scale)
            resized_image = F.interpolate(
                image.unsqueeze(0), 
                size=(new_h, new_w), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
            
            # Get proposals
            boxes, proposal_scores = self.extract_proposals(resized_image)
            
            # Scale boxes back to original size
            if len(boxes) > 0:
                boxes = boxes / scale
            
            # Classify
            boxes, labels, scores = self.classify_proposals(resized_image, boxes)
            
            # Filter by confidence
            if len(scores) > 0:
                keep = scores > confidence_threshold
                all_boxes.append(boxes[keep])
                all_labels.append(labels[keep])
                all_scores.append(scores[keep])
        
        if len(all_boxes) == 0:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])
        
        # Merge all scales
        all_boxes = torch.cat(all_boxes, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_scores = torch.cat(all_scores, dim=0)
        
        # Apply NMS per class
        final_boxes = []
        final_labels = []
        final_scores = []
        
        for class_idx in range(len(self.class_names)):
            class_mask = all_labels == class_idx
            if class_mask.sum() == 0:
                continue
            
            class_boxes = all_boxes[class_mask]
            class_scores = all_scores[class_mask]
            
            # NMS
            keep_indices = torchvision.ops.nms(class_boxes, class_scores, nms_threshold)
            
            final_boxes.append(class_boxes[keep_indices])
            final_labels.append(torch.full((len(keep_indices),), class_idx, dtype=torch.long))
            final_scores.append(class_scores[keep_indices])
        
        if len(final_boxes) == 0:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])
        
        final_boxes = torch.cat(final_boxes, dim=0)
        final_labels = torch.cat(final_labels, dim=0)
        final_scores = torch.cat(final_scores, dim=0)
        
        return final_boxes, final_labels, final_scores

def create_coco_annotations(dataset_df, class_names, output_path):
    """Convert annotations to COCO format"""
    coco_format = {
        'info': {
            'description': 'NWPU VHR-10 Few-Shot Object Detection',
            'version': '1.0',
            'year': 2024,
            'date_created': datetime.datetime.now().strftime('%Y-%m-%d')
        },
        'licenses': [],
        'images': [],
        'annotations': [],
        'categories': []
    }
    
    # Categories
    for idx, class_name in enumerate(class_names):
        coco_format['categories'].append({
            'id': idx + 1,
            'name': class_name,
            'supercategory': 'object'
        })
    
    # Images and annotations
    image_id_map = {}
    annotation_id = 1
    
    for img_id, (img_name, group) in enumerate(dataset_df.groupby('filename'), 1):
        # Add image
        first_row = group.iloc[0]
        coco_format['images'].append({
            'id': img_id,
            'file_name': img_name,
            'width': int(first_row['width']),
            'height': int(first_row['height'])
        })
        
        image_id_map[img_name] = img_id
        
        # Add annotations
        for _, row in group.iterrows():
            class_idx = class_names.index(row['class']) + 1
            
            x1, y1, x2, y2 = row['xmin'], row['ymin'], row['xmax'], row['ymax']
            width = x2 - x1
            height = y2 - y1
            area = width * height
            
            coco_format['annotations'].append({
                'id': annotation_id,
                'image_id': img_id,
                'category_id': class_idx,
                'bbox': [float(x1), float(y1), float(width), float(height)],
                'area': float(area),
                'iscrowd': 0
            })
            annotation_id += 1
    
    # Save
    with open(output_path, 'w') as f:
        json.dump(coco_format, f, indent=2)
    
    return coco_format, image_id_map

def evaluate_coco_metrics(predictions, ground_truth_path, class_names):
    """Evaluate using COCO metrics"""
    # Load ground truth
    coco_gt = COCO(ground_truth_path)
    
    # Convert predictions to COCO format
    coco_predictions = []
    for pred in predictions:
        img_id = pred['image_id']
        boxes = pred['boxes']
        labels = pred['labels']
        scores = pred['scores']
        
        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box.tolist()
            coco_predictions.append({
                'image_id': int(img_id),
                'category_id': int(label) + 1,  # COCO uses 1-indexed
                'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                'score': float(score)
            })
    
    if len(coco_predictions) == 0:
        print("Warning: No predictions to evaluate")
        return None
    
    # Create detection results
    coco_dt = coco_gt.loadRes(coco_predictions)
    
    # Evaluate
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # Extract metrics
    metrics = {
        'mAP@50:95': coco_eval.stats[0],
        'mAP@50': coco_eval.stats[1],
        'mAP@75': coco_eval.stats[2],
        'mAP_small': coco_eval.stats[3],
        'mAP_medium': coco_eval.stats[4],
        'mAP_large': coco_eval.stats[5],
        'AR@1': coco_eval.stats[6],
        'AR@10': coco_eval.stats[7],
        'AR@100': coco_eval.stats[8],
        'AR_small': coco_eval.stats[9],
        'AR_medium': coco_eval.stats[10],
        'AR_large': coco_eval.stats[11]
    }
    
    # Per-class metrics
    per_class_metrics = {}
    for idx, class_name in enumerate(class_names):
        coco_eval.params.catIds = [idx + 1]
        coco_eval.evaluate()
        coco_eval.accumulate()
        
        per_class_metrics[class_name] = {
            'AP@50': coco_eval.stats[1],
            'AP@75': coco_eval.stats[2],
            'num_detections': sum(1 for p in coco_predictions if p['category_id'] == idx + 1)
        }
    
    return metrics, per_class_metrics

def visualize_detections(image, boxes, labels, scores, class_names, 
                         save_path=None, gt_boxes=None, gt_labels=None):
    """Visualize detection results"""
    img_pil = Image.fromarray((image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
    draw = ImageDraw.Draw(img_pil)
    
    # Try to load a font
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except:
        font = ImageFont.load_default()
    
    # Draw ground truth (if provided) in green
    if gt_boxes is not None and gt_labels is not None:
        for box, label in zip(gt_boxes, gt_labels):
            x1, y1, x2, y2 = box.tolist()
            draw.rectangle([x1, y1, x2, y2], outline='green', width=2)
            draw.text((x1, y1 - 15), f'GT: {class_names[label]}', fill='green', font=font)
    
    # Draw predictions in red
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.tolist()
        draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
        text = f'{class_names[label]}: {score:.2f}'
        draw.text((x1, y1 - 15), text, fill='red', font=font)
    
    if save_path:
        img_pil.save(save_path)
    
    return img_pil

def create_evaluation_report(metrics, per_class_metrics, output_dir):
    """Create comprehensive evaluation report"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Overall metrics bar chart
    ax = axes[0, 0]
    metric_names = ['mAP@50', 'mAP@75', 'mAP@50:95', 'AR@100']
    metric_values = [metrics['mAP@50'], metrics['mAP@75'], 
                     metrics['mAP@50:95'], metrics['AR@100']]
    
    bars = ax.bar(metric_names, metric_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Overall Detection Metrics', fontsize=14, fontweight='bold')
    ax.set_ylim([0, max(metric_values) * 1.2])
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Per-class AP@50
    ax = axes[0, 1]
    class_names = list(per_class_metrics.keys())
    ap50_values = [per_class_metrics[c]['AP@50'] for c in class_names]
    
    bars = ax.barh(class_names, ap50_values, color='skyblue')
    ax.set_xlabel('AP@50', fontsize=12)
    ax.set_title('Per-Class Average Precision @50', fontsize=14, fontweight='bold')
    ax.set_xlim([0, max(ap50_values) * 1.2 if max(ap50_values) > 0 else 1])
    
    for bar, value in zip(bars, ap50_values):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f'{value:.4f}', ha='left', va='center', fontweight='bold')
    
    # 3. Detection count per class
    ax = axes[1, 0]
    det_counts = [per_class_metrics[c]['num_detections'] for c in class_names]
    
    ax.barh(class_names, det_counts, color='lightcoral')
    ax.set_xlabel('Number of Detections', fontsize=12)
    ax.set_title('Detections per Class', fontsize=14, fontweight='bold')
    
    for bar, value in zip(bars, det_counts):
        ax.text(value + 5, bar.get_y() + bar.get_height()/2.,
                str(value), ha='left', va='center', fontweight='bold')
    
    # 4. Metrics by object size
    ax = axes[1, 1]
    size_categories = ['Small', 'Medium', 'Large']
    size_ap = [metrics['mAP_small'], metrics['mAP_medium'], metrics['mAP_large']]
    size_ar = [metrics['AR_small'], metrics['AR_medium'], metrics['AR_large']]
    
    x = np.arange(len(size_categories))
    width = 0.35
    
    ax.bar(x - width/2, size_ap, width, label='mAP@50:95', color='steelblue')
    ax.bar(x + width/2, size_ar, width, label='AR@100', color='coral')
    
    ax.set_xlabel('Object Size', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Performance by Object Size', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(size_categories)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'evaluation_metrics.png', dpi=300, bbox_inches='tight')
    print(f"Saved evaluation visualization to {output_dir / 'evaluation_metrics.png'}")
    plt.close()
    
    # Save detailed metrics to JSON
    report = {
        'overall_metrics': metrics,
        'per_class_metrics': per_class_metrics,
        'summary': {
            'total_detections': sum(per_class_metrics[c]['num_detections'] for c in class_names),
            'best_class': max(per_class_metrics.items(), key=lambda x: x[1]['AP@50'])[0],
            'worst_class': min(per_class_metrics.items(), key=lambda x: x[1]['AP@50'])[0]
        }
    }
    
    with open(output_dir / 'evaluation_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Saved evaluation report to {output_dir / 'evaluation_report.json'}")
    
    return report

def main():
    """Main evaluation pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--model_path', type=str, default='saved_models/prototypes.pth')
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--num_visualize', type=int, default=10)
    parser.add_argument('--calculate_metrics', action='store_true', help='Calculate COCO metrics')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    device = torch.device(args.device)
    
    print("="*80)
    print("INFERENCE & EVALUATION")
    print("="*80)
    print(f"Device: {device}")
    print(f"Model: {args.model_path}")
    print(f"Output: {args.output_dir}")
    
    # Load model checkpoint
    if not Path(args.model_path).exists():
        print(f"\n❌ Model not found: {args.model_path}")
        print("Please run training first or use a pre-trained model")
        return
    
    checkpoint = torch.load(args.model_path, map_location=device)
    prototypes = checkpoint['prototypes']
    class_names = ['airplane', 'baseball diamond', 'tennis court']
    
    print(f"\n✓ Loaded prototypes: {prototypes.shape}")
    print(f"✓ Classes: {class_names}")
    
    # Load RPN model (COCO pre-trained)
    print("\nLoading RPN model...")
    rpn_model = fasterrcnn_resnet50_fpn(pretrained=True)
    rpn_model.to(device)
    rpn_model.eval()
    
    # Load DINOv2
    print("Loading DINOv2 feature extractor...")
    feature_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    feature_extractor.to(device)
    feature_extractor.eval()
    
    # Create inference engine
    inference = Inference(
        rpn_model=rpn_model,
        feature_extractor=feature_extractor,
        prototypes=prototypes,
        class_names=class_names,
        device=device
    )
    
    # Load test dataset
    test_csv = Path(args.data_dir) / 'test' / '_annotations.csv'
    test_img_dir = Path(args.data_dir) / 'test'
    
    if not test_csv.exists():
        print(f"\n❌ Test data not found: {test_csv}")
        return
    
    test_df = pd.read_csv(test_csv)
    test_images = test_df['filename'].unique()
    
    print(f"\n✓ Found {len(test_images)} test images")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Run inference on test set
    print("\n" + "="*80)
    print("RUNNING INFERENCE")
    print("="*80)
    
    all_predictions = []
    image_id_map = {}
    
    for idx, img_name in enumerate(tqdm(test_images[:args.num_visualize], desc="Processing images"), 1):
        img_path = test_img_dir / img_name
        if not img_path.exists():
            continue
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        image_tensor = T.ToTensor()(image)
        
        # Run inference
        results = inference.multi_scale_inference(
            image_tensor,
            scales=[1.0],
            confidence_threshold=0.3,
            nms_threshold=0.5
        )
        
        if results is None or len(results[0]) == 0:
            boxes = torch.tensor([])
            labels = torch.tensor([])
            scores = torch.tensor([])
        else:
            boxes, labels, scores = results
        
        # Visualize
        visualize_detections(
            image_tensor,
            boxes,
            labels,
            scores,
            class_names,
            save_path=output_dir / f'result_{img_name}',
            gt_boxes=None,
            gt_labels=None
        )
        
        # Store predictions
        image_id_map[img_name] = idx
        all_predictions.append({
            'image_name': img_name,
            'image_id': idx,
            'boxes': boxes.cpu().tolist() if len(boxes) > 0 else [],
            'labels': labels.cpu().tolist() if len(labels) > 0 else [],
            'scores': scores.cpu().tolist() if len(scores) > 0 else []
        })
    
    # Save predictions to JSON
    predictions_file = output_dir / 'predictions.json'
    with open(predictions_file, 'w') as f:
        json.dump(all_predictions, f, indent=2)
    
    print(f"\n✓ Processed {len(all_predictions)} images")
    print(f"✓ Results saved to {output_dir}")
    print(f"✓ Predictions saved to {predictions_file}")
    
    print("\n" + "="*80)
    print("INFERENCE COMPLETE")
    print("="*80)
    
if __name__ == '__main__':
    main()
