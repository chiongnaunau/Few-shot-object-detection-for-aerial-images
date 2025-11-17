"""
Validation Script - Complete
Combines inference + evaluation in one file
"""

import torch
import torch.nn.functional as F
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.ops import nms
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import json
from tqdm import tqdm
from collections import defaultdict


# ============================================================================
# EVALUATION FUNCTIONS (from evaluation.py)
# ============================================================================

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x1, y1, x2, y2]"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    inter_x1 = max(x1_min, x2_min)
    inter_y1 = max(y1_min, y2_min)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)
    
    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0.0
    
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def calculate_ap(precisions, recalls):
    """Calculate Average Precision using 11-point interpolation"""
    ap = 0.0
    for threshold in np.arange(0, 1.1, 0.1):
        if np.sum(recalls >= threshold) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= threshold])
        ap += p / 11.0
    return ap


def evaluate_detections(predictions_file, ground_truth_csv, class_names, iou_threshold=0.5):
    """Evaluate object detections"""
    
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)
    
    gt_df = pd.read_csv(ground_truth_csv)
    gt_df = gt_df[gt_df['class'].isin(class_names)]
    
    gt_by_image = defaultdict(lambda: defaultdict(list))
    for _, row in gt_df.iterrows():
        gt_by_image[row['filename']][row['class']].append([
            row['xmin'], row['ymin'], row['xmax'], row['ymax']
        ])
    
    class_metrics = {}
    all_precisions = []
    all_recalls = []
    
    for class_name in class_names:
        class_idx = class_names.index(class_name)
        all_pred_boxes = []
        all_pred_scores = []
        all_gt_boxes = []
        total_gt = 0
        
        for pred in predictions:
            img_name = pred['image_name']
            
            if 'boxes' in pred and len(pred['boxes']) > 0:
                pred_boxes = np.array(pred['boxes'])
                pred_labels = np.array(pred['labels'])
                pred_scores = np.array(pred['scores'])
                
                class_mask = pred_labels == class_idx
                class_boxes = pred_boxes[class_mask]
                class_scores = pred_scores[class_mask]
                
                all_pred_boxes.extend(class_boxes)
                all_pred_scores.extend(class_scores)
            
            if img_name in gt_by_image and class_name in gt_by_image[img_name]:
                gt_boxes = gt_by_image[img_name][class_name]
                total_gt += len(gt_boxes)
                all_gt_boxes.extend([(img_name, gt_box) for gt_box in gt_boxes])
        
        if len(all_pred_boxes) == 0:
            class_metrics[class_name] = {
                'AP@50': 0.0, 'precision': 0.0, 'recall': 0.0,
                'num_predictions': 0, 'num_ground_truth': total_gt,
                'true_positives': 0, 'false_positives': 0, 'false_negatives': total_gt
            }
            continue
        
        sorted_indices = np.argsort(all_pred_scores)[::-1]
        sorted_boxes = [all_pred_boxes[i] for i in sorted_indices]
        
        tp = np.zeros(len(sorted_boxes))
        fp = np.zeros(len(sorted_boxes))
        matched_gt = set()
        
        for pred_idx, pred_box in enumerate(sorted_boxes):
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, (img_name, gt_box) in enumerate(all_gt_boxes):
                if gt_idx in matched_gt:
                    continue
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold:
                tp[pred_idx] = 1
                matched_gt.add(best_gt_idx)
            else:
                fp[pred_idx] = 1
        
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
        recalls = tp_cumsum / (total_gt + 1e-8)
        ap = calculate_ap(precisions, recalls)
        
        total_tp = int(tp_cumsum[-1]) if len(tp_cumsum) > 0 else 0
        total_fp = int(fp_cumsum[-1]) if len(fp_cumsum) > 0 else 0
        total_fn = total_gt - total_tp
        
        final_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        final_recall = total_tp / total_gt if total_gt > 0 else 0
        
        class_metrics[class_name] = {
            'AP@50': ap,
            'precision': final_precision,
            'recall': final_recall,
            'num_predictions': len(all_pred_boxes),
            'num_ground_truth': total_gt,
            'true_positives': total_tp,
            'false_positives': total_fp,
            'false_negatives': total_fn
        }
        
        all_precisions.append(final_precision)
        all_recalls.append(final_recall)
    
    mean_ap = np.mean([m['AP@50'] for m in class_metrics.values()])
    mean_precision = np.mean(all_precisions)
    mean_recall = np.mean(all_recalls)
    
    return {
        'mAP@50': mean_ap,
        'mean_precision': mean_precision,
        'mean_recall': mean_recall,
        'per_class_metrics': class_metrics
    }


def print_evaluation_report(metrics):
    """Print formatted evaluation report"""
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    print(f"\nOverall Metrics:")
    print(f"  mAP@50:         {metrics['mAP@50']:.4f} ({metrics['mAP@50']*100:.2f}%)")
    print(f"  Mean Precision: {metrics['mean_precision']:.4f} ({metrics['mean_precision']*100:.2f}%)")
    print(f"  Mean Recall:    {metrics['mean_recall']:.4f} ({metrics['mean_recall']*100:.2f}%)")
    
    print(f"\nPer-Class Metrics:")
    print(f"{'Class':<25} {'AP@50':>8} {'Prec':>8} {'Recall':>8} {'TP':>6} {'FP':>6} {'FN':>6} {'GT':>6}")
    print("-"*80)
    
    for class_name, m in metrics['per_class_metrics'].items():
        print(f"{class_name:<25} {m['AP@50']:>7.1%} {m['precision']:>7.1%} {m['recall']:>7.1%} "
              f"{m['true_positives']:>6} {m['false_positives']:>6} {m['false_negatives']:>6} "
              f"{m['num_ground_truth']:>6}")
    
    print("="*80)


def save_evaluation_results(metrics, output_file):
    """Save evaluation results to JSON"""
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✓ Results saved to {output_file}")


# ============================================================================
# INFERENCE FUNCTIONS
# ============================================================================



def load_dinov2():
    """Load DINOv2 ViT-L/14"""
    print("Loading DINOv2 ViT-L/14...")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    model.eval()
    return model


def extract_dinov2_features(dinov2_model, image_tensor, device):
    """Extract features using DINOv2"""
    with torch.no_grad():
        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
        normalized = (image_tensor - mean) / std
        
        # Extract features
        features = dinov2_model.forward_features(normalized.unsqueeze(0))
        cls_token = features['x_norm_clstoken'].squeeze(0)  # [1024]
        
        return cls_token


def run_detection(image, rpn_model, dinov2_model, prototypes, device,
                  conf_threshold=0.05, nms_threshold=0.5):
    """
    Run object detection on single image
    Same logic as training.py
    """
    
    # Convert PIL to tensor if needed
    if isinstance(image, Image.Image):
        img_tensor = torchvision.transforms.ToTensor()(image).to(device)
    else:
        img_tensor = image.to(device)
    
    # Step 1: Get RPN proposals
    with torch.no_grad():
        rpn_out = rpn_model([img_tensor])
        boxes = rpn_out[0]['boxes']
    
    if len(boxes) == 0:
        return [], [], []
    
    # Limit proposals
    if len(boxes) > 1000:
        boxes = boxes[:1000]
    
    # Step 2: Extract features for each proposal
    all_boxes = []
    all_labels = []
    all_scores = []
    
    batch_size = 32
    for i in range(0, len(boxes), batch_size):
        batch_boxes = boxes[i:i+batch_size]
        regions = []
        valid_boxes = []
        
        for box in batch_boxes:
            x1, y1, x2, y2 = box.int().tolist()
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_tensor.shape[2], x2)
            y2 = min(img_tensor.shape[1], y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Crop region
            region = img_tensor[:, y1:y2, x1:x2]
            
            # Resize to 224x224
            region_resized = F.interpolate(
                region.unsqueeze(0),
                size=(224, 224),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            
            regions.append(region_resized)
            valid_boxes.append(box)
        
        if len(regions) == 0:
            continue
        
        # Stack and extract features
        regions_batch = torch.stack(regions)
        
        # Extract DINOv2 features
        batch_features = []
        for region in regions_batch:
            feat = extract_dinov2_features(dinov2_model, region, device)
            batch_features.append(feat)
        
        features = torch.stack(batch_features)  # [N, 1024]
        
        # Step 3: Classify with prototypes
        features_norm = F.normalize(features, dim=-1)
        prototypes_norm = F.normalize(prototypes, dim=-1)
        
        similarities = torch.mm(features_norm, prototypes_norm.t())  # [N, 3]
        scores, labels = similarities.max(dim=1)
        
        # Filter by confidence
        keep = scores > conf_threshold
        if keep.sum() == 0:
            continue
        
        valid_boxes_tensor = torch.stack(valid_boxes)
        all_boxes.append(valid_boxes_tensor[keep])
        all_labels.append(labels[keep])
        all_scores.append(scores[keep])
    
    if len(all_boxes) == 0:
        return [], [], []
    
    # Concatenate
    final_boxes = torch.cat(all_boxes)
    final_labels = torch.cat(all_labels)
    final_scores = torch.cat(all_scores)
    
    # Step 4: Apply NMS per class
    keep_all = []
    for class_id in range(3):
        mask = final_labels == class_id
        if mask.sum() == 0:
            continue
        
        class_boxes = final_boxes[mask]
        class_scores = final_scores[mask]
        
        keep = nms(class_boxes, class_scores, nms_threshold)
        keep_indices = torch.where(mask)[0][keep]
        keep_all.extend(keep_indices.tolist())
    
    if len(keep_all) == 0:
        return [], [], []
    
    keep_all = torch.tensor(keep_all)
    
    return (final_boxes[keep_all].cpu().numpy().tolist(),
            final_labels[keep_all].cpu().numpy().tolist(),
            final_scores[keep_all].cpu().numpy().tolist())


def main():
    print("="*80)
    print("VALIDATION SET EVALUATION")
    print("="*80)
    
    # Paths
    data_dir = Path('data/valid')
    output_dir = Path('results_validation')
    output_dir.mkdir(exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load data
    print(f"\nLoading annotations from {data_dir}...")
    annotations = pd.read_csv(data_dir / '_annotations.csv')
    image_files = sorted(annotations['filename'].unique())
    print(f"Found {len(image_files)} images")
    
    # Load models
    print("\nLoading models...")
    
    # 1. COCO RPN
    print("  1/3 Loading COCO RPN...")
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    rpn_model = fasterrcnn_resnet50_fpn(weights=weights, box_score_thresh=0.05)
    rpn_model.to(device)
    rpn_model.eval()
    
    # 2. DINOv2
    print("  2/3 Loading DINOv2...")
    dinov2_model = load_dinov2()
    dinov2_model.to(device)
    dinov2_model.eval()
    
    # 3. Prototypes
    print("  3/3 Loading prototypes...")
    checkpoint = torch.load('saved_models/prototypes.pth', 
                           map_location=device, weights_only=True)
    
    if isinstance(checkpoint, dict) and 'prototypes' in checkpoint:
        prototypes = checkpoint['prototypes']
    else:
        prototypes = checkpoint
    
    print(f"      Prototypes shape: {prototypes.shape}")
    prototypes = prototypes.to(device)
    
    # Class names
    class_names = ['airplane', 'baseball diamond', 'tennis court']
    
    # Run inference
    print(f"\n{'='*80}")
    print(f"Running inference on {len(image_files)} images...")
    print(f"{'='*80}\n")
    
    predictions = []
    
    for img_name in tqdm(image_files, desc="Processing"):
        img_path = data_dir / img_name
        image = Image.open(img_path).convert('RGB')
        
        boxes, labels, scores = run_detection(
            image, rpn_model, dinov2_model, prototypes, device,
            conf_threshold=0.05, nms_threshold=0.5
        )
        
        predictions.append({
            'image_name': img_name,
            'boxes': boxes,
            'labels': labels,
            'scores': scores
        })
    
    # Save predictions
    pred_file = output_dir / 'predictions_valid.json'
    with open(pred_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    print(f"\n✓ Predictions saved to {pred_file}")
    
    # Evaluate mAP@50
    print(f"\n{'='*80}")
    print("EVALUATION - mAP@50")
    print(f"{'='*80}")
    
    metrics_50 = evaluate_detections(
        predictions_file=str(pred_file),
        ground_truth_csv=str(data_dir / '_annotations.csv'),
        class_names=class_names,
        iou_threshold=0.5
    )
    
    print_evaluation_report(metrics_50)
    save_evaluation_results(metrics_50, str(output_dir / 'metrics_map50.json'))
    
    # Evaluate mAP@75
    print(f"\n{'='*80}")
    print("EVALUATION - mAP@75")
    print(f"{'='*80}\n")
    
    metrics_75 = evaluate_detections(
        predictions_file=str(pred_file),
        ground_truth_csv=str(data_dir / '_annotations.csv'),
        class_names=class_names,
        iou_threshold=0.75
    )
    
    print(f"mAP@75: {metrics_75['mAP@50']:.4f} ({metrics_75['mAP@50']*100:.2f}%)")
    save_evaluation_results(metrics_75, str(output_dir / 'metrics_map75.json'))
    
    # Final summary
    print(f"\n{'='*80}")
    print("VALIDATION RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"Images:    {len(image_files)}")
    print(f"mAP@50:    {metrics_50['mAP@50']:.1%}")
    print(f"mAP@75:    {metrics_75['mAP@50']:.1%}")
    print(f"Precision: {metrics_50['mean_precision']:.1%}")
    print(f"Recall:    {metrics_50['mean_recall']:.1%}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
