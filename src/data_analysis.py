"""
Dataset Analysis Script for Few-Shot Object Detection
Analyzes class distribution, image statistics, and prepares data splits
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def analyze_annotations(csv_path, split_name):
    """Analyze annotation CSV file"""
    df = pd.read_csv(csv_path)
    
    stats = {
        'split': split_name,
        'num_images': df['filename'].nunique(),
        'num_annotations': len(df),
        'num_classes': df['class'].nunique(),
        'classes': sorted(df['class'].unique().tolist()),
        'annotations_per_class': df['class'].value_counts().to_dict(),
        'images_per_class': df.groupby('class')['filename'].nunique().to_dict(),
        'avg_objects_per_image': len(df) / df['filename'].nunique(),
        'image_sizes': df.groupby('filename')[['width', 'height']].first().describe().to_dict()
    }
    
    # Calculate bbox sizes
    df['bbox_width'] = df['xmax'] - df['xmin']
    df['bbox_height'] = df['ymax'] - df['ymin']
    df['bbox_area'] = df['bbox_width'] * df['bbox_height']
    
    stats['bbox_stats'] = {
        'avg_width': df['bbox_width'].mean(),
        'avg_height': df['bbox_height'].mean(),
        'avg_area': df['bbox_area'].mean(),
        'min_area': df['bbox_area'].min(),
        'max_area': df['bbox_area'].max()
    }
    
    return stats, df

def visualize_class_distribution(stats_dict, output_dir):
    """Create visualization of class distribution across splits"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    splits = ['train', 'valid', 'test']
    
    # 1. Annotations per class
    ax = axes[0, 0]
    class_names = sorted(set().union(*[set(stats_dict[s]['classes']) for s in splits]))
    x = np.arange(len(class_names))
    width = 0.25
    
    for i, split in enumerate(splits):
        counts = [stats_dict[split]['annotations_per_class'].get(cls, 0) for cls in class_names]
        ax.bar(x + i*width, counts, width, label=split.capitalize())
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Number of Annotations', fontsize=12)
    ax.set_title('Annotations per Class by Split', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Images per class
    ax = axes[0, 1]
    for i, split in enumerate(splits):
        counts = [stats_dict[split]['images_per_class'].get(cls, 0) for cls in class_names]
        ax.bar(x + i*width, counts, width, label=split.capitalize())
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Number of Images', fontsize=12)
    ax.set_title('Images per Class by Split', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Dataset split sizes
    ax = axes[1, 0]
    split_sizes = [stats_dict[s]['num_images'] for s in splits]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    ax.bar(splits, split_sizes, color=colors)
    ax.set_xlabel('Split', fontsize=12)
    ax.set_ylabel('Number of Images', fontsize=12)
    ax.set_title('Dataset Split Sizes', fontsize=14, fontweight='bold')
    for i, v in enumerate(split_sizes):
        ax.text(i, v + 5, str(v), ha='center', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 4. Objects per image
    ax = axes[1, 1]
    avg_objs = [stats_dict[s]['avg_objects_per_image'] for s in splits]
    ax.bar(splits, avg_objs, color=colors)
    ax.set_xlabel('Split', fontsize=12)
    ax.set_ylabel('Avg Objects per Image', fontsize=12)
    ax.set_title('Average Objects per Image by Split', fontsize=14, fontweight='bold')
    for i, v in enumerate(avg_objs):
        ax.text(i, v + 0.1, f'{v:.2f}', ha='center', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'dataset_distribution.png', dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {output_dir / 'dataset_distribution.png'}")
    plt.close()

def define_base_novel_split(class_names):
    """Define base and novel classes for few-shot learning"""
    # Following NWPU VHR-10 standard split
    novel_classes = ['airplane', 'baseball diamond', 'tennis court']
    base_classes = [c for c in class_names if c not in novel_classes]
    
    return {
        'base_classes': base_classes,
        'novel_classes': novel_classes,
        'num_base': len(base_classes),
        'num_novel': len(novel_classes)
    }

def create_few_shot_support_set(df, novel_classes, k_shot=20, output_dir=None):
    """Create k-shot support set for novel classes"""
    support_set = {}
    
    for cls in novel_classes:
        cls_df = df[df['class'] == cls]
        # Get unique images with this class
        cls_images = cls_df['filename'].unique()
        
        if len(cls_images) >= k_shot:
            selected_images = np.random.choice(cls_images, k_shot, replace=False)
        else:
            print(f"Warning: Only {len(cls_images)} images available for class '{cls}', using all")
            selected_images = cls_images
        
        support_set[cls] = {
            'images': selected_images.tolist(),
            'num_images': len(selected_images),
            'annotations': cls_df[cls_df['filename'].isin(selected_images)].to_dict('records')
        }
    
    if output_dir:
        with open(output_dir / f'support_set_{k_shot}shot.json', 'w') as f:
            json.dump(support_set, f, indent=2)
        print(f"Saved {k_shot}-shot support set to {output_dir / f'support_set_{k_shot}shot.json'}")
    
    return support_set

def main():
    # Paths
    data_dir = Path(__file__).parent.parent / 'data'
    output_dir = Path(__file__).parent.parent / 'analysis_results'
    output_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("DATASET ANALYSIS FOR FEW-SHOT OBJECT DETECTION")
    print("="*80)
    
    # Analyze all splits
    stats_dict = {}
    dfs = {}
    
    for split in ['train', 'valid', 'test']:
        csv_path = data_dir / split / '_annotations.csv'
        if csv_path.exists():
            print(f"\n{'='*80}")
            print(f"Analyzing {split.upper()} split...")
            print(f"{'='*80}")
            
            stats, df = analyze_annotations(csv_path, split)
            stats_dict[split] = stats
            dfs[split] = df
            
            print(f"\n{split.upper()} Statistics:")
            print(f"  - Images: {stats['num_images']}")
            print(f"  - Annotations: {stats['num_annotations']}")
            print(f"  - Classes: {stats['num_classes']}")
            print(f"  - Avg objects/image: {stats['avg_objects_per_image']:.2f}")
            print(f"\n  Class distribution (annotations):")
            for cls, count in sorted(stats['annotations_per_class'].items()):
                img_count = stats['images_per_class'][cls]
                print(f"    {cls:20s}: {count:4d} annotations in {img_count:3d} images")
    
    # Save detailed statistics
    with open(output_dir / 'dataset_statistics.json', 'w') as f:
        json.dump(stats_dict, f, indent=2)
    print(f"\nSaved detailed statistics to {output_dir / 'dataset_statistics.json'}")
    
    # Define base/novel split
    all_classes = sorted(stats_dict['train']['classes'])
    split_info = define_base_novel_split(all_classes)
    
    print(f"\n{'='*80}")
    print("FEW-SHOT LEARNING SPLIT")
    print(f"{'='*80}")
    print(f"Base classes ({split_info['num_base']}): {', '.join(split_info['base_classes'])}")
    print(f"Novel classes ({split_info['num_novel']}): {', '.join(split_info['novel_classes'])}")
    
    # Save split info
    with open(output_dir / 'base_novel_split.json', 'w') as f:
        json.dump(split_info, f, indent=2)
    
    # Create few-shot support sets
    print(f"\n{'='*80}")
    print("CREATING FEW-SHOT SUPPORT SETS")
    print(f"{'='*80}")
    
    for k_shot in [5, 10, 20, 30]:
        print(f"\nCreating {k_shot}-shot support set...")
        support_set = create_few_shot_support_set(
            dfs['train'], 
            split_info['novel_classes'], 
            k_shot=k_shot,
            output_dir=output_dir
        )
        for cls, info in support_set.items():
            print(f"  {cls}: {info['num_images']} images, {len(info['annotations'])} annotations")
    
    # Create visualizations
    print(f"\n{'='*80}")
    print("CREATING VISUALIZATIONS")
    print(f"{'='*80}")
    visualize_class_distribution(stats_dict, output_dir)
    
    # Summary for report
    print(f"\n{'='*80}")
    print("SUMMARY FOR TECHNICAL REPORT")
    print(f"{'='*80}")
    print(f"""
Dataset: NWPU VHR-10 Aerial Object Detection
- Total images: {sum(s['num_images'] for s in stats_dict.values())}
  - Train: {stats_dict['train']['num_images']} images
  - Valid: {stats_dict['valid']['num_images']} images
  - Test: {stats_dict['test']['num_images']} images
  
- Total annotations: {sum(s['num_annotations'] for s in stats_dict.values())}
- Classes: {len(all_classes)} ({', '.join(all_classes)})

Few-Shot Setting:
- Base classes: {split_info['num_base']} classes for training detector
- Novel classes: {split_info['num_novel']} classes for few-shot learning
- Support set sizes: 5, 10, 20, 30-shot configurations available

Key Challenges:
- Small object detection in aerial imagery
- Class imbalance across splits
- Limited examples for novel classes (few-shot constraint)
    """)
    
    print(f"\n{'='*80}")
    print("Analysis complete! Results saved to:", output_dir)
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
