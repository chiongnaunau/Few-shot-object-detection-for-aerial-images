"""
Run Complete Few-Shot Object Detection Pipeline

Usage:
    python run_pipeline.py [--k_shot K] [--device DEVICE]

Example:
    python run_pipeline.py --k_shot 20 --device cuda
"""

import argparse
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"\nError: {description} failed with exit code {result.returncode}")
        sys.exit(1)
    
    print(f"\n✓ {description} completed successfully")

def main():
    parser = argparse.ArgumentParser(description='Run Few-Shot Object Detection Pipeline')
    parser.add_argument('--k_shot', type=int, default=20,
                       help='Number of shots for few-shot learning (default: 20)')
    parser.add_argument('--proto_epochs', type=int, default=50,
                       help='Prototype training epochs (default: 50)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use: cuda or cpu (default: cuda)')
    parser.add_argument('--skip_analysis', action='store_true',
                       help='Skip dataset analysis step')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training step (use existing model)')
    parser.add_argument('--confidence', type=float, default=0.3,
                       help='Confidence threshold for inference (default: 0.3)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("FEW-SHOT OBJECT DETECTION PIPELINE")
    print("="*80)
    print(f"Configuration:")
    print(f"  K-shot: {args.k_shot}")
    print(f"  Prototype epochs: {args.proto_epochs}")
    print(f"  Device: {args.device}")
    print(f"  Confidence threshold: {args.confidence}")
    
    # Step 1: Dataset Analysis
    if not args.skip_analysis:
        run_command(
            [sys.executable, 'src/data_analysis.py'],
            "Step 1: Dataset Analysis"
        )
    else:
        print("\nSkipping dataset analysis...")
    
    # Step 2: Training
    if not args.skip_training:
        run_command(
            [sys.executable, 'src/training.py',
             '--k_shot', str(args.k_shot),
             '--proto_epochs', str(args.proto_epochs),
             '--device', args.device],
            "Step 2: Prototype Training"
        )
    else:
        print("\nSkipping training (using existing model)...")
    
    # Step 3: Inference and Evaluation
    run_command(
        [sys.executable, 'src/inference.py',
         '--multi_scale',
         '--confidence', str(args.confidence)],
        "Step 3: Inference and Evaluation"
    )
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80)
    print("\nOutputs:")
    print("  - Dataset analysis: analysis_results/")
    print("  - Trained model: saved_model/improved_prototypes.pth")
    print("  - Evaluation results: results/evaluation_report.json")
    print("  - Visualizations: results/")

if __name__ == '__main__':
    main()
