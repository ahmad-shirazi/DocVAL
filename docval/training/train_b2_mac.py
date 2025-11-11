#!/usr/bin/env python3
"""
Phase B2: VAL Feedback Training on Mac
Assumes Phase B1 is complete

Iterative refinement:
1. Generate outputs on validation set
2. Verify with VAL (text detector + rules)
3. Collect incorrect examples
4. Retrain on corrections with feedback
5. Repeat until convergence
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from docval.training.train_phase_b2 import train_phase_b2, B2Config


def main():
    print("\n" + "="*80)
    print("PHASE B2: VAL FEEDBACK TRAINING (MAC)")
    print("="*80)
    print("\n📋 Prerequisites:")
    print("  ✓ Phase B1 must be complete")
    print("  ✓ Student model at: docval/models/student_b1_mac/final")
    print("  ✓ Text detector (DB-ResNet) will be used for validation")
    
    print("\n🔄 Process:")
    print("  1. Load fine-tuned student from Phase B1")
    print("  2. Generate answers on validation set")
    print("  3. Verify with VAL (detect text → check IoU → verify reasoning)")
    print("  4. Collect errors and generate corrections")
    print("  5. Train on corrections")
    print("  6. Repeat until convergence")
    
    print("\n⏱️  Expected time:")
    print("  • ~2-4 hours per iteration")
    print("  • ~3-5 iterations typical")
    print("  • Total: 6-20 hours")
    
    print("\n" + "="*80)
    
    # Check if Phase B1 model exists
    b1_model_path = Path("docval/models/student_b1_mac/final")
    if not b1_model_path.exists():
        print("\n❌ ERROR: Phase B1 model not found!")
        print(f"   Expected at: {b1_model_path}")
        print("\n   Options:")
        print("   1. Complete Phase B1 first: python train_b1_mac_full.py")
        print("   2. Use pre-trained model (if available)")
        print("   3. Change path in config")
        return
    
    print(f"\n✓ Found Phase B1 model at: {b1_model_path}")
    
    response = input("\nReady to start Phase B2 iterative training? (y/n): ")
    if response.lower() != 'y':
        print("Training cancelled.")
        return
    
    # Create config for Mac
    config = B2Config(
        student_model_path=str(b1_model_path),
        val_data_path="docval/data/phase_a_output/filtered/D4_val.json",
        image_base_dir="docval/data/cot_data",
        output_dir="docval/models/student_b2_mac",
        
        # VAL settings
        text_detector_name="db_resnet",
        iou_threshold=0.5,
        
        # Iterative training
        max_iterations=5,
        convergence_threshold=0.05,  # Stop if < 5% improvement
        correction_dataset_size=200,
        
        # Training per iteration
        num_epochs_per_iter=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,  # Lower LR for refinement
        
        # Hardware
        use_mps_device=True,
        max_length=4096,
    )
    
    # Start training
    train_phase_b2(config)


if __name__ == "__main__":
    main()

