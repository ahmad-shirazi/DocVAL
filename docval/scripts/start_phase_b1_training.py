#!/usr/bin/env python3
"""
Quick Start Script for Phase B1 Training
Downloads Gemma 3-12B and fine-tunes on filtered CoT data
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from docval.training.train_phase_b1 import train_phase_b1, B1TrainingConfig


def main():
    """Start Phase B1 training with recommended settings"""
    
    print("\n" + "="*80)
    print("PHASE B1 - CoT TRAINING SETUP")
    print("="*80)
    print("\nThis will:")
    print("  1. Download Gemma 3-12B from HuggingFace")
    print("  2. Fine-tune on 1,654 filtered CoT examples")
    print("  3. Validate on 354 examples")
    print("  4. Use LoRA for efficient training")
    print("\nRecommended settings:")
    print("  • LoRA rank: 32")
    print("  • Learning rate: 2e-5")
    print("  • Epochs: 3")
    print("  • Effective batch size: 16")
    print("  • Mixed precision: bfloat16")
    print("\n" + "="*80)
    
    # Check if HuggingFace token is needed
    print("\nNote: You may need to login to HuggingFace for gated models:")
    print("  Run: hf auth login")
    print("  Or set: export HF_TOKEN=<your_token>")
    print("\n" + "="*80)
    
    response = input("\nReady to start training? (y/n): ")
    if response.lower() != 'y':
        print("Training cancelled.")
        return
    
    # Create config with recommended settings
    config = B1TrainingConfig(
        model_name="google/gemma-3-12b-it",
        train_data_path="docval/data/phase_a_output/filtered/D3_train.json",
        val_data_path="docval/data/phase_a_output/filtered/D4_val.json",
        image_base_dir="docval/data/cot_data",
        output_dir="docval/models/student_b1",
        
        # Training settings
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-5,
        
        # LoRA settings
        use_lora=True,
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        
        # Memory optimization
        use_8bit=False,  # Set to True if running out of memory
        bf16=True,
        gradient_checkpointing=True,
        
        # Logging
        logging_steps=10,
        eval_steps=100,
        save_steps=100,
        
        # Callbacks
        report_to="none",  # Change to "wandb" for W&B logging
    )
    
    # Start training
    train_phase_b1(config)


if __name__ == "__main__":
    main()

