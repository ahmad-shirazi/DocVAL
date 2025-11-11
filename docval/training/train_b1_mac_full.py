#!/usr/bin/env python3
"""
Phase B1: Full Fine-Tuning on Mac M4 Max
Optimized for Apple Silicon with MPS backend

Hardware: Mac M4 Max, 64GB Unified Memory
Strategy: Full fine-tuning (no LoRA) with aggressive memory optimization
"""

import os
import json
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from PIL import Image
import numpy as np
from dotenv import load_dotenv

from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    TrainingArguments,
    Trainer,
)
from transformers.trainer_callback import TrainerCallback

# Load environment variables (including HF_TOKEN)
load_dotenv()


@dataclass
class MacB1Config:
    """Configuration optimized for Mac M4 Max full fine-tuning"""
    # Model
    model_name: str = "google/gemma-3-12b-it"
    
    # Data
    train_data_path: str = "docval/data/phase_a_output/filtered/D3_train.json"
    val_data_path: str = "docval/data/phase_a_output/filtered/D4_val.json"
    image_base_dir: str = "docval/data/cot_data"
    max_length: int = 4096  # Reduced for Mac memory
    
    # Training - Optimized for Mac
    output_dir: str = "docval/models/student_b1_mac"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1  # Must be 1 for Mac
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8  # Effective batch = 8
    learning_rate: float = 1e-5  # Lower for full fine-tuning
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    
    # Optimization
    optim: str = "adamw_torch"  # Use PyTorch AdamW (MPS compatible)
    lr_scheduler_type: str = "cosine"
    
    # Mac-specific: No FP16/BF16 support on MPS yet
    fp16: bool = False
    bf16: bool = False
    
    # Logging & Saving
    logging_steps: int = 10
    eval_steps: int = 200  # Less frequent for speed
    save_steps: int = 200
    save_total_limit: int = 2  # Keep only 2 checkpoints to save space
    evaluation_strategy: str = "steps"
    save_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    
    # Reporting
    report_to: str = "none"
    run_name: str = "gemma3-12b-docval-mac-full"
    
    # Hardware - Mac specific
    dataloader_num_workers: int = 0  # Must be 0 on Mac for MPS
    dataloader_pin_memory: bool = False  # Not needed on Mac
    use_mps_device: bool = True
    
    # Memory optimization for Mac
    gradient_checkpointing: bool = True
    max_train_samples: Optional[int] = None  # Set to limit dataset size for testing
    max_val_samples: Optional[int] = None


class DocVALDataset(torch.utils.data.Dataset):
    """Dataset for DocVAL Phase B1 training"""
    
    def __init__(self, data_path: str, image_base_dir: str, processor, 
                 max_length: int = 4096, max_samples: Optional[int] = None):
        self.processor = processor
        self.max_length = max_length
        self.image_base_dir = Path(image_base_dir)
        
        # Load training data
        print(f"Loading data from {data_path}...")
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        # Limit samples if specified
        if max_samples:
            self.data = self.data[:max_samples]
            print(f"Limited to {max_samples} samples")
        
        print(f"Loaded {len(self.data)} examples")
        
        # Filter out examples with missing images
        self.data = [ex for ex in self.data if self._get_image_path(ex).exists()]
        print(f"Filtered to {len(self.data)} examples with valid images")
    
    def _get_image_path(self, example: Dict) -> Path:
        """Get full image path from example"""
        dataset = example['image_id'].split('_')[0]
        
        # Dataset-specific paths
        if dataset == 'cord':
            return self.image_base_dir / 'CORD' / example['image_file']
        elif dataset == 'docvqa':
            return self.image_base_dir / 'DocVQA' / 'images' / example['image_file']
        elif dataset == 'funsd':
            return self.image_base_dir / 'FUNSD' / example['image_file']
        elif dataset == 'sroie':
            return self.image_base_dir / 'SROIE' / example['image_file']
        elif dataset == 'visualmrc':
            return self.image_base_dir / 'VisualMRC' / example['image_file']
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
    
    def _format_cot_output(self, example: Dict) -> str:
        """Format the expected output with 7-step CoT"""
        cot_steps = example.get('cot_steps', [])
        answer = example.get('answer_pred', '')
        bbox = example.get('bbox_pred', [])
        
        # Format CoT steps
        reasoning = "REASONING:\n"
        for i, step in enumerate(cot_steps, 1):
            reasoning += f"Step {i}: {step}\n"
        
        # Format answer and bbox
        output = f"{reasoning}\nANSWER: {answer}\nBBOX: {bbox}"
        return output
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        
        # Load image with Mac-compatible PIL
        image_path = self._get_image_path(example)
        try:
            image = Image.open(image_path).convert('RGB')
            # Resize large images to save memory
            if max(image.size) > 1024:
                image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image = Image.new('RGB', (224, 224), color='white')
        
        # Format input message
        question = example['question']
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            }
        ]
        
        # Format expected output (teacher's CoT)
        target_text = self._format_cot_output(example)
        
        # Process with chat template
        try:
            inputs = self.processor.apply_chat_template(
                messages,
                images=[image],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        except Exception as e:
            print(f"Error processing example {idx}: {e}")
            # Return dummy data if processing fails
            return {
                'input_ids': torch.zeros(self.max_length, dtype=torch.long),
                'labels': torch.full((self.max_length,), -100, dtype=torch.long),
                'pixel_values': torch.zeros(3, 224, 224),
                'attention_mask': torch.ones(self.max_length, dtype=torch.long)
            }
        
        # Add target text
        target_ids = self.processor.tokenizer(
            target_text,
            max_length=self.max_length // 2,  # Leave room for input
            truncation=True,
            padding=False,
            return_tensors="pt"
        ).input_ids
        
        # Concatenate input and target
        input_ids = inputs['input_ids'].squeeze(0)
        labels = torch.cat([
            torch.full_like(input_ids, -100),  # Ignore input tokens in loss
            target_ids.squeeze(0)
        ], dim=0)
        
        # Truncate if needed
        if labels.shape[0] > self.max_length:
            labels = labels[:self.max_length]
            input_ids = input_ids[:self.max_length - target_ids.shape[1]]
        
        # Concatenate for model input
        full_input_ids = torch.cat([input_ids, target_ids.squeeze(0)], dim=0)
        
        return {
            'input_ids': full_input_ids,
            'labels': labels,
            'pixel_values': inputs.get('pixel_values', torch.zeros(1, 3, 224, 224)).squeeze(0),
            'attention_mask': torch.ones_like(full_input_ids)
        }


class DocVALDataCollator:
    """Custom data collator for DocVAL training"""
    
    def __init__(self, processor):
        self.processor = processor
        self.pad_token_id = processor.tokenizer.pad_token_id
    
    def __call__(self, features):
        # Get max length in batch
        max_length = max(f['input_ids'].shape[0] for f in features)
        
        batch = {
            'input_ids': [],
            'labels': [],
            'pixel_values': [],
            'attention_mask': []
        }
        
        for feature in features:
            input_ids = feature['input_ids']
            labels = feature['labels']
            
            # Pad
            padding_length = max_length - input_ids.shape[0]
            if padding_length > 0:
                input_ids = torch.cat([
                    input_ids,
                    torch.full((padding_length,), self.pad_token_id, dtype=input_ids.dtype)
                ])
                labels = torch.cat([
                    labels,
                    torch.full((padding_length,), -100, dtype=labels.dtype)
                ])
                attention_mask = torch.cat([
                    feature['attention_mask'],
                    torch.zeros(padding_length, dtype=feature['attention_mask'].dtype)
                ])
            else:
                attention_mask = feature['attention_mask']
            
            batch['input_ids'].append(input_ids)
            batch['labels'].append(labels)
            batch['pixel_values'].append(feature['pixel_values'])
            batch['attention_mask'].append(attention_mask)
        
        return {
            'input_ids': torch.stack(batch['input_ids']),
            'labels': torch.stack(batch['labels']),
            'pixel_values': torch.stack(batch['pixel_values']),
            'attention_mask': torch.stack(batch['attention_mask'])
        }


class MacProgressCallback(TrainerCallback):
    """Custom callback for Mac training with progress updates"""
    
    def __init__(self):
        self.start_time = None
        
    def on_train_begin(self, args, state, control, **kwargs):
        import time
        self.start_time = time.time()
        print("\n" + "="*80)
        print("🍎 TRAINING STARTED ON MAC M4 MAX")
        print("="*80)
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            step = state.global_step
            if 'loss' in logs:
                print(f"Step {step}: loss = {logs['loss']:.4f}")
            if 'eval_loss' in logs:
                print(f"Step {step}: eval_loss = {logs['eval_loss']:.4f}")
                
                # Estimate remaining time
                if self.start_time:
                    import time
                    elapsed = time.time() - self.start_time
                    steps_done = state.global_step
                    total_steps = state.max_steps
                    if steps_done > 0:
                        time_per_step = elapsed / steps_done
                        remaining_steps = total_steps - steps_done
                        eta_seconds = time_per_step * remaining_steps
                        eta_hours = eta_seconds / 3600
                        print(f"  Progress: {steps_done}/{total_steps} steps ({steps_done/total_steps*100:.1f}%)")
                        print(f"  ETA: {eta_hours:.1f} hours")


def train_phase_b1_mac(config: MacB1Config):
    """Main training function for Phase B1 on Mac"""
    
    print("\n" + "="*80)
    print("PHASE B1: FULL FINE-TUNING ON MAC M4 MAX")
    print("="*80)
    print(f"\n🍎 Device: Mac M4 Max (64GB Unified Memory)")
    print(f"🧠 Model: {config.model_name}")
    print(f"⚡ Backend: MPS (Metal Performance Shaders)")
    print(f"🎯 Strategy: Full Fine-Tuning (all parameters)")
    print(f"\nTrain data: {config.train_data_path}")
    print(f"Val data: {config.val_data_path}")
    print(f"Output: {config.output_dir}\n")
    
    # Setup output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check MPS availability
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✓ MPS backend available")
    else:
        device = torch.device("cpu")
        print("⚠ MPS not available, using CPU (very slow!)")
    
    # Load model and processor
    print(f"\n{'='*80}")
    print("LOADING MODEL")
    print(f"{'='*80}\n")
    
    # Get HuggingFace token from environment
    hf_token = os.getenv('HF_TOKEN')
    if hf_token:
        print("✓ Using HuggingFace token from .env file")
    else:
        print("⚠️  No HF_TOKEN found in .env - using cached credentials")
    
    print(f"Loading processor: {config.model_name}")
    processor = AutoProcessor.from_pretrained(
        config.model_name, 
        trust_remote_code=True,
        token=hf_token
    )
    
    print(f"Loading model: {config.model_name}")
    print("⚠️  This will download ~24GB - first time only")
    print("⏳ Please wait, this may take 10-15 minutes...\n")
    
    model = AutoModelForImageTextToText.from_pretrained(
        config.model_name,
        torch_dtype=torch.float32,  # MPS doesn't support FP16/BF16 well yet
        low_cpu_mem_usage=True,
        token=hf_token
    )
    
    # Move to MPS device
    model = model.to(device)
    
    # Enable gradient checkpointing for memory efficiency
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("✓ Gradient checkpointing enabled")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 Model parameters:")
    print(f"  Total: {total_params:,} ({total_params/1e9:.1f}B)")
    print(f"  Trainable: {trainable_params:,} ({trainable_params/1e9:.1f}B)")
    print(f"  Training: 100% of parameters (full fine-tuning)")
    
    # Create datasets
    print(f"\n{'='*80}")
    print("LOADING DATASETS")
    print(f"{'='*80}\n")
    
    train_dataset = DocVALDataset(
        config.train_data_path,
        config.image_base_dir,
        processor,
        config.max_length,
        config.max_train_samples
    )
    
    val_dataset = DocVALDataset(
        config.val_data_path,
        config.image_base_dir,
        processor,
        config.max_length,
        config.max_val_samples
    )
    
    print(f"\n✓ Train examples: {len(train_dataset)}")
    print(f"✓ Val examples: {len(val_dataset)}")
    
    # Data collator
    data_collator = DocVALDataCollator(processor)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        max_grad_norm=config.max_grad_norm,
        optim=config.optim,
        lr_scheduler_type=config.lr_scheduler_type,
        fp16=config.fp16,
        bf16=config.bf16,
        logging_steps=config.logging_steps,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        evaluation_strategy=config.evaluation_strategy,
        save_strategy=config.save_strategy,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        report_to=config.report_to,
        run_name=config.run_name,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=config.dataloader_pin_memory,
        remove_unused_columns=False,
        gradient_checkpointing=config.gradient_checkpointing,
        use_mps_device=config.use_mps_device,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[MacProgressCallback()],
    )
    
    # Train
    print(f"\n{'='*80}")
    print("TRAINING")
    print(f"{'='*80}\n")
    print(f"Epochs: {config.num_train_epochs}")
    print(f"Batch size: {config.per_device_train_batch_size}")
    print(f"Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"Effective batch size: {config.per_device_train_batch_size * config.gradient_accumulation_steps}")
    print(f"Learning rate: {config.learning_rate}")
    total_steps = len(train_dataset) * config.num_train_epochs // (config.per_device_train_batch_size * config.gradient_accumulation_steps)
    print(f"Total steps: ~{total_steps}")
    print(f"\n⏱️  Estimated time: {total_steps * 30 / 3600:.1f} hours (at ~30 sec/step)")
    print(f"💾 Memory usage: ~50-60GB unified memory")
    print("\n⚠️  IMPORTANT:")
    print("  • Keep your Mac plugged in")
    print("  • Close other memory-intensive apps")
    print("  • Training will take 24-48 hours")
    print("  • Checkpoints saved every 200 steps")
    print()
    
    trainer.train()
    
    # Save final model
    print(f"\n{'='*80}")
    print("SAVING MODEL")
    print(f"{'='*80}\n")
    
    trainer.save_model(output_dir / "final")
    processor.save_pretrained(output_dir / "final")
    
    print(f"✓ Model saved to {output_dir / 'final'}")
    
    # Save training config
    with open(output_dir / "training_config.json", 'w') as f:
        json.dump(config.__dict__, f, indent=2)
    
    print(f"✓ Config saved to {output_dir / 'training_config.json'}")
    
    print(f"\n{'='*80}")
    print("✓ PHASE B1 TRAINING COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Phase B1 on Mac M4 Max (Full Fine-Tuning)")
    parser.add_argument("--test-run", action="store_true", help="Run with small dataset for testing")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--output-dir", type=str, default="docval/models/student_b1_mac")
    
    args = parser.parse_args()
    
    # Create config
    config = MacB1Config(
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        output_dir=args.output_dir,
    )
    
    # Test run with limited data
    if args.test_run:
        print("\n⚠️  TEST RUN MODE: Using only 50 training / 10 validation examples")
        config.max_train_samples = 50
        config.max_val_samples = 10
        config.num_train_epochs = 1
        config.eval_steps = 10
        config.save_steps = 10
    
    # Train
    train_phase_b1_mac(config)

