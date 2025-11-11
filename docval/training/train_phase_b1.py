"""
Phase B1: CoT Training
Fine-tune Gemma 3-12B student model on teacher-generated CoT data

Input: D3_train.json (1,654 examples)
Validation: D4_val.json (354 examples)
Output: Fine-tuned student model with 7-step CoT reasoning
"""

import os
import json
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from PIL import Image
import numpy as np

from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from transformers.trainer_callback import TrainerCallback
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import wandb


@dataclass
class B1TrainingConfig:
    """Configuration for Phase B1 training"""
    # Model
    model_name: str = "google/gemma-3-12b-it"
    use_8bit: bool = False  # Use 8-bit quantization for memory efficiency
    use_lora: bool = True  # Use LoRA for parameter-efficient fine-tuning
    
    # LoRA config
    lora_r: int = 32  # LoRA rank
    lora_alpha: int = 64  # LoRA alpha
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Data
    train_data_path: str = "docval/data/phase_a_output/filtered/D3_train.json"
    val_data_path: str = "docval/data/phase_a_output/filtered/D4_val.json"
    image_base_dir: str = "docval/data/cot_data"
    max_length: int = 8192
    
    # Training
    output_dir: str = "docval/models/student_b1"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16  # Effective batch size = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    
    # Optimization
    optim: str = "adamw_torch"
    lr_scheduler_type: str = "cosine"
    fp16: bool = False
    bf16: bool = True  # Use bfloat16 if available
    
    # Logging & Saving
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 3
    evaluation_strategy: str = "steps"
    save_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    
    # Reporting
    report_to: str = "none"  # "wandb" to enable W&B logging
    run_name: str = "gemma3-12b-docval-b1"
    
    # Hardware
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    
    # Resume
    resume_from_checkpoint: Optional[str] = None


class DocVALDataset(torch.utils.data.Dataset):
    """Dataset for DocVAL Phase B1 training"""
    
    def __init__(self, data_path: str, image_base_dir: str, processor, max_length: int = 8192):
        self.processor = processor
        self.max_length = max_length
        self.image_base_dir = Path(image_base_dir)
        
        # Load training data
        print(f"Loading data from {data_path}...")
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
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
        
        # Load image
        image_path = self._get_image_path(example)
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a blank image if loading fails
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
        inputs = self.processor.apply_chat_template(
            messages,
            images=[image],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        
        # Add target text
        target_ids = self.processor.tokenizer(
            target_text,
            max_length=self.max_length,
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


class LoggingCallback(TrainerCallback):
    """Custom callback for logging training progress"""
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            step = state.global_step
            if 'loss' in logs:
                print(f"Step {step}: loss = {logs['loss']:.4f}")
            if 'eval_loss' in logs:
                print(f"Step {step}: eval_loss = {logs['eval_loss']:.4f}")


def setup_model(config: B1TrainingConfig):
    """Setup model with optional quantization and LoRA"""
    print(f"\n{'='*80}")
    print("LOADING MODEL")
    print(f"{'='*80}\n")
    
    # Quantization config (optional)
    quantization_config = None
    if config.use_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )
        print("✓ Using 8-bit quantization")
    
    # Load processor and model
    print(f"Loading processor: {config.model_name}")
    processor = AutoProcessor.from_pretrained(config.model_name)
    
    print(f"Loading model: {config.model_name}")
    model = AutoModelForImageTextToText.from_pretrained(
        config.model_name,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
        device_map="auto",
    )
    
    # Prepare for k-bit training if using quantization
    if config.use_8bit:
        model = prepare_model_for_kbit_training(model)
    
    # Apply LoRA if enabled
    if config.use_lora:
        print("\n✓ Applying LoRA configuration:")
        print(f"  Rank: {config.lora_r}")
        print(f"  Alpha: {config.lora_alpha}")
        print(f"  Dropout: {config.lora_dropout}")
        
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return processor, model


def train_phase_b1(config: B1TrainingConfig):
    """Main training function for Phase B1"""
    
    print("\n" + "="*80)
    print("PHASE B1: CoT TRAINING")
    print("="*80)
    print(f"\nModel: {config.model_name}")
    print(f"Train data: {config.train_data_path}")
    print(f"Val data: {config.val_data_path}")
    print(f"Output: {config.output_dir}\n")
    
    # Setup output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize W&B if enabled
    if config.report_to == "wandb":
        wandb.init(
            project="docval-phase-b1",
            name=config.run_name,
            config=config.__dict__
        )
    
    # Load model and processor
    processor, model = setup_model(config)
    
    # Create datasets
    print(f"\n{'='*80}")
    print("LOADING DATASETS")
    print(f"{'='*80}\n")
    
    train_dataset = DocVALDataset(
        config.train_data_path,
        config.image_base_dir,
        processor,
        config.max_length
    )
    
    val_dataset = DocVALDataset(
        config.val_data_path,
        config.image_base_dir,
        processor,
        config.max_length
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
        gradient_checkpointing=True,  # Save memory
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[LoggingCallback()],
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
    print(f"Total steps: ~{len(train_dataset) * config.num_train_epochs // (config.per_device_train_batch_size * config.gradient_accumulation_steps)}")
    print()
    
    if config.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    else:
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
    
    parser = argparse.ArgumentParser(description="Train Phase B1: CoT Training")
    parser.add_argument("--model", type=str, default="google/gemma-3-12b-it")
    parser.add_argument("--train-data", type=str, default="docval/data/phase_a_output/filtered/D3_train.json")
    parser.add_argument("--val-data", type=str, default="docval/data/phase_a_output/filtered/D4_val.json")
    parser.add_argument("--image-dir", type=str, default="docval/data/cot_data")
    parser.add_argument("--output-dir", type=str, default="docval/models/student_b1")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--use-lora", action="store_true", default=True)
    parser.add_argument("--use-8bit", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    
    args = parser.parse_args()
    
    # Create config
    config = B1TrainingConfig(
        model_name=args.model,
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        image_base_dir=args.image_dir,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        use_lora=args.use_lora,
        use_8bit=args.use_8bit,
        report_to="wandb" if args.wandb else "none",
        resume_from_checkpoint=args.resume,
    )
    
    # Train
    train_phase_b1(config)

