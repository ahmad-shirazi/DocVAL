# Phase B1: CoT Training Guide

## Overview

Phase B1 trains the **Gemma 3-12B student model** to mimic the teacher's 7-step Chain-of-Thought reasoning on document VQA tasks.

**Goal**: Student learns to generate CoT reasoning + answer + bbox from image + question alone.

## Prerequisites

### 1. Data Ready ✅
- `D3_train.json` - 1,654 training examples
- `D4_val.json` - 354 validation examples
- Images in `docval/data/cot_data/`

### 2. HuggingFace Access

Gemma models require authentication:

```bash
# Option 1: Login via CLI
huggingface-cli login

# Option 2: Set environment variable
export HF_TOKEN=<your_token>
```

Get your token from: https://huggingface.co/settings/tokens

**Note**: You may need to accept the Gemma license at https://huggingface.co/google/gemma-3-12b-it

### 3. Hardware Requirements

**Recommended**:
- GPU: A100 (80GB) or H100
- RAM: 64GB+
- Storage: 100GB+

**Minimum** (with 8-bit quantization):
- GPU: RTX 3090 (24GB) or A10
- RAM: 32GB+
- Storage: 50GB+

**Model Size**:
- Gemma 3-12B: ~24GB (FP32), ~12GB (FP16), ~6GB (8-bit)
- With LoRA: Only ~1-2% parameters are trainable

## Installation

### Install Dependencies

```bash
cd /Users/ahmadshirazi/Desktop/DocVAL
source envval/bin/activate

# Install training dependencies
pip install -r requirements.txt

# Install flash-attention for faster training (optional, recommended)
pip install flash-attn --no-build-isolation
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import peft; print(f'PEFT: {peft.__version__}')"
```

## Training Methods

### Method 1: Quick Start (Recommended)

```bash
cd /Users/ahmadshirazi/Desktop/DocVAL
source envval/bin/activate

# Login to HuggingFace
huggingface-cli login

# Start training with interactive setup
python start_phase_b1_training.py
```

This will:
1. Download Gemma 3-12B (auto-cached after first run)
2. Load filtered training data
3. Fine-tune with LoRA (parameter-efficient)
4. Save checkpoints every 100 steps
5. Validate every 100 steps

### Method 2: Command Line

```bash
python -m docval.training.train_phase_b1 \
    --model google/gemma-3-12b-it \
    --train-data docval/data/phase_a_output/filtered/D3_train.json \
    --val-data docval/data/phase_a_output/filtered/D4_val.json \
    --image-dir docval/data/cot_data \
    --output-dir docval/models/student_b1 \
    --epochs 3 \
    --batch-size 1 \
    --grad-accum 16 \
    --lr 2e-5 \
    --use-lora
```

### Method 3: Custom Configuration

```python
from docval.training.train_phase_b1 import train_phase_b1, B1TrainingConfig

config = B1TrainingConfig(
    model_name="google/gemma-3-12b-it",
    train_data_path="docval/data/phase_a_output/filtered/D3_train.json",
    val_data_path="docval/data/phase_a_output/filtered/D4_val.json",
    
    # Training
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,  # Effective batch = 16
    learning_rate=2e-5,
    
    # LoRA
    use_lora=True,
    lora_r=32,
    lora_alpha=64,
    
    # Memory
    use_8bit=False,  # Set True if OOM
    bf16=True,
    
    # Output
    output_dir="docval/models/student_b1",
)

train_phase_b1(config)
```

## Training Configuration

### Default Settings

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Model** | `google/gemma-3-12b-it` | Base model |
| **LoRA Rank** | 32 | Number of LoRA dimensions |
| **LoRA Alpha** | 64 | LoRA scaling factor |
| **Learning Rate** | 2e-5 | Initial LR |
| **Batch Size** | 1 | Per-device batch |
| **Grad Accumulation** | 16 | Effective batch = 16 |
| **Epochs** | 3 | Full passes through data |
| **Max Length** | 8192 | Max tokens (input + output) |
| **Precision** | BF16 | Mixed precision training |
| **Scheduler** | Cosine | LR schedule |
| **Warmup** | 10% | Warmup ratio |

### Memory Optimization

If you encounter OOM (Out of Memory) errors:

1. **Enable 8-bit Quantization**:
   ```bash
   python -m docval.training.train_phase_b1 --use-8bit
   ```

2. **Reduce Batch Size**:
   ```bash
   python -m docval.training.train_phase_b1 --batch-size 1 --grad-accum 32
   ```

3. **Reduce LoRA Rank**:
   ```python
   config.lora_r = 16  # Instead of 32
   ```

4. **Reduce Max Length**:
   ```python
   config.max_length = 4096  # Instead of 8192
   ```

## Expected Training Time

With **A100 (80GB)**:
- ~8-10 hours for 3 epochs
- ~300 steps per epoch
- ~10-15 min/100 steps

With **RTX 3090 (24GB)** + 8-bit:
- ~12-16 hours for 3 epochs
- Slower due to quantization overhead

## Monitoring Training

### Option 1: Terminal Output

Training prints progress every 10 steps:
```
Step 10: loss = 2.3456
Step 20: loss = 2.1234
Step 100: eval_loss = 1.9876
```

### Option 2: Weights & Biases

Enable W&B logging:
```bash
pip install wandb
wandb login

python -m docval.training.train_phase_b1 --wandb
```

View at: https://wandb.ai/your-username/docval-phase-b1

### Option 3: TensorBoard

```bash
# Install tensorboard
pip install tensorboard

# Run training with default logging
python -m docval.training.train_phase_b1

# View in another terminal
tensorboard --logdir docval/models/student_b1/runs
```

## Output Files

Training creates these files in `docval/models/student_b1/`:

```
student_b1/
├── checkpoint-100/          # Checkpoint at step 100
│   ├── adapter_config.json  # LoRA config
│   ├── adapter_model.bin    # LoRA weights
│   └── ...
├── checkpoint-200/          # Checkpoint at step 200
├── checkpoint-300/          # Checkpoint at step 300
├── final/                   # Final trained model
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   ├── config.json
│   ├── preprocessor_config.json
│   └── ...
└── training_config.json     # Training configuration
```

**Important**: Only LoRA adapters are saved (~50-100MB), not the full model (12GB).

## Resume Training

If training is interrupted:

```bash
python -m docval.training.train_phase_b1 \
    --resume docval/models/student_b1/checkpoint-200
```

## Validation During Training

Model is validated every 100 steps on `D4_val.json` (354 examples):
- Computes validation loss
- Saves checkpoint if best model
- Early stopping if no improvement

## Next Step: Inference

After training completes, test the model:

```python
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
from PIL import Image

# Load base model
base_model = AutoModelForImageTextToText.from_pretrained("google/gemma-3-12b-it")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "docval/models/student_b1/final")
processor = AutoProcessor.from_pretrained("docval/models/student_b1/final")

# Test
image = Image.open("path/to/document.png")
question = "What is the total amount?"

messages = [{
    "role": "user",
    "content": [
        {"type": "image"},
        {"type": "text", "text": question}
    ]
}]

inputs = processor.apply_chat_template(
    messages,
    images=[image],
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=1024)
response = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:])
print(response)
```

## Troubleshooting

### Issue: "Gemma model requires authentication"
**Solution**: Login to HuggingFace and accept the license
```bash
huggingface-cli login
# Visit: https://huggingface.co/google/gemma-3-12b-it
```

### Issue: CUDA Out of Memory
**Solutions**:
1. Enable 8-bit quantization: `--use-8bit`
2. Reduce batch size: `--batch-size 1 --grad-accum 32`
3. Reduce LoRA rank: Set `lora_r=16`
4. Use gradient checkpointing (enabled by default)

### Issue: Slow training
**Solutions**:
1. Install flash-attention: `pip install flash-attn --no-build-isolation`
2. Use multiple GPUs: `CUDA_VISIBLE_DEVICES=0,1 python ...`
3. Check dataloader workers: Increase `dataloader_num_workers`

### Issue: Model not learning (high loss)
**Checks**:
1. Verify data paths are correct
2. Check images are loading (not all blank/errors)
3. Increase learning rate: `--lr 5e-5`
4. Train for more epochs: `--epochs 5`

### Issue: Loss is NaN
**Solutions**:
1. Reduce learning rate: `--lr 1e-5`
2. Enable gradient clipping (enabled by default)
3. Check for corrupted images in dataset

## Training Tips

1. **Start Small**: Test with 1 epoch first to verify everything works
2. **Monitor Closely**: Watch first 100 steps for any issues
3. **Save Often**: Default saves every 100 steps (good for 24GB GPUs)
4. **Use LoRA**: Much faster than full fine-tuning, nearly same quality
5. **BF16 > FP16**: BFloat16 is more stable for mixed precision
6. **Gradient Accumulation**: Simulates larger batch without more memory
7. **Checkpointing**: Enabled by default, reduces memory by ~30%

## Expected Results

After 3 epochs of training:
- **Training Loss**: Should decrease to ~0.5-1.0
- **Validation Loss**: Should be ~0.8-1.5
- **Generation**: Model should produce structured CoT with 7 steps
- **Quality**: Answers should be grounded in document text

## Post-Training

Once training completes:

1. **Evaluate on Test Set** (Phase C):
   ```bash
   python -m docval.inference.evaluate \
       --model docval/models/student_b1/final \
       --test-data docval/data/phase_a_output/filtered/D_test.json
   ```

2. **Compare with Teacher**:
   - Student should achieve 70-80% of teacher performance
   - Faster inference (12B vs 235B parameters)
   - No text detector needed

3. **Proceed to Phase B2**:
   - Train with VAL feedback for bbox refinement
   - Use correction dataset for iterative improvement

## Advanced: Full Fine-Tuning

For maximum quality (requires more memory):

```python
config = B1TrainingConfig(
    use_lora=False,  # Disable LoRA
    learning_rate=1e-5,  # Lower LR for full fine-tuning
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,  # Larger effective batch
)
```

**Requirements**: 
- A100 80GB or multiple GPUs
- ~80GB GPU memory
- 2-3x longer training time

---

## Quick Reference

### Start Training
```bash
cd /Users/ahmadshirazi/Desktop/DocVAL
source envval/bin/activate
huggingface-cli login
python start_phase_b1_training.py
```

### Monitor Progress
```bash
# Watch output
tail -f docval/models/student_b1/training.log

# Or use W&B
python -m docval.training.train_phase_b1 --wandb
```

### Resume Training
```bash
python -m docval.training.train_phase_b1 \
    --resume docval/models/student_b1/checkpoint-<step>
```

---

**Ready to train?** Run `python start_phase_b1_training.py` to begin! 🚀

