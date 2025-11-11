# Phase B2: VAL Feedback Iterative Training

## Overview

Phase B2 refines the student model from Phase B1 using **VAL (Visual Answer Localization) feedback** in an iterative loop.

### Key Concept

Instead of just learning from teacher examples, the student:
1. **Generates** answers on new examples
2. **Gets verified** by VAL (text detector + rules)
3. **Receives detailed feedback** on errors
4. **Learns from corrections** iteratively

This is like having a teacher that:
- ✅ Marks your homework
- ❌ Points out exactly what's wrong
- 📝 Shows you the correct answer
- 🔄 Makes you practice until you get it right

## Prerequisites

### 1. Phase B1 Complete ✅

You need the fine-tuned student model from Phase B1:
```
docval/models/student_b1_mac/final/
├── config.json
├── model.safetensors
├── preprocessor_config.json
└── ...
```

If you assumed Phase B1 is complete, you can create a dummy model or use a pre-trained one.

### 2. Data Ready ✅

- Validation set: `D4_val.json` (354 examples)
- Images in: `docval/data/cot_data/`

### 3. Text Detector ✅

DB-ResNet50 from Phase A (for validation)

### 4. Hardware Requirements

**Recommended (Full Fine-Tuning)**:
- GPU: **2x H100 80GB** (multi-GPU training)
- RAM: 72GB+
- Storage: 100GB+ (for models + corrections dataset)
- Batch Size: 128 (64 per GPU)
- Epochs per Iteration: 2
- Training Time: ~2-4 hours per iteration

**Alternative (LoRA Fine-Tuning)**:
- GPU: 1x A100 (40GB) or RTX 3090 (24GB)
- RAM: 32GB+
- Storage: 50GB+
- Batch Size: 16-32
- Training Time: ~1-2 hours per iteration

**Note**: Phase B2 iterative training is more memory-intensive due to dynamic correction dataset generation and VAL feedback integration.

## How It Works

### The Iterative Loop

```
┌─────────────────────────────────────────────────────────────┐
│ ITERATION 1                                                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Load Student (from B1)                                  │
│  2. Generate on D4_val (354 examples)                       │
│     ├─> Student predicts: CoT + Answer + BBox              │
│     └─> For each example                                    │
│                                                             │
│  3. VAL Verification                                        │
│     ├─> Detect text with DB-ResNet                         │
│     ├─> Check answer in regions? ✓ or ✗                    │
│     ├─> Compute IoU (predicted bbox vs regions)            │
│     │   ├─> IoU > 0.5? ✓ or ✗                              │
│     ├─> Check 7-step reasoning                             │
│     │   ├─> Each step complete? ✓ or ✗                     │
│     └─> Overall correct? ✓ or ✗                            │
│                                                             │
│  4. Collect Errors (~40-60% initially)                      │
│     For each error:                                         │
│     ├─> Student answer: "42"                               │
│     ├─> Ground truth: "108.51"                             │
│     ├─> Feedback: "❌ Answer incorrect. Should be 108.51   │
│     │              ❌ BBox IoU too low (0.23). Should align │
│     │                  with text region containing answer"  │
│     └─> Correction pair: (feedback → correct output)       │
│                                                             │
│  5. Train on Corrections (200 examples)                     │
│     Input:  Question + Image + Feedback                     │
│     Output: Corrected CoT + Answer + BBox                   │
│     Epochs: 1                                               │
│                                                             │
│  6. Measure Improvement                                     │
│     Error rate: 55% → 45% (10% improvement) ✓              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ ITERATION 2                                                 │
├─────────────────────────────────────────────────────────────┤
│  ... repeat with improved model ...                         │
│  Error rate: 45% → 35% (10% improvement) ✓                 │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ ITERATION 3                                                 │
├─────────────────────────────────────────────────────────────┤
│  ... repeat ...                                             │
│  Error rate: 35% → 30% (5% improvement) ✓                  │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ ITERATION 4                                                 │
├─────────────────────────────────────────────────────────────┤
│  ... repeat ...                                             │
│  Error rate: 30% → 27% (3% improvement)                     │
│  ✗ Improvement < 5% threshold → STOP (converged)           │
└─────────────────────────────────────────────────────────────┘
```

### VAL Verification Details

For each student output, VAL checks:

**1. OCR Grounding** (via DB-ResNet)
- Detect all text regions in image
- Check if answer exists in detected text
- ✓ Pass: Answer found in regions
- ✗ Fail: Answer not in any region

**2. Answer Quality** (ANLS)
- Compare with ground truth (if available)
- Or verify presence in OCR regions
- ✓ Pass: ANLS > 0.8 or found in text
- ✗ Fail: Doesn't match or not found

**3. BBox Quality** (IoU)
- Compute IoU with ground truth bbox
- Or with closest OCR region
- ✓ Pass: IoU > 0.5
- ✗ Fail: IoU ≤ 0.5

**4. Reasoning Quality** (7-step check)
- Step 1: Document understanding?
- Step 2: Question interpretation?
- Step 3: Visual localization?
- Step 4: Field identification?
- Step 5: Answer extraction (mentions answer)?
- Step 6: Spatial verification?
- Step 7: Bbox determination (mentions bbox)?
- ✓ Pass: All 7 steps complete
- ✗ Fail: Missing or incomplete steps

**Overall**: ✓ if all 4 checks pass, ✗ otherwise

### Feedback Format

Example feedback for incorrect output:

```
❌ Answer incorrect or not grounded in text.
   Your answer: '42'
   Ensure answer exists in detected text regions.

❌ Bounding box incorrect (IoU: 0.23).
   Your bbox: [100, 200, 150, 220]
   Should align with detected text region containing answer.

❌ Reasoning issues in: Step 5, Step 7
   Review the 7-step structure and ensure each step is complete.
```

## Running Phase B2

### On Mac

```bash
cd /Users/ahmadshirazi/Desktop/DocVAL
source envval/bin/activate

python train_b2_mac.py
```

### Expected Timeline

| Stage | Time | What Happens |
|-------|------|--------------|
| **Iteration 1** | | |
| - Generation | 30-45 min | Generate on 354 examples |
| - Verification | 20-30 min | VAL check + collect errors |
| - Training | 60-90 min | Train on ~200 corrections |
| **Total Iter 1** | **2-3 hours** | |
| | | |
| **Iterations 2-4** | 2-3 hours each | Fewer errors each time |
| | | |
| **Total** | **6-12 hours** | 3-5 iterations typical |

### Memory Usage

- **Generation**: ~30-40GB (loading student + inference)
- **Training**: ~50-60GB (same as Phase B1)
- **Safe with**: 64GB unified memory ✅

## Configuration

```python
config = B2Config(
    # Model from Phase B1
    student_model_path="docval/models/student_b1_mac/final",
    
    # Data
    val_data_path="docval/data/phase_a_output/filtered/D4_val.json",
    image_base_dir="docval/data/cot_data",
    
    # VAL settings
    text_detector_name="db_resnet",
    iou_threshold=0.5,  # Minimum IoU for "correct"
    
    # Convergence
    max_iterations=5,
    convergence_threshold=0.05,  # Stop if improvement < 5%
    correction_dataset_size=200,  # Use 200 worst examples per iteration
    
    # Training per iteration
    num_epochs_per_iter=1,
    learning_rate=5e-6,  # Lower than B1 (1e-5)
)
```

## Output

### During Training

```
================================================================================
ITERATION 1/5
================================================================================

GENERATING & VERIFYING OUTPUTS
================================================================================
Verifying: 100%|██████████| 200/200 [10:23<00:00,  3.12s/it]

Errors: 112/200 (56.0%)
Corrections collected: 112

Iteration 1 Error Rate: 56.0%
Improvement: 44.0%

Training on 112 corrections...
Epoch 1/1: 100%|██████████| 14/14 [1:23:45<00:00, 358.93s/it, loss=0.8234]

================================================================================
ITERATION 2/5
================================================================================

GENERATING & VERIFYING OUTPUTS
================================================================================
Verifying: 100%|██████████| 200/200 [10:15<00:00,  3.08s/it]

Errors: 85/200 (42.5%)
Corrections collected: 85

Iteration 2 Error Rate: 42.5%
Improvement: 13.5%

Training on 85 corrections...
...
```

### Files Created

```
docval/models/student_b2_mac/
├── iter_1/
│   ├── checkpoint-50/
│   └── final/
├── iter_2/
│   ├── checkpoint-50/
│   └── final/
├── iter_3/
│   └── ...
└── final/              ← Best performing model
    ├── config.json
    ├── model.safetensors
    └── ...
```

## Monitoring Progress

### Error Rate Tracking

| Iteration | Error Rate | Improvement | Status |
|-----------|------------|-------------|--------|
| 0 (B1) | 60% | - | Baseline |
| 1 | 45% | 15% | ✓ Continue |
| 2 | 35% | 10% | ✓ Continue |
| 3 | 30% | 5% | ✓ Continue |
| 4 | 27% | 3% | ✗ Converged (< 5%) |

### Expected Improvements

- **Iteration 1**: 10-20% error reduction (biggest jump)
- **Iteration 2-3**: 5-10% each
- **Iteration 4+**: < 5% (diminishing returns)

### Convergence Signs

✓ **Good convergence** (stop here):
- Error rate plateaus (< 5% improvement)
- Most outputs are correct (70-80%)
- Bbox IoU consistently > 0.5

⚠️ **Keep going**:
- Still seeing 10%+ improvements
- Error rate > 40%
- Low bbox IoU (< 0.4)

❌ **Problem** (investigate):
- Error rate increases
- No improvement after 2 iterations
- All validation examples failing

## Comparison: B1 vs B2

| Aspect | Phase B1 | Phase B2 |
|--------|----------|----------|
| **Training Data** | Teacher CoT examples | Corrections from errors |
| **Learning Style** | Imitation | Error correction |
| **Iterations** | 1 pass | 3-5 iterations |
| **Feedback** | None | Detailed VAL feedback |
| **BBox Quality** | Moderate | Much better |
| **Final Error Rate** | ~40-50% | ~25-30% |
| **Time** | 24-48 hours | 6-12 hours |

## Why Phase B2 Matters

Phase B1 teaches the student to **mimic** the teacher's reasoning.

Phase B2 teaches the student to **correct its mistakes** using **deterministic feedback**.

**Key Benefits**:
1. **Spatial reasoning improves** - BBox IoU increases significantly
2. **Answer accuracy increases** - Better grounding in detected text
3. **Reasoning quality** - 7-step structure becomes more consistent
4. **Convergence proof** - Iterative refinement ensures quality

## After Phase B2

Once training completes, you have a **refined student model** ready for **Phase C: Pure VLM Inference**.

### Phase C Preview

```python
# Phase C: No text detector needed!
model = AutoModelForImageTextToText.from_pretrained(
    "docval/models/student_b2_mac/final"
)

# Just image + question → Answer + BBox
image = Image.open("document.png")
question = "What is the total?"

# Student generates everything (no DB-ResNet!)
output = model.generate(image, question)
# → CoT + Answer + BBox
```

**The Goal**: Student performs document VQA with spatial grounding **without** text detection at inference time.

## Troubleshooting

### Issue: "Phase B1 model not found"
**Solution**: Either run Phase B1 first or point to a different model path

### Issue: High error rate not improving
**Solutions**:
1. Lower learning rate: `learning_rate=1e-6`
2. More corrections per iteration: `correction_dataset_size=300`
3. Multiple epochs per iteration: `num_epochs_per_iter=2`

### Issue: OOM during generation
**Solutions**:
1. Process fewer examples: Modify script to use `val_data[:100]`
2. Close other apps
3. Reduce `max_length` to 2048

### Issue: Text detector errors
**Solution**: Ensure `doctr` is installed: `pip install python-doctr[torch]`

## Quick Reference

### Start Training
```bash
cd /Users/ahmadshirazi/Desktop/DocVAL
source envval/bin/activate
python train_b2_mac.py
```

### Monitor Progress
Watch for:
- Error rate decreasing each iteration
- Convergence message (< 5% improvement)
- Final model saved to `/final`

### Test Final Model
```python
from transformers import AutoProcessor, AutoModelForImageTextToText

model = AutoModelForImageTextToText.from_pretrained(
    "docval/models/student_b2_mac/final"
)
processor = AutoProcessor.from_pretrained(
    "docval/models/student_b2_mac/final"
)

# Test on new image
# ... (use same inference code as Phase B1)
```

---

**Phase B2 is the secret sauce** 🔥

It's what turns a good model into a great one through iterative refinement!

