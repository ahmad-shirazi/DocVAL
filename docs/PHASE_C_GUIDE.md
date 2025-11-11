## Phase C: Inference & Evaluation - Complete Guide

## Overview

**Phase C** is the **final evaluation phase** where we test the trained student model on the held-out test set **without any text detector** - pure VLM inference!

### What Phase C Does:

```
Input:  Image + Question (no OCR!)
        ↓
   [Student VLM]
   (Gemma 3-12B)
        ↓
Output: 7-step CoT + Answer + BBox
```

**Key Point**: Unlike Phase A (teacher with OCR), Phase C uses **pure VLM** - the student must localize text and answer questions using only visual understanding!

## Prerequisites

### Phase B1 Must Be Complete

You need a trained student model from either:
- **Phase B1**: Basic CoT training → `docval/models/student_b1_mac/final`
- **Phase B2**: VAL feedback training → `docval/models/student_b2_mac/final` (better)

### Test Data Ready

- **D_test.json**: 355 held-out examples
- **Images**: In `docval/data/cot_data/`

## Quick Start

### Mac M4 Max

```bash
cd /Users/ahmadshirazi/Desktop/DocVAL
source envval/bin/activate

# Run evaluation
python run_phase_c_mac.py
```

This will:
1. Load your trained model
2. Run inference on all 355 test examples
3. Compute ANLS (answer quality) and IoU (bbox quality)
4. Save predictions and summary

**Time**: ~10-20 minutes on Mac M4 Max

## Detailed Usage

### Option 1: Simple Script (Recommended)

```bash
python run_phase_c_mac.py
```

Interactive - prompts for model path if not found.

### Option 2: Command Line

```bash
python -m docval.inference.evaluate_phase_c \
    --model docval/models/student_b1_mac/final \
    --test-data docval/data/phase_a_output/filtered/D_test.json \
    --image-dir docval/data/cot_data \
    --output-dir docval/results/phase_c_mac \
    --device mps
```

### Option 3: Python API

```python
from docval.inference.evaluate_phase_c import PhaseCEvaluator

evaluator = PhaseCEvaluator(
    model_path="docval/models/student_b1_mac/final",
    test_data_path="docval/data/phase_a_output/filtered/D_test.json",
    image_base_dir="docval/data/cot_data",
    device="mps"
)

summary, results = evaluator.evaluate(
    output_dir="docval/results/phase_c_mac",
    save_predictions=True
)

print(f"ANLS: {summary['avg_anls']:.3f}")
print(f"IoU: {summary['avg_iou']:.3f}")
```

## Evaluation Metrics

### 1. Answer Quality (ANLS)

**Average Normalized Levenshtein Similarity**
- Measures text similarity between predicted and ground truth answers
- Range: 0.0 (completely wrong) to 1.0 (perfect match)
- **Target**: >0.7 (good), >0.8 (excellent)

### 2. BBox Quality (IoU)

**Intersection over Union**
- Measures spatial localization accuracy
- Range: 0.0 (no overlap) to 1.0 (perfect match)
- **Target**: >0.5 (acceptable), >0.7 (good)

### 3. CoT Quality

- Number of reasoning steps generated
- **Target**: 7 steps (as trained)

### 4. Inference Speed

- Time per example
- Throughput (examples/sec)
- **Mac M4 Max**: ~2-5 sec/example

## Expected Results

### Phase B1 (Basic Training)

After Phase B1 training:

| Metric | Expected | Notes |
|--------|----------|-------|
| **ANLS** | 0.65-0.75 | Answer quality |
| **IoU** | 0.45-0.60 | BBox localization |
| **Success Rate** | 90-95% | Generated valid output |
| **Inference Time** | 2-5s | Per example on Mac |

### Phase B2 (With VAL Feedback)

After Phase B2 refinement:

| Metric | Expected | Notes |
|--------|----------|-------|
| **ANLS** | 0.70-0.80 | +5-10% improvement |
| **IoU** | 0.55-0.70 | +10-15% improvement |
| **Success Rate** | 95-98% | More robust |
| **Inference Time** | 2-5s | Same speed |

### Comparison with Teacher

| Model | ANLS | IoU | Speed | Cost |
|-------|------|-----|-------|------|
| **Teacher** (Gemini 2.5 Pro + OCR) | 0.85-0.90 | 0.70-0.80 | Slow | High |
| **Student** (Gemma 3-12B, no OCR) | 0.70-0.80 | 0.55-0.70 | Fast | Low |

**Trade-off**: Student achieves **70-80% of teacher performance** but is **10-20x faster** and **runs locally** without OCR!

## Output Files

After evaluation, you'll find:

```
docval/results/phase_c_mac/
├── predictions.json              # All 355 predictions
├── evaluation_summary.json       # Summary statistics
└── dataset_breakdown.json        # Per-dataset metrics (optional)
```

### predictions.json Structure

```json
[
  {
    "image_id": "cord_00005",
    "question": "What is the total amount?",
    "answer_gt": "60.000",
    "bbox_gt": [280, 534, 366, 552],
    "answer_pred": "60.000",
    "bbox_pred": [278, 532, 368, 554],
    "cot_steps": [
      "Step 1: This is a receipt...",
      "Step 2: Question asks for...",
      "...7 steps total..."
    ],
    "anls": 1.000,
    "iou": 0.943,
    "inference_time": 3.24,
    "success": true
  },
  ...
]
```

### evaluation_summary.json

```json
{
  "timestamp": "2025-11-11T...",
  "model_path": "docval/models/student_b1_mac/final",
  "total_examples": 355,
  "successful": 348,
  "failed": 7,
  "avg_anls": 0.732,
  "avg_iou": 0.584,
  "avg_cot_steps": 6.8,
  "avg_inference_time": 3.12,
  "throughput": 0.32,
  "dataset_metrics": {
    "cord": {"count": 5, "avg_anls": 0.812, "avg_iou": 0.623},
    "docvqa": {"count": 273, "avg_anls": 0.718, "avg_iou": 0.571},
    ...
  }
}
```

## Analyzing Results

### View Summary

```bash
# Quick view
cat docval/results/phase_c_mac/evaluation_summary.json | jq '.'

# Key metrics
jq '.avg_anls, .avg_iou, .avg_inference_time' \
   docval/results/phase_c_mac/evaluation_summary.json
```

### Find Best/Worst Examples

```python
import json

# Load predictions
with open('docval/results/phase_c_mac/predictions.json', 'r') as f:
    preds = json.load(f)

# Best ANLS
best = sorted([p for p in preds if p['success']], 
              key=lambda x: x['anls'], reverse=True)[:10]
print("Top 10 best answers:")
for p in best:
    print(f"{p['image_id']}: ANLS={p['anls']:.3f}")

# Worst IoU
worst = sorted([p for p in preds if p['success']], 
               key=lambda x: x['iou'])[:10]
print("\nTop 10 worst bbox:")
for p in worst:
    print(f"{p['image_id']}: IoU={p['iou']:.3f}")
```

### Dataset-Specific Performance

```python
from collections import defaultdict

dataset_metrics = defaultdict(list)
for p in preds:
    if p['success']:
        dataset = p['image_id'].split('_')[0]
        dataset_metrics[dataset].append({
            'anls': p['anls'],
            'iou': p['iou']
        })

for dataset, metrics in dataset_metrics.items():
    avg_anls = sum(m['anls'] for m in metrics) / len(metrics)
    avg_iou = sum(m['iou'] for m in metrics) / len(metrics)
    print(f"{dataset.upper()}: ANLS={avg_anls:.3f}, IoU={avg_iou:.3f}")
```

## Visualizing Results

### Create Visualization Script

```python
import matplotlib.pyplot as plt
import json

# Load results
with open('docval/results/phase_c_mac/predictions.json', 'r') as f:
    preds = json.load(f)

successful = [p for p in preds if p['success']]

# ANLS distribution
anls_scores = [p['anls'] for p in successful]
plt.figure(figsize=(10, 5))
plt.hist(anls_scores, bins=20, edgecolor='black')
plt.xlabel('ANLS Score')
plt.ylabel('Count')
plt.title('Answer Quality Distribution')
plt.savefig('anls_distribution.png')

# IoU distribution
iou_scores = [p['iou'] for p in successful]
plt.figure(figsize=(10, 5))
plt.hist(iou_scores, bins=20, edgecolor='black')
plt.xlabel('IoU Score')
plt.ylabel('Count')
plt.title('BBox Quality Distribution')
plt.savefig('iou_distribution.png')

print("✓ Saved anls_distribution.png")
print("✓ Saved iou_distribution.png")
```

## Comparing with Teacher (Phase A)

### Teacher vs Student

```python
# Load teacher outputs (Phase A)
with open('docval/data/phase_a_output/filtered/D_test.json', 'r') as f:
    teacher_data = json.load(f)

# Load student predictions
with open('docval/results/phase_c_mac/predictions.json', 'r') as f:
    student_data = json.load(f)

# Compare by image_id
for teacher, student in zip(teacher_data, student_data):
    if teacher['image_id'] == student['image_id']:
        print(f"\n{teacher['image_id']}:")
        print(f"  Question: {teacher['question'][:60]}...")
        print(f"  Teacher answer: {teacher['answer_pred']}")
        print(f"  Student answer: {student.get('answer_pred', 'N/A')}")
        print(f"  Match (ANLS):   {student.get('anls', 0):.3f}")
```

## Inference on New Images

### Single Image Inference

```python
from PIL import Image
from docval.inference.evaluate_phase_c import PhaseCEvaluator

# Load evaluator
evaluator = PhaseCEvaluator(
    model_path="docval/models/student_b1_mac/final",
    test_data_path="docval/data/phase_a_output/filtered/D_test.json",
    image_base_dir="docval/data/cot_data",
    device="mps"
)

# Create test example
example = {
    'image_id': 'test_001',
    'image_file': 'your_receipt.png',  # Place in appropriate folder
    'question': 'What is the total amount?'
}

# Run inference
result = evaluator.infer_single(example)

if result['success']:
    print(f"Answer: {result['parsed']['answer']}")
    print(f"BBox: {result['parsed']['bbox']}")
    print(f"Time: {result['inference_time']:.2f}s")
    print("\nReasoning:")
    for i, step in enumerate(result['parsed']['cot_steps'], 1):
        print(f"  Step {i}: {step[:100]}...")
```

## Troubleshooting

### Issue: Model not found

**Solution**: Check model path
```bash
ls docval/models/student_b1_mac/final/
# Should see: config.json, model.safetensors, etc.
```

### Issue: Low ANLS scores

**Possible causes**:
1. Phase B1 training didn't converge (check training loss)
2. Model overfitted to training data
3. Test set has different distribution

**Solutions**:
- Retrain with more epochs
- Use Phase B2 for iterative refinement
- Check dataset balance

### Issue: Low IoU scores

**This is expected!** BBox localization without OCR is challenging.

**Solutions**:
- Phase B2 with VAL feedback improves this significantly
- Consider relaxing IoU threshold (0.5 instead of 0.7)
- BBox is approximate - ANLS is more important

### Issue: Slow inference

**Mac M4 Max**: 2-5 sec/example is normal

**Speed it up**:
- Batch inference (not yet implemented)
- Quantize model to FP16 (when MPS supports it better)
- Use smaller model (Gemma 3-4B)

### Issue: Out of memory

**Solution**: Process in batches
```python
# Modify evaluate() to process 10 at a time
for i in range(0, len(test_data), 10):
    batch = test_data[i:i+10]
    # Process batch
    torch.mps.empty_cache()  # Clear cache between batches
```

## Expected Timeline

On **Mac M4 Max**:
- **Load model**: 1-2 minutes
- **Per example**: 2-5 seconds
- **Total (355 examples)**: 10-20 minutes

## Success Criteria

Your Phase C is successful if:

✅ **ANLS > 0.65**: Reasonable answer quality  
✅ **IoU > 0.50**: Acceptable bbox localization  
✅ **Success rate > 90%**: Model is robust  
✅ **CoT steps ≈ 7**: Proper reasoning structure

## Next Steps After Phase C

1. **Analyze failures**: Look at low-scoring examples
2. **If ANLS < 0.65**: Re-train Phase B1 with more epochs
3. **If IoU < 0.50**: Run Phase B2 (VAL feedback)
4. **If satisfied**: Deploy model for production!

### Deployment

Your trained model can now:
- ✅ Answer questions about documents
- ✅ Localize answers (bbox)
- ✅ Explain reasoning (7-step CoT)
- ✅ Run on Mac (no cloud needed!)
- ✅ No OCR dependency

```python
# Simple deployment API
def answer_document_question(image_path, question):
    result = evaluator.infer_single({
        'image_id': 'api_query',
        'image_file': image_path,
        'question': question
    })
    return {
        'answer': result['parsed']['answer'],
        'bbox': result['parsed']['bbox'],
        'reasoning': result['parsed']['cot_steps']
    }
```

## Summary

**Phase C** evaluates your trained student model on the final test set using **pure VLM inference** (no text detector). 

**Key Metrics**:
- ANLS (answer quality): Target >0.70
- IoU (bbox quality): Target >0.50  
- Inference speed: 2-5 sec/example on Mac

**Ready to evaluate?** 

```bash
python run_phase_c_mac.py
```

---

**Phase C marks the completion of the DocVAL pipeline!** 🎉

Your student model now:
- Understands documents visually
- Answers questions with reasoning
- Localizes answers spatially
- Runs efficiently on your Mac

**Congratulations!** 🚀

