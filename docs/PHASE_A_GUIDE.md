# Phase A: Teacher Data Generation Guide

## 🎯 Overview

Phase A generates high-quality Chain-of-Thought (CoT) training data using:
- **Teacher Model**: Gemini 2.5 Pro (thinking model)
- **Text Detector**: DB-ResNet50 from [doctr](https://github.com/mindee/doctr)
- **Output**: 7-step structured CoT traces with bbox annotations

## 📋 Prerequisites

### 1. Install Dependencies

```bash
pip install python-doctr[torch]>=0.7.0
pip install google-generativeai>=0.3.0
pip install python-dotenv>=1.0.0
```

### 2. Set API Key

Create `.env` file in project root:

```bash
# /Users/ahmadshirazi/Desktop/DocVAL/.env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Prepare Data

Organize your datasets:

```
docval/data/cot_data/
├── CORD/
│   ├── images/
│   │   ├── image001.jpg
│   │   └── ...
│   └── annotations.json
├── DocVQA/
│   ├── images/
│   └── annotations.json
└── FUNSD/
    ├── images/
    └── annotations.json
```

**annotations.json format:**
```json
[
  {
    "image_id": "cord_001",
    "image_file": "image001.jpg",
    "question": "What is the total amount?",
    "answer": "$108.51",
    "bbox": [542, 876, 618, 904]
  },
  ...
]
```

## 🚀 Running Phase A

### Quick Start

```bash
cd /Users/ahmadshirazi/Desktop/DocVAL
python -m docval.scripts.run_phase_a
```

### With Options

```bash
# Process specific datasets
python -m docval.scripts.run_phase_a \
    --datasets CORD DocVQA FUNSD \
    --data-dir /Users/ahmadshirazi/Desktop/DocVAL/docval/data/cot_data \
    --output-dir ./outputs/phase_a

# Test with limited examples
python -m docval.scripts.run_phase_a \
    --max-examples 10 \
    --datasets CORD
```

### Full Options

```bash
python -m docval.scripts.run_phase_a \
    --data-dir /path/to/data \
    --output-dir /path/to/output \
    --datasets CORD DocVQA FUNSD \
    --max-examples 100 \
    --teacher-model gemini-2.5-pro
```

## 📊 7-Step CoT Structure

Phase A generates structured reasoning following this format:

### Step 1: Document Understanding
Identify document type and overall structure.

**Example:**
```
"This is a restaurant receipt with itemized charges and summary fields"
```

### Step 2: Question Interpretation
Parse what information is being requested.

**Example:**
```
"Question asks for 'total amount' - need the final payment total, not subtotal"
```

### Step 3: Visual Region Localization
Describe where to look in the document.

**Example:**
```
"Scanning lower portion of receipt where totals typically appear"
```

### Step 4: Field/Label Identification
Identify the specific label or field.

**Example:**
```
"Located 'TOTAL' label in bold text at bottom-right, distinct from 'SUBTOTAL' above"
```

### Step 5: Answer Extraction
Extract and validate the answer text.

**Example:**
```
"Amount '$108.51' positioned immediately to right of TOTAL label"
```

### Step 6: Spatial Verification
Confirm spatial relationships and positioning.

**Example:**
```
"This is the lowest/final line item, confirming it's the grand total"
```

### Step 7: Bbox Determination
Specify bounding box coordinates with reasoning.

**Example:**
```
"Bbox encompasses the numerical value: approximately [542, 876, 618, 904]"
```

## 🔧 Technical Details

### DB-ResNet Text Detection

The system uses **DB-ResNet50** from doctr:

```python
from doctr.models import ocr_predictor

# Initialize detector
detector = ocr_predictor(
    det_arch='db_resnet50',
    reco_arch='crnn_vgg16_bn',
    pretrained=True
)

# Run detection
regions = detector([image])
```

**Output Format:**
```python
[
    {
        'id': 0,
        'bbox': [100, 200, 300, 250],  # [x1, y1, x2, y2]
        'text': 'Total Amount',
        'confidence': 0.95
    },
    ...
]
```

### Gemini 2.5 Pro Integration

Teacher model generates structured CoT:

```python
from models.teacher_vlm import TeacherVLM

# Initialize teacher
teacher = TeacherVLM(
    model_name="gemini-2.5-pro",
    temperature=None,  # Thinking model
    max_tokens=8192
)

# Generate CoT
output = teacher.generate_cot(
    image=image,
    question="What is the total?",
    regions=detected_regions
)
```

**Output Format:**
```python
{
    'cot_steps': [
        'Step 1: This is a restaurant receipt...',
        'Step 2: Question asks for total amount...',
        ...
    ],
    'answer': '$108.51',
    'bbox': [542, 876, 618, 904],
    'raw_output': '...'  # Full Gemini response
}
```

## 📁 Output Structure

Phase A generates CoT data files:

```
outputs/phase_a/
├── cot_cord.json
├── cot_docvqa.json
└── cot_funsd.json
```

**Output JSON Format:**
```json
[
  {
    "dataset": "CORD",
    "image_id": "cord_001",
    "image_file": "image001.jpg",
    "question": "What is the total amount?",
    "answer_gt": "$108.51",
    "bbox_gt": [542, 876, 618, 904],
    "cot_steps": [
      "Step 1: This is a restaurant receipt...",
      "Step 2: Question asks for 'total amount'...",
      "Step 3: Scanning lower portion of receipt...",
      "Step 4: Located 'TOTAL' label...",
      "Step 5: Amount '$108.51' to right of label...",
      "Step 6: This is the final line item...",
      "Step 7: Bbox [542, 876, 618, 904]..."
    ],
    "answer_pred": "$108.51",
    "bbox_pred": [542, 876, 618, 904],
    "regions": [
      {"id": 0, "bbox": [...], "text": "...", "confidence": 0.95},
      ...
    ],
    "raw_output": "REASONING:\nStep 1: ..."
  },
  ...
]
```

## ⚡ Performance

### Speed
- **DB-ResNet Detection**: ~50ms per image (GPU)
- **Gemini 2.5 Pro Generation**: ~2-5s per example
- **Total**: ~2-6s per example

### Cost
- **Gemini 2.5 Pro**: ~$0.005 per example
- **102K examples**: ~$510 total

### Output
- **Typical**: 15-20 text regions per document
- **CoT Length**: 500-1000 tokens per example
- **Quality Score**: Q > 0.85 (85% pass VAL filter)

## 🐛 Troubleshooting

### API Key Not Found
```bash
# Error: GEMINI_API_KEY not found
# Solution: Check .env file exists and is properly formatted
cat .env
# Should show: GEMINI_API_KEY=AIza...
```

### doctr Not Installed
```bash
# Error: doctr not available
# Solution: Install with torch backend
pip install python-doctr[torch]
```

### Images Not Found
```bash
# Error: Image not found: /path/to/image.jpg
# Solution: Check data directory structure matches expected format
ls -R docval/data/cot_data/CORD/
```

### Rate Limiting
```bash
# Error: Gemini API rate limit exceeded
# Solution: Add delays between requests or use batch processing
# The script saves incrementally, so you can resume from last checkpoint
```

## 📊 Monitoring Progress

The script saves results incrementally every 10 examples:

```bash
# Check progress
tail -f outputs/phase_a/cot_cord.json

# Count completed examples
python -c "import json; print(len(json.load(open('outputs/phase_a/cot_cord.json'))))"
```

## 🔄 Resume After Interruption

The script saves incrementally, so you can filter already processed examples:

```python
# Load existing results
with open('outputs/phase_a/cot_cord.json') as f:
    existing = json.load(f)
    processed_ids = {ex['image_id'] for ex in existing}

# Filter new annotations
new_annotations = [
    ann for ann in all_annotations 
    if ann['image_id'] not in processed_ids
]
```

## 📚 References

- [doctr Documentation](https://mindee.github.io/doctr/)
- [DB-ResNet Paper](https://arxiv.org/abs/1911.08947)
- [Gemini API Docs](https://ai.google.dev/gemini-api/docs)
- [HuggingFace Model](https://huggingface.co/Felix92/doctr-tf-db-resnet50)

## 💡 Tips

1. **Start Small**: Test with `--max-examples 10` first
2. **Check Quality**: Manually review first few outputs
3. **Monitor Costs**: Track API usage in Google Cloud Console
4. **Save Checkpoints**: Script saves every 10 examples automatically
5. **Use GPU**: DB-ResNet runs much faster on GPU

## 🎯 Next Steps

After Phase A completes:

1. **Apply VAL Filter**: Filter to Q > 0.85 (Stage B preparation)
2. **Split Data**: Create D3/D4/Dtest splits (80/10/10)
3. **Start Stage B1**: Supervised fine-tuning on filtered data

```bash
# Next: Run VAL filter (if implemented)
python -m docval.scripts.apply_val_filter \
    --input outputs/phase_a/ \
    --output outputs/filtered/
```

---

**Ready to generate CoT data!** 🚀

