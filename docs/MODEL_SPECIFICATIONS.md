# DocVAL Model Specifications

## 🎓 Teacher Models (Large VLMs for CoT Generation)

### Default
**Gemini 2.5 Pro** ✅
- Model: `gemini-2.5-pro`
- Type: **Thinking Model** (no temperature parameter)
- Size: Large (proprietary)
- Provider: Google
- Cost: ~$510 for 102K examples
- Used for: Generating high-quality CoT traces in Phase A
- Features: Extended reasoning, deterministic output

### Alternative Proprietary Models
1. **GPT-5**
   - Provider: OpenAI
   - Latest GPT model

2. **Claude 4.5 Sonnet**
   - Provider: Anthropic
   - High-quality reasoning

3. **Gemini 2.5 Flash**
   - Provider: Google
   - Faster, more cost-effective variant

4. **GPT-4o**
   - Provider: OpenAI
   - Omni-modal capabilities

### Open-Source Alternatives
1. **Qwen3-VL-235B-A22B-Thinking**
   - Size: 235B parameters (22B active)
   - Provider: Alibaba Cloud
   - Features: Thinking/reasoning capabilities

2. **Llama4-400B-A17B**
   - Size: 400B parameters (17B active)
   - Provider: Meta
   - Features: Mixture-of-experts architecture

## 🎯 Student Models (Compact VLMs for Deployment)

### Default
**Gemma3-12B** ✅
- Model: `google/gemma-3-12b`
- Size: 12B parameters
- Provider: Google
- Expected Performance: 91.4% ANLS, 82.4% mAP

### Small Model (4B)
**Gemma3-4B**
- Model: `google/gemma-3-4b`
- Size: 4B parameters
- Expected Performance: 88.7% ANLS, 69.1% mAP
- Use case: Faster inference, lower memory

### Medium Models (8B)
1. **Qwen3-VL-8B-Thinking**
   - Model: `qwen/qwen3-vl-8b-thinking`
   - Size: 8B parameters
   - Features: Built-in reasoning capabilities

2. **InternVL3.5-8B**
   - Model: `OpenGVLab/InternVL3.5-8B`
   - Size: 8B parameters
   - Features: Strong vision understanding

### Large Models (11B-14B)
1. **Llama-3.2-11B-Vision**
   - Model: `meta-llama/Llama-3.2-11B-Vision`
   - Size: 11B parameters
   - Features: Meta's vision-language model

2. **InternVL3.5-14B**
   - Model: `OpenGVLab/InternVL3.5-14B`
   - Size: 14B parameters
   - Features: Highest capacity in medium range

## 🔍 Text Detection Models (Training-Time Only)

### Default
**DB-ResNet** ✅
- Framework: doctr
- Architecture: Differentiable Binarization with ResNet backbone
- Inference Time: ~50ms on GPU
- Regions per Document: 15-20
- Use: Phase A (teacher generation) and Phase B2 (VAL verification)

### Alternatives (Ablation Studies)
1. **CRAFT**
   - Character Region Awareness For Text detection
   - Features: Character-level detection

2. **PSENet**
   - Progressive Scale Expansion Network
   - Features: Arbitrary-shaped text detection

3. **EasyOCR**
   - All-in-one OCR solution
   - Features: Detection + Recognition combined

## 📊 Model Selection Guide

### For Best Performance
- Teacher: **Gemini 2.5 Pro**
- Student: **Gemma3-12B**
- Detector: **DB-ResNet**
- Expected: 91.4% ANLS, 82.4% mAP

### For Fast Inference
- Teacher: **Gemini 2.5 Flash**
- Student: **Gemma3-4B**
- Detector: **DB-ResNet**
- Expected: 88.7% ANLS, 69.1% mAP
- Inference: ~450ms/query

### For Open-Source Stack
- Teacher: **Qwen3-VL-235B-A22B-Thinking**
- Student: **Qwen3-VL-8B-Thinking**
- Detector: **EasyOCR**
- Cost: No API costs, but requires large GPU for teacher

### For Maximum Capacity
- Teacher: **Llama4-400B-A17B**
- Student: **InternVL3.5-14B**
- Detector: **DB-ResNet**
- Trade-off: Higher compute, potentially better performance

## 🔧 Configuration Examples

### Default (Recommended - Thinking Model)
```yaml
teacher:
  model_name: "gemini-2.5-pro"
  temperature: null  # Required: null for thinking models
  max_tokens: 2048

student:
  model_name: "google/gemma-3-12b"
  sequence_length: 2048

detector:
  model: "db_resnet"
  inference_time_ms: 50
```

### Fast & Efficient
```yaml
teacher:
  model_name: "gemini-2.5-flash"

student:
  model_name: "google/gemma-3-4b"

detector:
  model: "db_resnet"
```

### Open-Source
```yaml
teacher:
  model_name: "qwen/qwen3-vl-235b-a22b-thinking"

student:
  model_name: "qwen/qwen3-vl-8b-thinking"

detector:
  model: "easyocr"
```

## 🎯 Model Comparison Matrix

| Student Model | Size | DocVQA ANLS | DocVQA mAP | Inference Speed |
|--------------|------|-------------|------------|-----------------|
| Gemma3-4B | 4B | 88.7% | 69.1% | ~450ms |
| Qwen3-VL-8B | 8B | ~89-90% | ~72-75% | ~600ms |
| InternVL3.5-8B | 8B | ~89-90% | ~73-76% | ~620ms |
| Llama-3.2-11B | 11B | ~90-91% | ~78-80% | ~800ms |
| **Gemma3-12B** | **12B** | **91.4%** | **82.4%** | **~950ms** |
| InternVL3.5-14B | 14B | ~91-92% | ~80-83% | ~1100ms |

## 📝 Notes

1. **Teacher Model**: Only used during Phase A for generating training data. No inference-time cost.

2. **Student Model**: Used throughout training (B1, B2) and at inference. Choose based on deployment constraints.

3. **Text Detector**: Only used during training phases (A, B2) for validation. Student never uses it at inference.

4. **Model Availability**: 
   - Gemma3 models: Check HuggingFace for latest releases
   - Some models may require authentication/licenses
   - Open-source alternatives available for all components

5. **Hardware Requirements**:
   - Teacher (Phase A): API-based or 4x A100 80GB for open-source
   - Student Training: 1x H100 80GB or A100 80GB
   - Inference: RTX 4090 (28GB) or better

## 🔄 Switching Models

To use a different model, simply update the config:

```bash
# Edit docval/config/default_config.yaml
teacher:
  model_name: "your-preferred-teacher-model"

student:
  model_name: "your-preferred-student-model"
```

Or pass as arguments (if implemented):
```bash
python -m docval.main \
    --teacher-model "gemini-2.5-pro" \
    --student-model "google/gemma-3-4b"
```

