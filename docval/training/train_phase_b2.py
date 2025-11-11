"""
Phase B2: Student CoT Instruction Tuning with VAL Feedback
Iterative refinement using DocVAL feedback on reasoning quality

Input: Fine-tuned student from B1 + D4 validation set
Process: Generate → VAL verify → Collect errors → Retrain on corrections
Output: Refined student with improved spatial reasoning
"""

import os
import json
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from PIL import Image
import numpy as np
from tqdm import tqdm

from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    TrainingArguments,
    Trainer,
)

# Import VAL filter for verification
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from docval.models.text_detector import TextDetector


@dataclass
class B2Config:
    """Configuration for Phase B2 iterative training"""
    # Model
    student_model_path: str = "docval/models/student_b1_mac/final"  # From Phase B1
    
    # Data
    val_data_path: str = "docval/data/phase_a_output/filtered/D4_val.json"
    image_base_dir: str = "docval/data/cot_data"
    
    # VAL feedback settings
    text_detector_name: str = "db_resnet"  # For validation
    iou_threshold: float = 0.5  # Minimum IoU for correct bbox
    
    # Iterative training
    output_dir: str = "docval/models/student_b2_mac"
    max_iterations: int = 5  # Maximum refinement iterations
    convergence_threshold: float = 0.05  # Stop if error rate improvement < 5%
    correction_dataset_size: int = 200  # Examples per iteration
    
    # Training per iteration
    num_epochs_per_iter: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-6  # Lower for refinement
    
    # Hardware
    use_mps_device: bool = True
    max_length: int = 4096


class VALVerifier:
    """Verifies student outputs using VAL criteria"""
    
    def __init__(self, text_detector: TextDetector, iou_threshold: float = 0.5):
        self.text_detector = text_detector
        self.iou_threshold = iou_threshold
    
    def verify_output(self, student_output: Dict, ground_truth: Dict, 
                     image_path: Path) -> Dict:
        """
        Verify student output against ground truth using VAL.
        
        Returns:
            {
                'overall_correct': bool,
                'answer_correct': bool,
                'bbox_correct': bool,
                'reasoning_issues': List[int],  # Which steps are problematic
                'feedback': str,  # Detailed feedback
                'iou_score': float
            }
        """
        # Detect text regions for verification
        regions = self.text_detector.detect(image_path, return_text=True)
        
        result = {
            'overall_correct': False,
            'answer_correct': False,
            'bbox_correct': False,
            'reasoning_issues': [],
            'feedback': '',
            'iou_score': 0.0
        }
        
        # Check answer
        answer_pred = student_output.get('answer', '')
        answer_gt = ground_truth.get('answer_pred', ground_truth.get('answer_gt', ''))
        
        if answer_gt:
            # ANLS-based check
            answer_similarity = self._compute_anls(answer_pred, answer_gt)
            result['answer_correct'] = answer_similarity > 0.8
        else:
            # Check if answer exists in detected regions
            result['answer_correct'] = self._answer_in_regions(answer_pred, regions)
        
        # Check bbox
        bbox_pred = student_output.get('bbox', [])
        bbox_gt = ground_truth.get('bbox_pred', ground_truth.get('bbox_gt'))
        
        if bbox_pred and bbox_gt:
            # Check IoU with ground truth
            iou_gt = self._compute_iou(bbox_pred, bbox_gt)
            result['iou_score'] = iou_gt
            result['bbox_correct'] = iou_gt > self.iou_threshold
        elif bbox_pred and regions:
            # Check IoU with detected regions
            max_iou = max([self._compute_iou(bbox_pred, r['bbox']) for r in regions], default=0.0)
            result['iou_score'] = max_iou
            result['bbox_correct'] = max_iou > self.iou_threshold
        
        # Check reasoning quality
        cot_steps = student_output.get('cot_steps', [])
        result['reasoning_issues'] = self._check_reasoning(cot_steps, answer_pred, bbox_pred)
        
        # Overall correctness
        result['overall_correct'] = (
            result['answer_correct'] and 
            result['bbox_correct'] and 
            len(result['reasoning_issues']) == 0
        )
        
        # Generate feedback
        result['feedback'] = self._generate_feedback(result, answer_pred, bbox_pred, regions)
        
        return result
    
    def _compute_anls(self, pred: str, gt: str) -> float:
        """Compute ANLS similarity"""
        if not pred or not gt:
            return 0.0
        pred = pred.lower().strip()
        gt = gt.lower().strip()
        if pred == gt:
            return 1.0
        distance = self._levenshtein_distance(pred, gt)
        max_len = max(len(pred), len(gt))
        return max(0.0, 1.0 - distance / max_len) if max_len > 0 else 0.0
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Compute edit distance"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]
    
    def _answer_in_regions(self, answer: str, regions: List[Dict]) -> bool:
        """Check if answer exists in detected regions"""
        if not answer:
            return False
        answer_lower = answer.lower().strip()
        for region in regions:
            region_text = region.get('text', '').lower().strip()
            if answer_lower in region_text or region_text in answer_lower:
                return True
        return False
    
    def _compute_iou(self, bbox1: List, bbox2: List) -> float:
        """Compute IoU between two bboxes"""
        if not bbox1 or not bbox2 or len(bbox1) != 4 or len(bbox2) != 4:
            return 0.0
        
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _check_reasoning(self, cot_steps: List[str], answer: str, bbox: List) -> List[int]:
        """Check which reasoning steps are problematic"""
        issues = []
        
        if len(cot_steps) != 7:
            return list(range(len(cot_steps)))  # All steps problematic if not 7
        
        # Step 1: Document understanding
        if len(cot_steps[0]) < 20 or not any(kw in cot_steps[0].lower() 
                                              for kw in ['document', 'receipt', 'form', 'invoice']):
            issues.append(0)
        
        # Step 2: Question interpretation
        if len(cot_steps[1]) < 20 or 'question' not in cot_steps[1].lower():
            issues.append(1)
        
        # Step 3: Visual localization
        if len(cot_steps[2]) < 20 or not any(kw in cot_steps[2].lower() 
                                              for kw in ['visual', 'region', 'location', 'where']):
            issues.append(2)
        
        # Step 4: Field identification
        if len(cot_steps[3]) < 20 or not any(kw in cot_steps[3].lower() 
                                              for kw in ['field', 'label', 'located', 'found']):
            issues.append(3)
        
        # Step 5: Answer extraction (should mention the answer)
        if len(cot_steps[4]) < 20 or not answer or answer.lower() not in cot_steps[4].lower():
            issues.append(4)
        
        # Step 6: Spatial verification
        if len(cot_steps[5]) < 20 or not any(kw in cot_steps[5].lower() 
                                              for kw in ['spatial', 'position', 'verify', 'confirm']):
            issues.append(5)
        
        # Step 7: Bbox determination (should mention bbox/coordinates)
        if len(cot_steps[6]) < 20 or 'bbox' not in cot_steps[6].lower():
            issues.append(6)
        
        return issues
    
    def _generate_feedback(self, result: Dict, answer: str, bbox: List, 
                          regions: List[Dict]) -> str:
        """Generate detailed feedback for corrections"""
        feedback = []
        
        if not result['answer_correct']:
            feedback.append(
                f"❌ Answer incorrect or not grounded in text.\n"
                f"   Your answer: '{answer}'\n"
                f"   Ensure answer exists in detected text regions."
            )
        else:
            feedback.append("✓ Answer correct")
        
        if not result['bbox_correct']:
            feedback.append(
                f"❌ Bounding box incorrect (IoU: {result['iou_score']:.2f}).\n"
                f"   Your bbox: {bbox}\n"
                f"   Should align with detected text region containing answer."
            )
        else:
            feedback.append(f"✓ Bbox correct (IoU: {result['iou_score']:.2f})")
        
        if result['reasoning_issues']:
            steps_str = ', '.join([f"Step {i+1}" for i in result['reasoning_issues']])
            feedback.append(
                f"❌ Reasoning issues in: {steps_str}\n"
                f"   Review the 7-step structure and ensure each step is complete."
            )
        else:
            feedback.append("✓ Reasoning complete")
        
        return '\n'.join(feedback)


class CorrectionDataset(torch.utils.data.Dataset):
    """Dataset of corrections for iterative training"""
    
    def __init__(self, corrections: List[Dict], processor, max_length: int = 4096):
        self.corrections = corrections
        self.processor = processor
        self.max_length = max_length
    
    def __len__(self):
        return len(self.corrections)
    
    def __getitem__(self, idx):
        correction = self.corrections[idx]
        
        # Load image
        try:
            image = Image.open(correction['image_path']).convert('RGB')
            if max(image.size) > 1024:
                image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        except:
            image = Image.new('RGB', (224, 224), color='white')
        
        # Input: question + feedback
        question = correction['question']
        feedback = correction['feedback']
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"{question}\n\nFeedback on previous attempt:\n{feedback}"}
            ]
        }]
        
        # Target: corrected output
        target_text = correction['corrected_output']
        
        # Process
        inputs = self.processor.apply_chat_template(
            messages,
            images=[image],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        
        target_ids = self.processor.tokenizer(
            target_text,
            max_length=self.max_length // 2,
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        input_ids = inputs['input_ids'].squeeze(0)
        full_input_ids = torch.cat([input_ids, target_ids.squeeze(0)])
        labels = torch.cat([
            torch.full_like(input_ids, -100),
            target_ids.squeeze(0)
        ])
        
        return {
            'input_ids': full_input_ids[:self.max_length],
            'labels': labels[:self.max_length],
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'attention_mask': torch.ones_like(full_input_ids[:self.max_length])
        }


def generate_and_verify(model, processor, val_data: List[Dict], 
                       verifier: VALVerifier, image_base_dir: Path,
                       device: torch.device) -> Tuple[List[Dict], float]:
    """
    Generate outputs for validation set and verify with VAL.
    
    Returns:
        (corrections, error_rate)
    """
    print("\n" + "="*80)
    print("GENERATING & VERIFYING OUTPUTS")
    print("="*80)
    
    model.eval()
    corrections = []
    errors = 0
    
    for example in tqdm(val_data[:200], desc="Verifying"):  # Limit for speed
        # Get image path
        dataset = example['image_id'].split('_')[0]
        if dataset == 'cord':
            image_path = image_base_dir / 'CORD' / example['image_file']
        elif dataset == 'docvqa':
            image_path = image_base_dir / 'DocVQA' / 'images' / example['image_file']
        elif dataset == 'funsd':
            image_path = image_base_dir / 'FUNSD' / example['image_file']
        elif dataset == 'sroie':
            image_path = image_base_dir / 'SROIE' / example['image_file']
        elif dataset == 'visualmrc':
            image_path = image_base_dir / 'VisualMRC' / example['image_file']
        else:
            continue
        
        if not image_path.exists():
            continue
        
        # Generate output
        try:
            image = Image.open(image_path).convert('RGB')
            question = example['question']
            
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
            ).to(device)
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=1024)
            
            response = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            
            # Parse response
            student_output = parse_cot_response(response)
            
            # Verify with VAL
            verification = verifier.verify_output(student_output, example, image_path)
            
            if not verification['overall_correct']:
                errors += 1
                
                # Create correction
                corrected_output = format_cot_output(
                    example.get('cot_steps', []),
                    example.get('answer_pred', ''),
                    example.get('bbox_pred', [])
                )
                
                corrections.append({
                    'image_path': str(image_path),
                    'question': question,
                    'feedback': verification['feedback'],
                    'student_output': student_output,
                    'corrected_output': corrected_output,
                    'verification': verification
                })
        
        except Exception as e:
            print(f"Error processing {example['image_id']}: {e}")
            continue
    
    error_rate = errors / len(val_data[:200]) if val_data else 0.0
    
    print(f"\nErrors: {errors}/{len(val_data[:200])} ({error_rate:.1%})")
    print(f"Corrections collected: {len(corrections)}")
    
    return corrections, error_rate


def parse_cot_response(response: str) -> Dict:
    """Parse model response into structured format"""
    import re
    
    cot_steps = []
    reasoning_match = re.search(r'REASONING:(.*?)(?=ANSWER:|$)', response, re.DOTALL | re.IGNORECASE)
    if reasoning_match:
        reasoning_text = reasoning_match.group(1).strip()
        steps = re.findall(r'Step \d+:\s*(.*?)(?=\n\nStep \d+:|\nStep \d+:|$)', reasoning_text, re.DOTALL)
        cot_steps = [step.strip() for step in steps if step.strip()]
    
    answer_match = re.search(r'ANSWER:\s*(.+?)(?=\nBBOX:|$)', response, re.IGNORECASE)
    answer = answer_match.group(1).strip() if answer_match else ""
    
    bbox_match = re.search(r'BBOX:\s*\[([^\]]+)\]', response, re.IGNORECASE)
    bbox = []
    if bbox_match:
        try:
            bbox = [float(x.strip()) for x in bbox_match.group(1).split(',')]
        except:
            pass
    
    return {
        'cot_steps': cot_steps,
        'answer': answer,
        'bbox': bbox
    }


def format_cot_output(cot_steps: List[str], answer: str, bbox: List) -> str:
    """Format CoT output for training"""
    reasoning = "REASONING:\n"
    for i, step in enumerate(cot_steps, 1):
        reasoning += f"Step {i}: {step}\n"
    return f"{reasoning}\nANSWER: {answer}\nBBOX: {bbox}"


def train_phase_b2(config: B2Config):
    """Main iterative training loop for Phase B2"""
    
    print("\n" + "="*80)
    print("PHASE B2: VAL FEEDBACK ITERATIVE TRAINING")
    print("="*80)
    print(f"\nStudent model: {config.student_model_path}")
    print(f"Val data: {config.val_data_path}")
    print(f"Max iterations: {config.max_iterations}")
    print(f"Convergence threshold: {config.convergence_threshold:.1%}")
    
    # Setup device
    device = torch.device("mps" if torch.backends.mps.is_available() and config.use_mps_device else "cpu")
    print(f"Device: {device}")
    
    # Load student model from Phase B1
    print(f"\nLoading student model from Phase B1...")
    processor = AutoProcessor.from_pretrained(config.student_model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        config.student_model_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    ).to(device)
    
    # Load validation data
    with open(config.val_data_path, 'r') as f:
        val_data = json.load(f)
    print(f"Loaded {len(val_data)} validation examples")
    
    # Initialize VAL verifier
    print("\nInitializing VAL verifier...")
    text_detector = TextDetector(config.text_detector_name)
    verifier = VALVerifier(text_detector, config.iou_threshold)
    
    # Iterative training loop
    image_base_dir = Path(config.image_base_dir)
    prev_error_rate = 1.0
    
    for iteration in range(config.max_iterations):
        print(f"\n{'='*80}")
        print(f"ITERATION {iteration + 1}/{config.max_iterations}")
        print(f"{'='*80}")
        
        # Generate and verify
        corrections, error_rate = generate_and_verify(
            model, processor, val_data, verifier, image_base_dir, device
        )
        
        print(f"\nIteration {iteration + 1} Error Rate: {error_rate:.1%}")
        
        # Check convergence
        improvement = prev_error_rate - error_rate
        print(f"Improvement: {improvement:.1%}")
        
        if improvement < config.convergence_threshold:
            print(f"\n✓ Converged! Improvement < {config.convergence_threshold:.1%}")
            break
        
        if not corrections:
            print("\n✓ No corrections needed!")
            break
        
        # Train on corrections
        print(f"\nTraining on {len(corrections)} corrections...")
        
        correction_dataset = CorrectionDataset(corrections, processor, config.max_length)
        
        training_args = TrainingArguments(
            output_dir=f"{config.output_dir}/iter_{iteration+1}",
            num_train_epochs=config.num_epochs_per_iter,
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            logging_steps=10,
            save_steps=50,
            use_mps_device=config.use_mps_device,
            remove_unused_columns=False,
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=correction_dataset,
        )
        
        trainer.train()
        
        prev_error_rate = error_rate
    
    # Save final model
    final_path = Path(config.output_dir) / "final"
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    
    print(f"\n{'='*80}")
    print("✓ PHASE B2 COMPLETE")
    print(f"{'='*80}")
    print(f"\nFinal model saved to: {final_path}")


if __name__ == "__main__":
    config = B2Config()
    train_phase_b2(config)

