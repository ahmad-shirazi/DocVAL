"""
Phase C: Pure VLM Inference & Evaluation
Test the trained student model without text detector

Input: D_test.json (355 examples)
Model: Fine-tuned Gemma 3-12B from Phase B1/B2
Output: CoT reasoning + Answer + BBox (pure VLM, no OCR)
"""

import os
import json
import torch
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image
import numpy as np
from tqdm import tqdm
import time
from datetime import datetime

from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel


class PhaseCEvaluator:
    """Evaluator for Phase C inference"""
    
    def __init__(self, model_path: str, test_data_path: str, 
                 image_base_dir: str, device: str = "mps"):
        self.model_path = Path(model_path)
        self.test_data_path = Path(test_data_path)
        self.image_base_dir = Path(image_base_dir)
        self.device = torch.device(device)
        
        # Load model and processor
        print("\n" + "="*80)
        print("LOADING MODEL")
        print("="*80)
        self._load_model()
        
        # Load test data
        print("\n" + "="*80)
        print("LOADING TEST DATA")
        print("="*80)
        self._load_test_data()
    
    def _load_model(self):
        """Load trained student model"""
        print(f"\nModel path: {self.model_path}")
        
        # Check if this is a LoRA model or full fine-tuned
        if (self.model_path / "adapter_config.json").exists():
            print("✓ Detected LoRA adapter")
            # Load base model
            base_model_name = "google/gemma-3-12b-it"
            print(f"Loading base model: {base_model_name}")
            
            base_model = AutoModelForImageTextToText.from_pretrained(
                base_model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
            
            # Load LoRA adapter
            print(f"Loading LoRA adapter from: {self.model_path}")
            self.model = PeftModel.from_pretrained(base_model, str(self.model_path))
            self.model = self.model.merge_and_unload()  # Merge for faster inference
            
            self.processor = AutoProcessor.from_pretrained(self.model_path)
        else:
            print("✓ Detected full fine-tuned model")
            self.model = AutoModelForImageTextToText.from_pretrained(
                str(self.model_path),
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
            self.processor = AutoProcessor.from_pretrained(self.model_path)
        
        # Move to device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Model loaded on {self.device}")
    
    def _load_test_data(self):
        """Load test dataset"""
        with open(self.test_data_path, 'r') as f:
            self.test_data = json.load(f)
        
        print(f"✓ Loaded {len(self.test_data)} test examples")
        
        # Dataset breakdown
        from collections import defaultdict
        dataset_counts = defaultdict(int)
        for ex in self.test_data:
            dataset = ex['image_id'].split('_')[0]
            dataset_counts[dataset] += 1
        
        print("\nDataset breakdown:")
        for dataset, count in sorted(dataset_counts.items()):
            print(f"  {dataset.upper()}: {count} examples")
    
    def _get_image_path(self, example: Dict) -> Path:
        """Get full image path from example"""
        dataset = example['image_id'].split('_')[0]
        
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
    
    def _parse_model_output(self, text: str) -> Dict:
        """Parse model output into structured format"""
        import re
        
        result = {
            'cot_steps': [],
            'answer': '',
            'bbox': None,
            'raw_output': text
        }
        
        # Extract reasoning steps
        reasoning_match = re.search(r'REASONING:(.*?)(?=ANSWER:|$)', text, re.DOTALL | re.IGNORECASE)
        if reasoning_match:
            reasoning_text = reasoning_match.group(1).strip()
            steps = re.findall(r'Step \d+:\s*(.*?)(?=\n\nStep \d+:|\nStep \d+:|$)', reasoning_text, re.DOTALL)
            result['cot_steps'] = [step.strip() for step in steps if step.strip()]
        
        # Extract answer
        answer_match = re.search(r'ANSWER:\s*(.+?)(?=\nBBOX:|$)', text, re.DOTALL | re.IGNORECASE)
        if answer_match:
            result['answer'] = answer_match.group(1).strip()
        
        # Extract bbox
        bbox_match = re.search(r'BBOX:\s*\[([0-9., ]+)\]', text, re.IGNORECASE)
        if bbox_match:
            try:
                coords = [float(x.strip()) for x in bbox_match.group(1).split(',')]
                if len(coords) == 4:
                    result['bbox'] = coords
            except:
                pass
        
        return result
    
    def _compute_anls(self, pred: str, gt: str) -> float:
        """Compute Average Normalized Levenshtein Similarity"""
        if not pred and not gt:
            return 1.0
        if not pred or not gt:
            return 0.0
        
        pred = pred.lower().strip()
        gt = gt.lower().strip()
        
        if pred == gt:
            return 1.0
        
        # Levenshtein distance
        distance = self._levenshtein_distance(pred, gt)
        max_len = max(len(pred), len(gt))
        
        if max_len == 0:
            return 1.0
        
        anls = 1.0 - (distance / max_len)
        return max(0.0, anls)
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Compute Levenshtein edit distance"""
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
    
    def _compute_iou(self, bbox1: List, bbox2: List) -> float:
        """Compute Intersection over Union for bounding boxes"""
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
        
        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def infer_single(self, example: Dict) -> Dict:
        """Run inference on single example"""
        # Load image
        image_path = self._get_image_path(example)
        try:
            image = Image.open(image_path).convert('RGB')
            if max(image.size) > 1024:
                image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
        
        # Prepare input
        question = example['question']
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question}
            ]
        }]
        
        # Process
        try:
            inputs = self.processor.apply_chat_template(
                messages,
                images=[image],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                start_time = time.time()
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=False,  # Deterministic
                    temperature=None,
                    num_beams=1,
                )
                inference_time = time.time() - start_time
            
            # Decode
            generated_text = self.processor.decode(
                outputs[0][inputs['input_ids'].shape[-1]:],
                skip_special_tokens=True
            )
            
            # Parse output
            parsed = self._parse_model_output(generated_text)
            
            return {
                'success': True,
                'inference_time': inference_time,
                'generated_text': generated_text,
                'parsed': parsed
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def evaluate(self, output_dir: str, save_predictions: bool = True):
        """Run full evaluation on test set"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*80)
        print("PHASE C: EVALUATION")
        print("="*80)
        print(f"\nTest examples: {len(self.test_data)}")
        print(f"Output directory: {output_dir}")
        print(f"Device: {self.device}")
        print()
        
        results = []
        metrics = {
            'total': len(self.test_data),
            'successful': 0,
            'failed': 0,
            'total_time': 0,
            'anls_scores': [],
            'iou_scores': [],
            'cot_step_counts': [],
        }
        
        # Process each example
        for idx, example in enumerate(tqdm(self.test_data, desc="Evaluating")):
            result = {
                'image_id': example['image_id'],
                'question': example['question'],
                'answer_gt': example.get('answer_pred', ''),  # From teacher
                'bbox_gt': example.get('bbox_pred'),
            }
            
            # Inference
            inference_result = self.infer_single(example)
            
            if inference_result['success']:
                metrics['successful'] += 1
                metrics['total_time'] += inference_result['inference_time']
                
                parsed = inference_result['parsed']
                result.update({
                    'success': True,
                    'inference_time': inference_result['inference_time'],
                    'answer_pred': parsed['answer'],
                    'bbox_pred': parsed['bbox'],
                    'cot_steps': parsed['cot_steps'],
                    'raw_output': parsed['raw_output']
                })
                
                # Compute metrics
                anls = self._compute_anls(parsed['answer'], result['answer_gt'])
                result['anls'] = anls
                metrics['anls_scores'].append(anls)
                
                if parsed['bbox'] and result['bbox_gt']:
                    iou = self._compute_iou(parsed['bbox'], result['bbox_gt'])
                    result['iou'] = iou
                    metrics['iou_scores'].append(iou)
                else:
                    result['iou'] = 0.0
                    metrics['iou_scores'].append(0.0)
                
                metrics['cot_step_counts'].append(len(parsed['cot_steps']))
                
            else:
                metrics['failed'] += 1
                result.update({
                    'success': False,
                    'error': inference_result['error']
                })
            
            results.append(result)
        
        # Compute summary statistics
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80)
        
        print(f"\nSuccessful: {metrics['successful']}/{metrics['total']} ({metrics['successful']/metrics['total']*100:.1f}%)")
        print(f"Failed: {metrics['failed']}")
        
        if metrics['anls_scores']:
            avg_anls = np.mean(metrics['anls_scores'])
            print(f"\nAnswer Quality (ANLS): {avg_anls:.3f}")
        
        if metrics['iou_scores']:
            avg_iou = np.mean(metrics['iou_scores'])
            print(f"BBox Quality (IoU): {avg_iou:.3f}")
        
        if metrics['cot_step_counts']:
            avg_steps = np.mean(metrics['cot_step_counts'])
            print(f"Avg CoT Steps: {avg_steps:.1f}")
        
        if metrics['total_time'] > 0:
            avg_time = metrics['total_time'] / metrics['successful']
            throughput = metrics['successful'] / metrics['total_time']
            print(f"\nInference Time:")
            print(f"  Average: {avg_time:.2f}s per example")
            print(f"  Throughput: {throughput:.2f} examples/sec")
        
        # Dataset-specific metrics
        print("\n" + "="*80)
        print("DATASET-SPECIFIC METRICS")
        print("="*80)
        
        from collections import defaultdict
        dataset_metrics = defaultdict(lambda: {'anls': [], 'iou': []})
        
        for result in results:
            if result['success']:
                dataset = result['image_id'].split('_')[0]
                dataset_metrics[dataset]['anls'].append(result.get('anls', 0))
                dataset_metrics[dataset]['iou'].append(result.get('iou', 0))
        
        print(f"\n{'Dataset':<12} {'Count':<8} {'ANLS':<10} {'IoU':<10}")
        print("-" * 42)
        for dataset in sorted(dataset_metrics.keys()):
            dm = dataset_metrics[dataset]
            count = len(dm['anls'])
            anls = np.mean(dm['anls']) if dm['anls'] else 0
            iou = np.mean(dm['iou']) if dm['iou'] else 0
            print(f"{dataset.upper():<12} {count:<8} {anls:<10.3f} {iou:<10.3f}")
        
        # Save results
        if save_predictions:
            results_path = output_dir / "predictions.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n✓ Predictions saved: {results_path}")
        
        # Save summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'model_path': str(self.model_path),
            'test_data_path': str(self.test_data_path),
            'total_examples': metrics['total'],
            'successful': metrics['successful'],
            'failed': metrics['failed'],
            'avg_anls': float(np.mean(metrics['anls_scores'])) if metrics['anls_scores'] else 0,
            'avg_iou': float(np.mean(metrics['iou_scores'])) if metrics['iou_scores'] else 0,
            'avg_cot_steps': float(np.mean(metrics['cot_step_counts'])) if metrics['cot_step_counts'] else 0,
            'avg_inference_time': metrics['total_time'] / metrics['successful'] if metrics['successful'] > 0 else 0,
            'throughput': metrics['successful'] / metrics['total_time'] if metrics['total_time'] > 0 else 0,
            'dataset_metrics': {
                dataset: {
                    'count': len(dm['anls']),
                    'avg_anls': float(np.mean(dm['anls'])) if dm['anls'] else 0,
                    'avg_iou': float(np.mean(dm['iou'])) if dm['iou'] else 0
                }
                for dataset, dm in dataset_metrics.items()
            }
        }
        
        summary_path = output_dir / "evaluation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Summary saved: {summary_path}")
        
        print("\n" + "="*80)
        print("✓ EVALUATION COMPLETE")
        print("="*80 + "\n")
        
        return summary, results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase C: Evaluate trained student model")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--test-data", type=str, 
                       default="docval/data/phase_a_output/filtered/D_test.json")
    parser.add_argument("--image-dir", type=str, default="docval/data/cot_data")
    parser.add_argument("--output-dir", type=str, default="docval/results/phase_c")
    parser.add_argument("--device", type=str, default="mps", 
                       choices=["mps", "cuda", "cpu"])
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluator = PhaseCEvaluator(
        model_path=args.model,
        test_data_path=args.test_data,
        image_base_dir=args.image_dir,
        device=args.device
    )
    
    summary, results = evaluator.evaluate(
        output_dir=args.output_dir,
        save_predictions=True
    )


if __name__ == "__main__":
    main()

