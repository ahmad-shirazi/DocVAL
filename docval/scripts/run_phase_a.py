"""
Script to run Phase A: Teacher data generation with Gemini 2.5 Pro and DB-ResNet.
Processes CORD, DocVQA, and FUNSD datasets.
"""
import os
import sys
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse
import time
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.teacher_vlm import TeacherVLM
from models.text_detector import TextDetector
from config.hyperparameters import DocVALConfig


def load_dataset_annotations(data_dir: str, dataset_name: str):
    """
    Load annotations for a specific dataset.
    
    Expected structure:
    data_dir/
        CORD/
            images/
            annotations.json
        DocVQA/
            images/
            annotations.json
        FUNSD/
            images/
            annotations.json
    """
    dataset_path = Path(data_dir) / dataset_name
    ann_file = dataset_path / "annotations.json"
    
    if not ann_file.exists():
        print(f"Warning: {ann_file} not found")
        return []
    
    with open(ann_file, 'r') as f:
        annotations = json.load(f)
    
    return annotations


def process_dataset(
    dataset_name: str,
    data_dir: str,
    teacher_model: TeacherVLM,
    text_detector: TextDetector,
    output_dir: str,
    max_examples: int = None
):
    """
    Process a single dataset with Phase A pipeline.
    
    Steps:
    1. Load dataset annotations
    2. Run text detection (DB-ResNet)
    3. Generate CoT with Gemini 2.5 Pro
    4. Save results
    """
    print(f"\n{'='*60}")
    print(f"Processing Dataset: {dataset_name}")
    print(f"{'='*60}")
    
    # Load annotations
    annotations = load_dataset_annotations(data_dir, dataset_name)
    
    if not annotations:
        print(f"No annotations found for {dataset_name}, skipping...")
        return []
    
    # Limit examples if specified
    if max_examples:
        annotations = annotations[:max_examples]
    
    print(f"Found {len(annotations)} examples")
    
    # Output file
    output_path = Path(output_dir) / f"cot_{dataset_name.lower()}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing results if file exists (for resuming)
    results = []
    processed_image_ids = set()
    
    if output_path.exists():
        try:
            with open(output_path, 'r') as f:
                results = json.load(f)
            processed_image_ids = {r['image_id'] for r in results}
            print(f"Found existing {len(results)} results, will resume from there...")
        except Exception as e:
            print(f"Warning: Could not load existing results: {e}")
            results = []
    
    # Initialize metrics tracking
    metrics_path = output_path.parent / f"metrics_{dataset_name.lower()}.json"
    metrics = {
        'dataset': dataset_name,
        'start_time': datetime.now().isoformat(),
        'total_examples': len(annotations),
        'processed_examples': len(results),
        'timings': [],
        'errors': 0
    }
    
    # Filter out already processed examples
    remaining_annotations = [ann for ann in annotations if ann['image_id'] not in processed_image_ids]
    
    if len(remaining_annotations) < len(annotations):
        print(f"Skipping {len(annotations) - len(remaining_annotations)} already processed examples")
    
    # Process each example
    for idx, ann in enumerate(tqdm(remaining_annotations, desc=f"  {dataset_name}")):
        example_start = time.time()
        timing_data = {
            'example_id': ann['image_id'],
            'index': len(results)
        }
        
        try:
            # Load image - try multiple possible locations
            image_load_start = time.time()
            image_file = ann['image_file']
            possible_paths = [
                Path(data_dir) / dataset_name / "images" / image_file,  # DocVQA
                Path(data_dir) / dataset_name / image_file,  # CORD, FUNSD
                Path(data_dir) / dataset_name / image_file,  # VisualMRC (saved by prepare script)
                Path(data_dir) / dataset_name / "SROIE2019" / "train" / "img" / image_file,  # SROIE train
                Path(data_dir) / dataset_name / "SROIE2019" / "test" / "img" / image_file,  # SROIE test
            ]
            
            image_path = None
            for path in possible_paths:
                if path.exists():
                    image_path = path
                    break
            
            if not image_path:
                print(f"    Warning: Image not found: {image_file}")
                continue
            
            image = Image.open(image_path).convert('RGB')
            timing_data['image_load_time'] = time.time() - image_load_start
            
            # Step 1: Run text detection (DB-ResNet)
            detection_start = time.time()
            regions = text_detector.detect(image, return_text=True)
            timing_data['detection_time'] = time.time() - detection_start
            timing_data['num_regions'] = len(regions)
            
            # Step 2: Generate CoT with Gemini 2.5 Pro
            generation_start = time.time()
            output = teacher_model.generate_cot(
                image=image,
                question=ann['question'],
                regions=regions,
                answer_gt=ann.get('answer'),
                bbox_gt=ann.get('bbox')
            )
            timing_data['generation_time'] = time.time() - generation_start
            timing_data['num_cot_steps'] = len(output.get('cot_steps', []))
            
            # Save result
            result = {
                'dataset': dataset_name,
                'image_id': ann.get('image_id', f'{dataset_name}_{idx}'),
                'image_file': ann['image_file'],
                'question': ann['question'],
                'answer_gt': ann.get('answer'),
                'bbox_gt': ann.get('bbox'),
                'cot_steps': output['cot_steps'],
                'answer_pred': output['answer'],
                'bbox_pred': output['bbox'],
                'regions': regions,
                'raw_output': output['raw_output']
            }
            
            results.append(result)
            
            # Record total timing
            timing_data['total_time'] = time.time() - example_start
            timing_data['timestamp'] = datetime.now().isoformat()
            timing_data['success'] = True
            metrics['timings'].append(timing_data)
            
            # Save incrementally every 10 examples (checkpoint)
            if len(results) % 10 == 0:
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2)
                # Save metrics
                _save_metrics(metrics, metrics_path, len(results), len(annotations))
                    
            # Also save every 100 examples with backup
            if len(results) % 100 == 0:
                backup_path = output_path.parent / f"{output_path.stem}_backup.json"
                with open(backup_path, 'w') as f:
                    json.dump(results, f, indent=2)
            
        except Exception as e:
            print(f"    Error processing example {idx}: {e}")
            timing_data['total_time'] = time.time() - example_start
            timing_data['error'] = str(e)
            timing_data['success'] = False
            metrics['timings'].append(timing_data)
            metrics['errors'] += 1
            continue
    
    # Final save
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Final metrics save
    metrics['end_time'] = datetime.now().isoformat()
    metrics['processed_examples'] = len(results)
    _save_metrics(metrics, metrics_path, len(results), len(annotations))
    
    print(f"\n✓ Completed {dataset_name}")
    print(f"  Total results: {len(results)} examples")
    print(f"  Newly processed: {len(remaining_annotations)} examples")
    print(f"  Saved to: {output_path}")
    print(f"  Metrics: {metrics_path}")
    
    return results


def _save_metrics(metrics, metrics_path, current, total):
    """Save metrics with throughput calculations"""
    if metrics['timings']:
        recent_timings = metrics['timings'][-100:]  # Last 100 examples
        avg_time = sum(t.get('total_time', 0) for t in recent_timings) / len(recent_timings)
        avg_detection = sum(t.get('detection_time', 0) for t in recent_timings) / len(recent_timings)
        avg_generation = sum(t.get('generation_time', 0) for t in recent_timings) / len(recent_timings)
        
        metrics['summary'] = {
            'avg_time_per_example': round(avg_time, 2),
            'avg_detection_time': round(avg_detection, 2),
            'avg_generation_time': round(avg_generation, 2),
            'examples_per_hour': round(3600 / avg_time if avg_time > 0 else 0, 1),
            'progress_percent': round(current / total * 100, 2) if total > 0 else 0,
            'estimated_remaining_hours': round((total - current) * avg_time / 3600, 2) if avg_time > 0 else 0
        }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Phase A: Generate CoT data with Gemini 2.5 Pro + DB-ResNet"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/Users/ahmadshirazi/Desktop/DocVAL/docval/data/cot_data",
        help="Root directory containing datasets"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/Users/ahmadshirazi/Desktop/DocVAL/docval/data/phase_a_output",
        help="Output directory for CoT data"
    )
    parser.add_argument(
        "--datasets",
        nargs='+',
        default=["CORD", "DocVQA", "FUNSD", "SROIE", "VisualMRC"],
        help="Datasets to process"
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum examples per dataset (for testing)"
    )
    parser.add_argument(
        "--teacher-model",
        type=str,
        default="gemini-2.5-pro",
        help="Teacher model name"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("PHASE A: Teacher Data Generation")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Data Directory: {args.data_dir}")
    print(f"  Output Directory: {args.output_dir}")
    print(f"  Datasets: {args.datasets}")
    print(f"  Teacher Model: {args.teacher_model}")
    print(f"  Max Examples: {args.max_examples or 'All'}")
    
    # Initialize models
    print(f"\n{'='*60}")
    print("Initializing Models")
    print(f"{'='*60}")
    
    # 1. Initialize Text Detector (DB-ResNet)
    print("\n[1/2] Loading DB-ResNet text detector...")
    text_detector = TextDetector(detector_name="db_resnet")
    
    # 2. Initialize Teacher Model (Gemini 2.5 Pro)
    print("\n[2/2] Loading Gemini 2.5 Pro teacher model...")
    teacher_model = TeacherVLM(
        model_name=args.teacher_model,
        temperature=None,  # Thinking model
        max_tokens=8192
    )
    
    print("\n✓ Models initialized successfully!")
    
    # Process each dataset
    all_results = {}
    for dataset_name in args.datasets:
        results = process_dataset(
            dataset_name=dataset_name,
            data_dir=args.data_dir,
            teacher_model=teacher_model,
            text_detector=text_detector,
            output_dir=args.output_dir,
            max_examples=args.max_examples
        )
        all_results[dataset_name] = results
    
    # Summary
    print(f"\n{'='*60}")
    print("PHASE A COMPLETE - Summary")
    print(f"{'='*60}")
    
    total_examples = sum(len(results) for results in all_results.values())
    print(f"\nTotal examples processed: {total_examples}")
    
    for dataset_name, results in all_results.items():
        print(f"  {dataset_name}: {len(results)} examples")
    
    print(f"\nOutput directory: {args.output_dir}")
    print("\n✓ Phase A completed successfully!")


if __name__ == "__main__":
    main()

