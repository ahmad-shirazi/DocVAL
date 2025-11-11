#!/usr/bin/env python3
"""
Apply VAL Filter to Merged CoT Dataset
Phase A - Binary Mode Quality Assessment

Input: cot_merged_all.json (8,250 examples)
Output: Filtered dataset split into D3/D4/D_test

Q = 0.4*Q_ans + 0.4*Q_bbox + 0.2*Q_reason
Accept if Q > 0.85
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from docval.validation.val_filter import VALFilter


def apply_val_filter(input_file: Path, output_dir: Path, threshold: float = 0.85):
    """
    Apply VAL filter to merged CoT dataset.
    
    Args:
        input_file: Path to merged CoT JSON file
        output_dir: Directory to save filtered results
        threshold: Quality threshold (default 0.85)
    """
    print("\n" + "="*80)
    print("VAL FILTER - Phase A Binary Mode")
    print("="*80)
    print(f"\nInput: {input_file}")
    print(f"Threshold: Q > {threshold}")
    print(f"Formula: Q = 0.4*Q_ans + 0.4*Q_bbox + 0.2*Q_reason\n")
    
    # Load merged dataset
    print("Loading merged dataset...")
    with open(input_file, 'r') as f:
        all_examples = json.load(f)
    print(f"✓ Loaded {len(all_examples):,} examples")
    
    # Initialize VAL filter
    val_filter = VALFilter(threshold=threshold)
    
    # Filter examples
    print(f"\nApplying VAL filter (threshold={threshold})...")
    print("Modules: OCR Grounding | Answer (ANLS) | BBox (IoU) | Reasoning | Overall Q")
    print("-" * 80)
    
    accepted_examples = []
    rejected_examples = []
    detailed_results = []
    
    for i, example in enumerate(all_examples):
        accept, quality_score, detailed_scores = val_filter.filter_example(example)
        
        # Add quality scores to example
        example_with_scores = example.copy()
        example_with_scores['val_scores'] = detailed_scores
        example_with_scores['val_accept'] = accept
        
        if accept:
            accepted_examples.append(example_with_scores)
        else:
            rejected_examples.append(example_with_scores)
        
        detailed_results.append({
            'image_id': example['image_id'],
            'accept': accept,
            'quality_score': quality_score,
            'scores': detailed_scores
        })
        
        # Progress update
        if (i + 1) % 1000 == 0:
            acceptance_rate = len(accepted_examples) / (i + 1)
            print(f"  Processed {i+1:,}/{len(all_examples):,} | "
                  f"Accepted: {len(accepted_examples):,} ({acceptance_rate:.1%}) | "
                  f"Avg Q: {np.mean([r['quality_score'] for r in detailed_results[-1000:]]):.3f}")
    
    # Final statistics
    print("\n" + "="*80)
    print("FILTERING RESULTS")
    print("="*80)
    
    acceptance_rate = len(accepted_examples) / len(all_examples)
    rejection_rate = len(rejected_examples) / len(all_examples)
    
    print(f"\nTotal examples: {len(all_examples):,}")
    print(f"✓ Accepted (Q > {threshold}): {len(accepted_examples):,} ({acceptance_rate:.1%})")
    print(f"✗ Rejected (Q ≤ {threshold}): {len(rejected_examples):,} ({rejection_rate:.1%})")
    
    # Module statistics
    print("\n" + "="*80)
    print("MODULE STATISTICS")
    print("="*80)
    
    module_stats = {
        'ocr_grounding': [],
        'answer_quality': [],
        'bbox_quality': [],
        'reasoning_quality': [],
        'overall': []
    }
    
    for result in detailed_results:
        for module, score in result['scores'].items():
            if module in module_stats:
                module_stats[module].append(score)
    
    print(f"\n{'Module':<20} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-" * 60)
    for module, scores in module_stats.items():
        arr = np.array(scores)
        print(f"{module:<20} {np.mean(arr):<10.3f} {np.std(arr):<10.3f} "
              f"{np.min(arr):<10.3f} {np.max(arr):<10.3f}")
    
    # Dataset breakdown
    print("\n" + "="*80)
    print("ACCEPTANCE BY DATASET")
    print("="*80)
    
    dataset_stats = defaultdict(lambda: {'total': 0, 'accepted': 0})
    for result in detailed_results:
        dataset = result['image_id'].split('_')[0]
        dataset_stats[dataset]['total'] += 1
        if result['accept']:
            dataset_stats[dataset]['accepted'] += 1
    
    print(f"\n{'Dataset':<12} {'Total':<10} {'Accepted':<10} {'Rate':<10}")
    print("-" * 42)
    for dataset in sorted(dataset_stats.keys()):
        stats = dataset_stats[dataset]
        rate = stats['accepted'] / stats['total'] if stats['total'] > 0 else 0
        print(f"{dataset.upper():<12} {stats['total']:<10,} {stats['accepted']:<10,} {rate:<10.1%}")
    
    # Save filtered datasets
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("SAVING FILTERED DATA")
    print("="*80)
    
    # Save accepted examples
    accepted_path = output_dir / "cot_filtered_accepted.json"
    with open(accepted_path, 'w') as f:
        json.dump(accepted_examples, f, indent=2)
    print(f"\n✓ Saved accepted examples: {accepted_path}")
    print(f"  Count: {len(accepted_examples):,}")
    
    # Save rejected examples (for analysis)
    rejected_path = output_dir / "cot_filtered_rejected.json"
    with open(rejected_path, 'w') as f:
        json.dump(rejected_examples, f, indent=2)
    print(f"\n✓ Saved rejected examples: {rejected_path}")
    print(f"  Count: {len(rejected_examples):,}")
    
    # Save detailed results
    results_path = output_dir / "val_filter_results.json"
    with open(results_path, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    print(f"\n✓ Saved detailed results: {results_path}")
    
    # Save statistics
    stats = {
        'timestamp': datetime.now().isoformat(),
        'input_file': str(input_file),
        'threshold': threshold,
        'total_examples': len(all_examples),
        'accepted': len(accepted_examples),
        'rejected': len(rejected_examples),
        'acceptance_rate': acceptance_rate,
        'module_statistics': {
            module: {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores))
            }
            for module, scores in module_stats.items()
        },
        'dataset_breakdown': {
            dataset: {
                'total': stats['total'],
                'accepted': stats['accepted'],
                'rate': stats['accepted'] / stats['total'] if stats['total'] > 0 else 0
            }
            for dataset, stats in dataset_stats.items()
        }
    }
    
    stats_path = output_dir / "val_filter_statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\n✓ Saved statistics: {stats_path}")
    
    print("\n" + "="*80)
    print("NEXT STEP: SPLIT INTO TRAIN/VAL/TEST")
    print("="*80)
    print(f"\nAccepted examples will be split into:")
    print(f"  D3 (train):  70% → {int(len(accepted_examples) * 0.7):,} examples")
    print(f"  D4 (val):    15% → {int(len(accepted_examples) * 0.15):,} examples")
    print(f"  D_test:      15% → {int(len(accepted_examples) * 0.15):,} examples")
    
    return accepted_examples, rejected_examples, stats


def split_dataset(examples: list, output_dir: Path, 
                  train_ratio: float = 0.7, 
                  val_ratio: float = 0.15,
                  test_ratio: float = 0.15,
                  seed: int = 42):
    """
    Split filtered dataset into D3 (train), D4 (val), and D_test.
    
    Args:
        examples: Filtered accepted examples
        output_dir: Directory to save splits
        train_ratio: Ratio for training set (D3)
        val_ratio: Ratio for validation set (D4)
        test_ratio: Ratio for test set (D_test)
        seed: Random seed for reproducibility
    """
    print("\n" + "="*80)
    print("SPLITTING DATASET INTO D3 / D4 / D_test")
    print("="*80)
    
    np.random.seed(seed)
    
    # Shuffle examples
    indices = np.random.permutation(len(examples))
    shuffled = [examples[i] for i in indices]
    
    # Calculate split points
    n_train = int(len(examples) * train_ratio)
    n_val = int(len(examples) * val_ratio)
    
    # Split
    d3_train = shuffled[:n_train]
    d4_val = shuffled[n_train:n_train + n_val]
    d_test = shuffled[n_train + n_val:]
    
    print(f"\nTotal examples: {len(examples):,}")
    print(f"  D3 (train):  {len(d3_train):,} ({len(d3_train)/len(examples):.1%})")
    print(f"  D4 (val):    {len(d4_val):,} ({len(d4_val)/len(examples):.1%})")
    print(f"  D_test:      {len(d_test):,} ({len(d_test)/len(examples):.1%})")
    
    # Save splits
    d3_path = output_dir / "D3_train.json"
    d4_path = output_dir / "D4_val.json"
    dtest_path = output_dir / "D_test.json"
    
    with open(d3_path, 'w') as f:
        json.dump(d3_train, f, indent=2)
    print(f"\n✓ Saved D3 (train): {d3_path}")
    
    with open(d4_path, 'w') as f:
        json.dump(d4_val, f, indent=2)
    print(f"✓ Saved D4 (val): {d4_path}")
    
    with open(dtest_path, 'w') as f:
        json.dump(d_test, f, indent=2)
    print(f"✓ Saved D_test: {dtest_path}")
    
    # Dataset composition statistics
    print("\n" + "="*80)
    print("DATASET COMPOSITION")
    print("="*80)
    
    for split_name, split_data in [('D3 (train)', d3_train), 
                                    ('D4 (val)', d4_val), 
                                    ('D_test', d_test)]:
        print(f"\n{split_name}:")
        dataset_counts = defaultdict(int)
        for ex in split_data:
            dataset = ex['image_id'].split('_')[0]
            dataset_counts[dataset] += 1
        
        for dataset in sorted(dataset_counts.keys()):
            count = dataset_counts[dataset]
            ratio = count / len(split_data) if split_data else 0
            print(f"  {dataset.upper():<12}: {count:>5,} ({ratio:>6.1%})")
    
    return d3_train, d4_val, d_test


if __name__ == "__main__":
    # Paths
    input_file = Path("/Users/ahmadshirazi/Desktop/DocVAL/docval/data/phase_a_output/cot_merged_all.json")
    output_dir = Path("/Users/ahmadshirazi/Desktop/DocVAL/docval/data/phase_a_output/filtered")
    
    # Apply VAL filter
    accepted, rejected, stats = apply_val_filter(input_file, output_dir, threshold=0.85)
    
    # Split into D3/D4/D_test
    if accepted:
        d3_train, d4_val, d_test = split_dataset(accepted, output_dir)
        
        print("\n" + "="*80)
        print("✓ VAL FILTER COMPLETE")
        print("="*80)
        print(f"\nFiltered data ready for Phase B training!")
        print(f"\nOutput directory: {output_dir}")
        print("\nFiles created:")
        print("  • cot_filtered_accepted.json - All accepted examples")
        print("  • cot_filtered_rejected.json - All rejected examples")
        print("  • D3_train.json - Training set")
        print("  • D4_val.json - Validation set")
        print("  • D_test.json - Test set")
        print("  • val_filter_results.json - Detailed scores")
        print("  • val_filter_statistics.json - Summary statistics")
        print("\n" + "="*80)
    else:
        print("\n⚠ Warning: No examples passed the filter!")

