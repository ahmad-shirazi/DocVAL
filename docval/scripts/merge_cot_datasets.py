#!/usr/bin/env python3
"""
Merge all CoT dataset JSON files into one combined file.
"""

import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

def load_cot_file(file_path):
    """Load a CoT JSON file"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f"✓ Loaded {file_path.name}: {len(data)} examples")
        return data
    except Exception as e:
        print(f"✗ Error loading {file_path.name}: {e}")
        return []

def validate_cot_entry(entry, dataset_name):
    """Validate a single CoT entry"""
    # Required fields (actual field names from generated files)
    required_fields = ['image_id', 'image_file', 'question', 'answer_pred', 'bbox_pred', 'cot_steps', 'regions']
    missing_fields = [f for f in required_fields if f not in entry]
    
    if missing_fields:
        return False, f"Missing fields: {missing_fields}"
    
    # Check CoT steps
    if not isinstance(entry['cot_steps'], list):
        return False, "cot_steps is not a list"
    
    if len(entry['cot_steps']) != 7:
        return False, f"Expected 7 CoT steps, got {len(entry['cot_steps'])}"
    
    # Check dataset prefix in image_id
    dataset_prefix = dataset_name.lower().replace('docvqa', 'docvqa').replace('visualmrc', 'visualmrc')
    if not entry['image_id'].startswith(dataset_prefix):
        return False, f"image_id doesn't start with {dataset_prefix}"
    
    return True, "OK"

def merge_cot_datasets(output_dir):
    """Merge all CoT datasets into one file"""
    output_dir = Path(output_dir)
    
    # Dataset files to merge
    datasets = {
        'CORD': 'cot_cord.json',
        'DocVQA': 'cot_docvqa.json',
        'FUNSD': 'cot_funsd.json',
        'SROIE': 'cot_sroie.json',
        'VisualMRC': 'cot_visualmrc.json'
    }
    
    print("\n" + "="*80)
    print("MERGING CoT DATASETS")
    print("="*80 + "\n")
    
    all_data = []
    stats = defaultdict(int)
    validation_errors = []
    
    # Load and validate each dataset
    for dataset_name, filename in datasets.items():
        file_path = output_dir / filename
        
        if not file_path.exists():
            print(f"⚠ Warning: {filename} not found, skipping...")
            continue
        
        data = load_cot_file(file_path)
        
        # Validate entries
        valid_count = 0
        for entry in data:
            is_valid, msg = validate_cot_entry(entry, dataset_name)
            if is_valid:
                valid_count += 1
                all_data.append(entry)
            else:
                validation_errors.append({
                    'dataset': dataset_name,
                    'image_id': entry.get('image_id', 'unknown'),
                    'error': msg
                })
        
        stats[dataset_name] = {
            'total': len(data),
            'valid': valid_count,
            'invalid': len(data) - valid_count
        }
    
    # Save merged file
    output_path = output_dir / 'cot_merged_all.json'
    with open(output_path, 'w') as f:
        json.dump(all_data, f, indent=2)
    
    print(f"\n✓ Saved merged file: {output_path}")
    print(f"  Total examples: {len(all_data)}")
    
    # Save detailed statistics
    stats_output = {
        'merge_timestamp': datetime.now().isoformat(),
        'total_examples': len(all_data),
        'datasets': stats,
        'validation_errors': validation_errors[:100]  # First 100 errors
    }
    
    stats_path = output_dir / 'merge_statistics.json'
    with open(stats_path, 'w') as f:
        json.dump(stats_output, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("MERGE SUMMARY")
    print("="*80)
    
    for dataset_name, dataset_stats in stats.items():
        print(f"\n{dataset_name}:")
        print(f"  Total:   {dataset_stats['total']:,}")
        print(f"  Valid:   {dataset_stats['valid']:,}")
        if dataset_stats['invalid'] > 0:
            print(f"  Invalid: {dataset_stats['invalid']:,} ⚠")
    
    print(f"\n{'='*80}")
    print(f"COMBINED TOTAL: {len(all_data):,} examples")
    print(f"{'='*80}")
    
    if validation_errors:
        print(f"\n⚠ Found {len(validation_errors)} validation errors")
        print(f"  See {stats_path} for details")
    
    # Check for duplicate image_ids
    image_ids = [entry['image_id'] for entry in all_data]
    duplicates = [iid for iid in set(image_ids) if image_ids.count(iid) > 1]
    
    if duplicates:
        print(f"\n⚠ Warning: Found {len(duplicates)} duplicate image_ids:")
        for dup in duplicates[:10]:
            print(f"    - {dup}")
    else:
        print(f"\n✓ No duplicate image_ids found")
    
    # Sample structure check
    print("\n" + "="*80)
    print("SAMPLE ENTRY (first example):")
    print("="*80)
    if all_data:
        sample = all_data[0]
        print(f"\nDataset: {sample.get('dataset', 'N/A')}")
        print(f"Image ID: {sample['image_id']}")
        print(f"Image File: {sample['image_file']}")
        print(f"Question: {sample['question'][:80]}...")
        print(f"Answer (GT): {sample.get('answer_gt', 'N/A')}")
        print(f"Answer (Pred): {sample['answer_pred']}")
        print(f"Bbox (GT): {sample.get('bbox_gt', 'N/A')}")
        print(f"Bbox (Pred): {sample['bbox_pred']}")
        print(f"Regions: {len(sample['regions'])} detected regions")
        print(f"CoT Steps: {len(sample['cot_steps'])} steps")
        print("\nCoT Steps Preview:")
        for i, step in enumerate(sample['cot_steps'][:3], 1):
            print(f"  Step {i}: {step[:80]}...")
    
    print("\n" + "="*80)
    print(f"✓ Merge complete! Output: {output_path}")
    print("="*80 + "\n")
    
    return output_path

if __name__ == "__main__":
    output_dir = Path("/Users/ahmadshirazi/Desktop/DocVAL/docval/data/phase_a_output")
    merge_cot_datasets(output_dir)

