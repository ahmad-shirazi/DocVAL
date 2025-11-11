"""Check Phase A progress"""
import json
from pathlib import Path

output_dir = Path('docval/data/phase_a_output')
data_dir = Path('docval/data/cot_data')

datasets = ['CORD', 'DocVQA', 'FUNSD', 'SROIE', 'VisualMRC']

print("="*70)
print("PHASE A PROGRESS CHECK")
print("="*70)

total_completed = 0
total_expected = 0

for dataset in datasets:
    dataset_lower = dataset.lower()
    output_file = output_dir / f"cot_{dataset_lower}.json"
    annotation_file = data_dir / dataset / 'annotations.json'
    
    completed = 0
    expected = 0
    
    if output_file.exists():
        with open(output_file) as f:
            data = json.load(f)
            completed = len(data)
    
    if annotation_file.exists():
        with open(annotation_file) as f:
            annotations = json.load(f)
            expected = len(annotations)
    
    total_completed += completed
    total_expected += expected
    
    progress_pct = (completed / expected * 100) if expected > 0 else 0
    status = "✓ Complete" if completed == expected else f"⏳ {progress_pct:.1f}%"
    
    print(f"\n{dataset:12s}: {completed:5d}/{expected:5d} {status}")
    
    if completed < expected:
        print(f"  Remaining: {expected - completed} examples")
        est_time_min = (expected - completed) * 20 / 60  # ~20 sec per example
        print(f"  Est. time: {est_time_min:.1f} minutes ({est_time_min/60:.1f} hours)")

print(f"\n{'='*70}")
print(f"TOTAL: {total_completed}/{total_expected} examples")
overall_pct = (total_completed / total_expected * 100) if total_expected > 0 else 0
print(f"Progress: {overall_pct:.1f}%")
print(f"{'='*70}")

if total_completed < total_expected:
    print(f"\n📝 To resume, run:")
    print(f"python -m docval.scripts.run_phase_a")
else:
    print(f"\n✅ All datasets complete!")

