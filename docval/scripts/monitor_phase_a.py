"""
Real-time monitoring dashboard for Phase A
Run this in a separate terminal while Phase A is processing
"""
import json
from pathlib import Path
import time
import os
from datetime import datetime

def clear_screen():
    os.system('clear' if os.name != 'nt' else 'cls')

def monitor():
    output_dir = Path('docval/data/phase_a_output')
    datasets = ['CORD', 'DocVQA', 'FUNSD', 'SROIE', 'VisualMRC']
    
    while True:
        clear_screen()
        print("="*80)
        print(f"🔴 PHASE A - LIVE MONITORING  ({datetime.now().strftime('%H:%M:%S')})")
        print("="*80)
        print("Press Ctrl+C to stop monitoring\n")
        
        total_processed = 0
        total_expected = 0
        any_active = False
        
        for dataset in datasets:
            dataset_lower = dataset.lower()
            
            # Check annotations
            ann_file = Path(f'docval/data/cot_data/{dataset}/annotations.json')
            expected = 0
            if ann_file.exists():
                with open(ann_file) as f:
                    expected = len(json.load(f))
            
            # Check results
            output_file = output_dir / f"cot_{dataset_lower}.json"
            processed = 0
            if output_file.exists():
                with open(output_file) as f:
                    processed = len(json.load(f))
            
            # Check metrics
            metrics_file = output_dir / f"metrics_{dataset_lower}.json"
            speed = "N/A"
            eta = "N/A"
            
            if metrics_file.exists():
                with open(metrics_file) as f:
                    metrics = json.load(f)
                    summary = metrics.get('summary', {})
                    speed = f"{summary.get('examples_per_hour', 0):.1f}/hr"
                    eta_hours = summary.get('estimated_remaining_hours', 0)
                    if eta_hours > 0:
                        eta = f"{eta_hours:.1f}h"
                        any_active = True
            
            total_processed += processed
            total_expected += expected
            
            # Display
            progress_pct = (processed / expected * 100) if expected > 0 else 0
            bar_length = 30
            filled = int(bar_length * progress_pct / 100)
            bar = '█' * filled + '░' * (bar_length - filled)
            
            status = "🟢" if processed == expected else "🔴" if any_active else "⚪"
            print(f"{status} {dataset:12s} [{bar}] {progress_pct:5.1f}% | {processed:5d}/{expected:5d} | {speed:>10s} | ETA: {eta:>8s}")
        
        # Overall stats
        overall_pct = (total_processed / total_expected * 100) if total_expected > 0 else 0
        print(f"\n{'='*80}")
        print(f"TOTAL: {total_processed}/{total_expected} ({overall_pct:.1f}%)")
        print(f"{'='*80}")
        
        if not any_active and total_processed < total_expected:
            print("\n⚠️  No active processing detected. Start with:")
            print("   python -m docval.scripts.run_phase_a")
        elif total_processed == total_expected:
            print("\n✅ All datasets complete!")
            break
        
        time.sleep(5)  # Update every 5 seconds

if __name__ == "__main__":
    try:
        monitor()
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")

