"""
Display real-time metrics and throughput for Phase A processing
"""
import json
from pathlib import Path
from datetime import datetime, timedelta
import statistics

output_dir = Path('docval/data/phase_a_output')
datasets = ['cord', 'docvqa', 'funsd', 'sroie', 'visualmrc']

print("="*80)
print("PHASE A - LATENCY & THROUGHPUT METRICS")
print("="*80)

total_processed = 0
total_time = 0
all_timings = []

for dataset_name in datasets:
    metrics_file = output_dir / f"metrics_{dataset_name}.json"
    
    if not metrics_file.exists():
        continue
    
    with open(metrics_file) as f:
        metrics = json.load(f)
    
    print(f"\n{'='*80}")
    print(f"📊 {metrics['dataset'].upper()}")
    print(f"{'='*80}")
    
    # Basic info
    processed = metrics.get('processed_examples', 0)
    total = metrics.get('total_examples', 0)
    errors = metrics.get('errors', 0)
    
    print(f"\n📈 Progress:")
    print(f"  Processed: {processed}/{total} examples ({processed/total*100:.1f}%)" if total > 0 else "  No data")
    print(f"  Errors: {errors}")
    
    # Timing statistics
    timings = metrics.get('timings', [])
    if timings:
        successful_timings = [t for t in timings if t.get('success', True)]
        
        if successful_timings:
            total_times = [t['total_time'] for t in successful_timings]
            detection_times = [t.get('detection_time', 0) for t in successful_timings]
            generation_times = [t.get('generation_time', 0) for t in successful_timings]
            
            print(f"\n⏱️  Latency (seconds):")
            print(f"  Total per example:")
            print(f"    Average: {statistics.mean(total_times):.2f}s")
            print(f"    Median:  {statistics.median(total_times):.2f}s")
            print(f"    Min:     {min(total_times):.2f}s")
            print(f"    Max:     {max(total_times):.2f}s")
            
            print(f"\n  Text Detection (DB-ResNet):")
            print(f"    Average: {statistics.mean(detection_times):.2f}s")
            print(f"    Median:  {statistics.median(detection_times):.2f}s")
            
            print(f"\n  CoT Generation (Gemini 2.5 Pro):")
            print(f"    Average: {statistics.mean(generation_times):.2f}s")
            print(f"    Median:  {statistics.median(generation_times):.2f}s")
            
            # Throughput
            avg_time = statistics.mean(total_times)
            examples_per_hour = 3600 / avg_time if avg_time > 0 else 0
            examples_per_day = examples_per_hour * 24
            
            print(f"\n🚀 Throughput:")
            print(f"  Examples/hour: {examples_per_hour:.1f}")
            print(f"  Examples/day:  {examples_per_day:.1f}")
            
            # Time estimates
            remaining = total - processed
            if remaining > 0:
                est_hours = remaining * avg_time / 3600
                est_completion = datetime.now() + timedelta(hours=est_hours)
                
                print(f"\n📅 Estimates:")
                print(f"  Remaining: {remaining} examples")
                print(f"  Est. time: {est_hours:.1f} hours ({est_hours/24:.1f} days)")
                print(f"  Est. completion: {est_completion.strftime('%Y-%m-%d %H:%M')}")
            
            # Region statistics
            regions_counts = [t.get('num_regions', 0) for t in successful_timings]
            if regions_counts:
                print(f"\n📍 Text Regions Detected:")
                print(f"  Average: {statistics.mean(regions_counts):.1f} regions/doc")
                print(f"  Range: {min(regions_counts)}-{max(regions_counts)} regions")
            
            # CoT steps
            cot_steps = [t.get('num_cot_steps', 0) for t in successful_timings]
            if cot_steps:
                print(f"\n📝 CoT Steps Generated:")
                print(f"  Average: {statistics.mean(cot_steps):.1f} steps")
                correct_steps = sum(1 for s in cot_steps if s == 7)
                print(f"  With 7 steps: {correct_steps}/{len(cot_steps)} ({correct_steps/len(cot_steps)*100:.1f}%)")
            
            # Cost estimates
            cost_per_example = 0.005  # ~$0.005 per example
            total_cost = processed * cost_per_example
            remaining_cost = remaining * cost_per_example
            
            print(f"\n💰 Cost Estimates:")
            print(f"  Processed: ${total_cost:.2f}")
            print(f"  Remaining: ${remaining_cost:.2f}")
            print(f"  Total est: ${(processed + remaining) * cost_per_example:.2f}")
            
            total_processed += processed
            total_time += sum(total_times)
            all_timings.extend(total_times)

# Overall summary
if all_timings:
    print(f"\n{'='*80}")
    print(f"📊 OVERALL SUMMARY")
    print(f"{'='*80}")
    print(f"\n  Total processed: {total_processed} examples")
    print(f"  Average time: {statistics.mean(all_timings):.2f}s per example")
    print(f"  Total time spent: {total_time/3600:.1f} hours")
    print(f"  Overall throughput: {3600/statistics.mean(all_timings):.1f} examples/hour")
    print(f"  Total cost: ${total_processed * 0.005:.2f}")

print(f"\n{'='*80}")

