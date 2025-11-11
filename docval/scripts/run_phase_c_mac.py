#!/usr/bin/env python3
"""
Phase C: Quick Inference on Mac
Evaluate trained student model on test set
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from docval.inference.evaluate_phase_c import PhaseCEvaluator


def main():
    print("\n" + "="*80)
    print("PHASE C: STUDENT MODEL EVALUATION")
    print("="*80)
    print("\n🍎 Running on Mac M4 Max")
    print("📊 Evaluating on D_test.json (355 examples)")
    print("🎯 Pure VLM inference (no text detector)")
    print()
    
    # Default paths
    model_path = "docval/models/student_b1_mac/final"
    test_data = "docval/data/phase_a_output/filtered/D_test.json"
    image_dir = "docval/data/cot_data"
    output_dir = "docval/results/phase_c_mac"
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        print("\nAlternative model paths to try:")
        print("  • docval/models/student_b1/final")
        print("  • docval/models/student_b2/final")
        print()
        model_path = input("Enter model path: ").strip()
        
        if not Path(model_path).exists():
            print(f"❌ Model still not found: {model_path}")
            return
    
    print(f"✓ Using model: {model_path}")
    print(f"✓ Test data: {test_data}")
    print(f"✓ Output: {output_dir}")
    print()
    
    # Confirm
    response = input("Start evaluation? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    # Create evaluator
    evaluator = PhaseCEvaluator(
        model_path=model_path,
        test_data_path=test_data,
        image_base_dir=image_dir,
        device="mps"
    )
    
    # Run evaluation
    summary, results = evaluator.evaluate(
        output_dir=output_dir,
        save_predictions=True
    )
    
    # Display results
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"\n📊 Overall Performance:")
    print(f"  Answer Quality (ANLS): {summary['avg_anls']:.3f}")
    print(f"  BBox Quality (IoU):    {summary['avg_iou']:.3f}")
    print(f"  Success Rate:          {summary['successful']}/{summary['total_examples']} ({summary['successful']/summary['total_examples']*100:.1f}%)")
    print(f"\n⏱️  Performance:")
    print(f"  Avg Time:    {summary['avg_inference_time']:.2f}s per example")
    print(f"  Throughput:  {summary['throughput']:.2f} examples/sec")
    print(f"  Total Time:  {summary['successful'] * summary['avg_inference_time'] / 60:.1f} minutes")
    print(f"\n📁 Results saved to: {output_dir}")
    print("  • predictions.json        - All predictions")
    print("  • evaluation_summary.json - Summary statistics")
    print()


if __name__ == "__main__":
    main()

