"""
Quick test script for Phase A components.
Tests DB-ResNet and Gemini 2.5 Pro integration.
"""
import sys
import os
from PIL import Image
from pathlib import Path

# Add docval to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'docval'))

from models.text_detector import TextDetector
from models.teacher_vlm import TeacherVLM


def test_text_detector():
    """Test DB-ResNet text detection"""
    print("="*60)
    print("Testing DB-ResNet Text Detector")
    print("="*60)
    
    # Initialize detector
    detector = TextDetector(detector_name="db_resnet")
    
    # Create a dummy image
    dummy_image = Image.new('RGB', (800, 600), color='white')
    
    # Test detection
    print("\nRunning detection on dummy image...")
    regions = detector.detect(dummy_image, return_text=True)
    
    print(f"✓ Detected {len(regions)} regions")
    if regions:
        print(f"  Sample region: {regions[0]}")
    
    return detector


def test_teacher_model():
    """Test Gemini 2.5 Pro teacher model"""
    print("\n" + "="*60)
    print("Testing Gemini 2.5 Pro Teacher Model")
    print("="*60)
    
    # Initialize teacher
    teacher = TeacherVLM(
        model_name="gemini-2.5-pro",
        temperature=None,
        max_tokens=8192
    )
    
    print("✓ Teacher model initialized")
    
    return teacher


def test_full_pipeline():
    """Test full Phase A pipeline"""
    print("\n" + "="*60)
    print("Testing Full Phase A Pipeline")
    print("="*60)
    
    # Initialize models
    detector = TextDetector(detector_name="db_resnet")
    teacher = TeacherVLM(
        model_name="gemini-2.5-pro",
        temperature=None,
        max_tokens=8192
    )
    
    # Create a test image with some text
    from PIL import ImageDraw, ImageFont
    img = Image.new('RGB', (800, 600), color='white')
    draw = ImageDraw.Draw(img)
    
    # Add some text
    draw.text((50, 50), "TOTAL: $108.51", fill='black')
    draw.text((50, 100), "Date: 2024-01-15", fill='black')
    
    # Step 1: Detect text
    print("\n[1/2] Running text detection...")
    regions = detector.detect(img, return_text=True)
    print(f"  Detected {len(regions)} regions")
    
    # Step 2: Generate CoT
    print("\n[2/2] Generating CoT with Gemini 2.5 Pro...")
    question = "What is the total amount?"
    
    try:
        output = teacher.generate_cot(
            image=img,
            question=question,
            regions=regions
        )
        
        print("\n✓ CoT Generation Successful!")
        print(f"\nAnswer: {output['answer']}")
        print(f"BBox: {output['bbox']}")
        print(f"\nCoT Steps ({len(output['cot_steps'])} steps):")
        for i, step in enumerate(output['cot_steps'], 1):
            print(f"  {i}. {step[:100]}...")
        
        print(f"\n✓ Full pipeline test completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error during CoT generation: {e}")
        print("  This may be due to:")
        print("  - Invalid API key")
        print("  - Network issues")
        print("  - Model availability")


def main():
    print("\n" + "="*60)
    print("PHASE A QUICK TEST")
    print("="*60)
    
    # Test 1: Text Detector
    try:
        test_text_detector()
    except Exception as e:
        print(f"\n✗ Text detector test failed: {e}")
        print("  Install doctr: pip install python-doctr[torch]")
    
    # Test 2: Teacher Model
    try:
        test_teacher_model()
    except Exception as e:
        print(f"\n✗ Teacher model test failed: {e}")
        print("  Check .env file has GEMINI_API_KEY")
    
    # Test 3: Full Pipeline
    try:
        test_full_pipeline()
    except Exception as e:
        print(f"\n✗ Full pipeline test failed: {e}")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()

