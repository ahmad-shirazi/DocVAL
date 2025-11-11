#!/usr/bin/env python3
"""
Test Phase B1 Setup
Verifies data, model access, and hardware before training
"""

import json
import torch
from pathlib import Path
from PIL import Image


def test_hardware():
    """Test GPU availability"""
    print("\n" + "="*80)
    print("TESTING HARDWARE")
    print("="*80)
    
    # PyTorch version
    print(f"\nPyTorch version: {torch.__version__}")
    
    # CUDA
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        
        # GPU info
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  Total memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"  Compute capability: {props.major}.{props.minor}")
        
        # Test allocation
        try:
            x = torch.randn(1000, 1000).cuda()
            print(f"\n✓ Successfully allocated tensor on GPU")
            del x
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"\n✗ Error allocating on GPU: {e}")
    else:
        print("\n⚠ Warning: No GPU available. Training will be very slow!")
        print("  Consider using a GPU instance (A100, H100, RTX 3090, etc.)")
    
    return cuda_available


def test_data():
    """Test training data availability"""
    print("\n" + "="*80)
    print("TESTING DATA")
    print("="*80)
    
    base_dir = Path("/Users/ahmadshirazi/Desktop/DocVAL")
    
    # Check training data
    train_path = base_dir / "docval/data/phase_a_output/filtered/D3_train.json"
    val_path = base_dir / "docval/data/phase_a_output/filtered/D4_val.json"
    image_dir = base_dir / "docval/data/cot_data"
    
    print(f"\nChecking files:")
    print(f"  Train data: {train_path}")
    print(f"    Exists: {train_path.exists()}")
    
    if train_path.exists():
        with open(train_path, 'r') as f:
            train_data = json.load(f)
        print(f"    Examples: {len(train_data)}")
        
        # Check first example structure
        if train_data:
            ex = train_data[0]
            print(f"\n  Sample example:")
            print(f"    image_id: {ex.get('image_id')}")
            print(f"    question: {ex.get('question')[:50]}...")
            print(f"    cot_steps: {len(ex.get('cot_steps', []))} steps")
            print(f"    answer: {ex.get('answer_pred')}")
            print(f"    bbox: {ex.get('bbox_pred')}")
            
            # Check if image exists
            dataset = ex['image_id'].split('_')[0]
            if dataset == 'cord':
                img_path = image_dir / 'CORD' / ex['image_file']
            elif dataset == 'docvqa':
                img_path = image_dir / 'DocVQA' / 'images' / ex['image_file']
            elif dataset == 'funsd':
                img_path = image_dir / 'FUNSD' / ex['image_file']
            elif dataset == 'sroie':
                img_path = image_dir / 'SROIE' / ex['image_file']
            elif dataset == 'visualmrc':
                img_path = image_dir / 'VisualMRC' / ex['image_file']
            
            print(f"\n  Sample image:")
            print(f"    Path: {img_path}")
            print(f"    Exists: {img_path.exists()}")
            
            if img_path.exists():
                try:
                    img = Image.open(img_path)
                    print(f"    Size: {img.size}")
                    print(f"    Mode: {img.mode}")
                    print(f"    ✓ Image loads successfully")
                except Exception as e:
                    print(f"    ✗ Error loading image: {e}")
            else:
                print(f"    ✗ Image not found!")
    
    print(f"\n  Val data: {val_path}")
    print(f"    Exists: {val_path.exists()}")
    
    if val_path.exists():
        with open(val_path, 'r') as f:
            val_data = json.load(f)
        print(f"    Examples: {len(val_data)}")
    
    print(f"\n  Image directory: {image_dir}")
    print(f"    Exists: {image_dir.exists()}")
    
    if image_dir.exists():
        datasets = ['CORD', 'DocVQA', 'FUNSD', 'SROIE', 'VisualMRC']
        for ds in datasets:
            ds_path = image_dir / ds
            if ds_path.exists():
                if ds == 'DocVQA':
                    images = list((ds_path / 'images').glob('*.png')) if (ds_path / 'images').exists() else []
                else:
                    images = list(ds_path.glob('*.png')) + list(ds_path.glob('*.jpg'))
                print(f"    {ds}: {len(images)} images")
    
    return train_path.exists() and val_path.exists()


def test_dependencies():
    """Test required packages"""
    print("\n" + "="*80)
    print("TESTING DEPENDENCIES")
    print("="*80)
    
    packages = {
        'transformers': None,
        'peft': None,
        'accelerate': None,
        'bitsandbytes': None,
        'pillow': 'PIL',
        'numpy': None,
    }
    
    all_ok = True
    for package, import_name in packages.items():
        try:
            mod = __import__(import_name or package)
            version = getattr(mod, '__version__', 'unknown')
            print(f"  ✓ {package}: {version}")
        except ImportError:
            print(f"  ✗ {package}: NOT INSTALLED")
            all_ok = False
    
    return all_ok


def test_model_access():
    """Test HuggingFace model access"""
    print("\n" + "="*80)
    print("TESTING MODEL ACCESS")
    print("="*80)
    
    print("\nAttempting to load processor...")
    
    try:
        from transformers import AutoProcessor
        
        processor = AutoProcessor.from_pretrained(
            "google/gemma-3-12b-it",
            trust_remote_code=True
        )
        print("✓ Processor loaded successfully")
        print(f"  Tokenizer vocab size: {len(processor.tokenizer)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading processor: {e}")
        print("\nPossible issues:")
        print("  1. Not logged in to HuggingFace")
        print("     Solution: Run 'huggingface-cli login'")
        print("  2. Gemma license not accepted")
        print("     Solution: Visit https://huggingface.co/google/gemma-3-12b-it")
        print("  3. No internet connection")
        print("     Solution: Check your network")
        
        return False


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("PHASE B1 SETUP VERIFICATION")
    print("="*80)
    print("\nThis script verifies:")
    print("  1. Hardware (GPU availability)")
    print("  2. Data (training files and images)")
    print("  3. Dependencies (required packages)")
    print("  4. Model access (HuggingFace authentication)")
    
    results = {
        'hardware': test_hardware(),
        'data': test_data(),
        'dependencies': test_dependencies(),
        'model_access': test_model_access(),
    }
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    all_passed = all(results.values())
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name.upper()}: {status}")
    
    print("\n" + "="*80)
    
    if all_passed:
        print("✓ ALL TESTS PASSED - Ready to start training!")
        print("\nTo start training, run:")
        print("  python start_phase_b1_training.py")
    else:
        print("✗ SOME TESTS FAILED - Please fix issues before training")
        print("\nFailed tests:")
        for test_name, passed in results.items():
            if not passed:
                print(f"  • {test_name}")
    
    print("="*80 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

