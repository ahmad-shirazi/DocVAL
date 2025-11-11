"""
Prepare all 5 datasets for Phase A processing.
Handles different formats: PNG images, JSON, Arrow files.
Creates consistent annotations.json for each dataset.
"""
import json
import os
from pathlib import Path
from datasets import load_from_disk
from PIL import Image


def prepare_cord(data_dir):
    """CORD: PNG images directly in folder"""
    cord_dir = Path(data_dir) / "CORD"
    images = sorted(list(cord_dir.glob("*.png")))  # All images
    
    annotations = []
    questions = [
        "What is the total amount?",
        "What is the menu number?",
        "What is the store name?",
        "What is the date?",
        "What is the subtotal?"
    ]
    
    for idx, img_path in enumerate(images):
        # Extract receipt number from filename (e.g., receipt_00010.png -> 00010)
        receipt_num = img_path.stem.split('_')[-1] if '_' in img_path.stem else f"{idx:05d}"
        
        annotations.append({
            "image_id": f"cord_{receipt_num}",
            "image_file": img_path.name,
            "question": questions[idx % len(questions)],
            "answer": "",  # Will be extracted by model
            "bbox": None
        })
    
    return annotations


def prepare_docvqa(data_dir):
    """DocVQA: images/ folder + qas/val.json"""
    docvqa_dir = Path(data_dir) / "DocVQA"
    qas_file = docvqa_dir / "qas" / "val.json"
    
    if not qas_file.exists():
        print(f"  Warning: {qas_file} not found")
        return []
    
    with open(qas_file, 'r') as f:
        data = json.load(f)
    
    annotations = []
    for idx, item in enumerate(data.get('data', [])):  # All examples
        # DocVQA format: page_ids contains the image filename without extension
        page_ids = item.get('page_ids', [])
        answer_page_idx = item.get('answer_page_idx', 0)
        
        if page_ids:
            # Get the page where answer is located
            page_id = page_ids[answer_page_idx] if answer_page_idx < len(page_ids) else page_ids[0]
            image_file = f"{page_id}.jpg"
        else:
            continue  # Skip if no page_ids
        
        question = item.get('question', '')
        answers = item.get('answers', [])
        answer = answers[0] if answers else ""
        
        # Use the page_id as the unique identifier
        page_id = page_ids[answer_page_idx] if answer_page_idx < len(page_ids) else page_ids[0]
        
        annotations.append({
            "image_id": f"docvqa_{page_id}",
            "image_file": image_file,
            "question": question,
            "answer": answer,
            "bbox": None
        })
    
    return annotations


def prepare_funsd(data_dir):
    """FUNSD: PNG images directly in folder"""
    funsd_dir = Path(data_dir) / "FUNSD"
    images = sorted(list(funsd_dir.glob("*.png")))  # All images
    
    annotations = []
    questions = [
        "What is the company name?",
        "What is the date?",
        "What is the address?",
        "Who is the sender?",
        "What is the document about?"
    ]
    
    for idx, img_path in enumerate(images):
        # Use the actual filename stem as the ID (e.g., 82092117.png -> 82092117)
        doc_id = img_path.stem
        
        annotations.append({
            "image_id": f"funsd_{doc_id}",
            "image_file": img_path.name,
            "question": questions[idx % len(questions)],
            "answer": "",  # Will be extracted by model
            "bbox": None
        })
    
    return annotations


def prepare_sroie(data_dir):
    """SROIE: SROIE2019 subfolder structure"""
    sroie_dir = Path(data_dir) / "SROIE" / "SROIE2019"
    
    # Check train or test folder with img subfolder
    possible_dirs = [
        sroie_dir / "train" / "img",
        sroie_dir / "test" / "img",
        sroie_dir / "train",
        sroie_dir / "test"
    ]
    
    image_dir = None
    for d in possible_dirs:
        if d.exists():
            image_dir = d
            break
    
    if not image_dir:
        print(f"  Warning: No images found in {sroie_dir}")
        return []
    
    images = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))  # All images
    
    annotations = []
    questions = [
        "What is the company name?",
        "What is the total amount?",
        "What is the date?",
        "What is the address?",
        "What is the receipt number?"
    ]
    
    for idx, img_path in enumerate(images):
        # Use the actual filename stem as the ID (e.g., X51005663300.jpg -> X51005663300)
        doc_id = img_path.stem
        
        annotations.append({
            "image_id": f"sroie_{doc_id}",
            "image_file": img_path.name,
            "question": questions[idx % len(questions)],
            "answer": "",  # Will be extracted by model
            "bbox": None
        })
    
    return annotations


def prepare_visualmrc(data_dir):
    """VisualMRC: HuggingFace Arrow format"""
    visualmrc_dir = Path(data_dir) / "VisualMRC"
    
    try:
        # Load dataset using datasets library
        dataset = load_from_disk(str(visualmrc_dir))
        
        # Use validation split if available, otherwise test
        split_name = 'val' if 'val' in dataset else 'test'
        split_data = dataset[split_name]
        
        annotations = []
        for idx in range(len(split_data)):  # All examples
            item = split_data[idx]
            
            # Extract image and save to file
            image = item.get('image')
            if image is not None:
                # Save image if it's PIL Image
                image_filename = f"visualmrc_{idx:05d}.png"
                image_path = visualmrc_dir / image_filename
                
                if isinstance(image, Image.Image):
                    image.save(image_path)
                
                # Use idx with proper formatting but make it clear it's sequential
                annotations.append({
                    "image_id": f"visualmrc_{split_name}_{idx:05d}",
                    "image_file": image_filename,
                    "question": item.get('question', ''),
                    "answer": item.get('answer', item.get('answers', [''])[0]),
                    "bbox": None
                })
        
        return annotations
        
    except Exception as e:
        print(f"  Error loading VisualMRC: {e}")
        return []


def main():
    data_dir = Path('/Users/ahmadshirazi/Desktop/DocVAL/docval/data/cot_data')
    
    print("="*70)
    print("PREPARING ALL 5 DATASETS FOR PHASE A")
    print("="*70)
    print("\nLoading ALL examples from each dataset...\n")
    
    datasets = {
        'CORD': prepare_cord,
        'DocVQA': prepare_docvqa,
        'FUNSD': prepare_funsd,
        'SROIE': prepare_sroie,
        'VisualMRC': prepare_visualmrc
    }
    
    results = {}
    
    for name, prepare_func in datasets.items():
        print(f"[{name}] Preparing annotations...")
        try:
            annotations = prepare_func(data_dir)
            
            if annotations:
                # Save annotations
                output_file = data_dir / name / 'annotations.json'
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_file, 'w') as f:
                    json.dump(annotations, f, indent=2)
                
                results[name] = len(annotations)
                print(f"  ✓ Created {len(annotations)} annotations → {output_file}")
            else:
                results[name] = 0
                print(f"  ✗ No annotations created")
                
        except Exception as e:
            results[name] = 0
            print(f"  ✗ Error: {e}")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for name, count in results.items():
        print(f"  {name:15s}: {count:3d} annotations")
    print(f"  {'TOTAL':15s}: {sum(results.values()):3d} annotations")
    
    print("\n✓ All datasets prepared!")
    print("Ready to run Phase A with: python -m docval.scripts.run_phase_a")


if __name__ == "__main__":
    main()

