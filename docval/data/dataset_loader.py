"""
Dataset loading and management for DocVQA benchmarks.
Handles: DocVQA, VisualMRC, FUNSD, CORD, SROIE
"""
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import Dict, List, Tuple, Optional
import json
import os
from pathlib import Path

class DocVQADataset(Dataset):
    """
    Base dataset class for document VQA tasks.
    
    Returns:
        - image: PIL Image or tensor (H, W, 3)
        - question: str
        - answer_gt: str
        - bbox_gt: [x1, y1, x2, y2]
        - image_id: str
        - regions: List of detected text regions (Phase A only)
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str,
        transform=None,
        include_regions: bool = False,
        annotations: Optional[List[Dict]] = None
    ):
        """
        Args:
            data_dir: Root directory containing dataset
            split: 'train', 'val', or 'test'
            transform: Image transformations
            include_regions: Whether to include detected regions (Phase A/B2)
            annotations: Pre-loaded annotations (optional)
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.include_regions = include_regions
        
        # Load annotations
        if annotations is not None:
            self.annotations = annotations
        else:
            self.annotations = self._load_annotations()
        
        # Load detected regions if needed (for Phase A and B2)
        self.regions = {}
        if include_regions:
            self.regions = self._load_regions()
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        """
        Returns dictionary with:
        - image: processed image
        - question: question text
        - answer_gt: ground truth answer
        - bbox_gt: ground truth bounding box [x1, y1, x2, y2]
        - regions: detected text regions (if include_regions=True)
        - metadata: additional info
        """
        ann = self.annotations[idx]
        
        # Load image
        image_path = os.path.join(self.data_dir, ann['image_path'])
        if os.path.exists(image_path):
            image = Image.open(image_path).convert('RGB')
        else:
            # Create dummy image if file doesn't exist
            image = Image.new('RGB', (800, 600), color='white')
        
        if self.transform:
            image = self.transform(image)
        
        sample = {
            'image': image,
            'question': ann['question'],
            'answer_gt': ann['answer'],
            'bbox_gt': torch.tensor(ann['bbox']) if isinstance(ann['bbox'], list) else ann['bbox'],
            'image_id': ann['image_id']
        }
        
        # Add CoT if available (from teacher or ground truth)
        if 'cot_steps' in ann:
            sample['cot_steps'] = ann['cot_steps']
        
        # Add regions if available (for validation only)
        if self.include_regions and ann['image_id'] in self.regions:
            sample['regions'] = self.regions[ann['image_id']]
        
        return sample
    
    def _load_annotations(self) -> List[Dict]:
        """Load dataset annotations from JSON"""
        # Look for annotation file
        ann_path = os.path.join(self.data_dir, f"{self.split}_annotations.json")
        
        if os.path.exists(ann_path):
            with open(ann_path, 'r') as f:
                return json.load(f)
        
        # Return empty list if no annotations found
        return []
    
    def _load_regions(self) -> Dict:
        """Load pre-detected text regions"""
        regions_path = os.path.join(self.data_dir, f"{self.split}_regions.json")
        
        if os.path.exists(regions_path):
            with open(regions_path, 'r') as f:
                return json.load(f)
        
        return {}


class MultiDatasetLoader:
    """
    Manages multiple document VQA datasets.
    Combines DocVQA, VisualMRC, FUNSD, CORD, SROIE into unified format.
    """
    
    def __init__(self, config):
        self.config = config
        self.datasets = {}
        
    def load_all(self, data_dir: str, split: str) -> Dataset:
        """Load and combine all datasets"""
        all_annotations = []
        
        # Load each dataset
        for dataset_name in self.config.data.datasets:
            dataset_dir = os.path.join(data_dir, dataset_name)
            if os.path.exists(dataset_dir):
                dataset = DocVQADataset(dataset_dir, split)
                all_annotations.extend(dataset.annotations)
        
        # Create combined dataset
        combined = DocVQADataset(
            data_dir=data_dir,
            split=split,
            annotations=all_annotations
        )
        
        return combined
    
    def get_dataloader(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 4
    ) -> DataLoader:
        """Create DataLoader for dataset"""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch):
        """Custom collate function for variable-length sequences"""
        # Stack images if they're tensors
        images = [item['image'] for item in batch]
        if torch.is_tensor(images[0]):
            images = torch.stack(images)
        
        return {
            'image': images,
            'question': [item['question'] for item in batch],
            'answer_gt': [item['answer_gt'] for item in batch],
            'bbox_gt': torch.stack([item['bbox_gt'] for item in batch]) if torch.is_tensor(batch[0]['bbox_gt']) else [item['bbox_gt'] for item in batch],
            'image_id': [item['image_id'] for item in batch],
            'cot_steps': [item.get('cot_steps', []) for item in batch],
            'regions': [item.get('regions', []) for item in batch]
        }


def create_dataloaders(config, data_dir: str = "./data") -> Dict[str, DataLoader]:
    """
    Create DataLoaders for all phases.
    
    Returns:
        {
            'd1_raw': DataLoader for Phase A (with regions),
            'd3_train': DataLoader for B1 (no regions),
            'd4_val': DataLoader for B2 (with regions for VAL),
            'test': DataLoader for evaluation
        }
    """
    loader = MultiDatasetLoader(config)
    
    dataloaders = {}
    
    # D1: Raw dataset for Phase A
    if os.path.exists(data_dir):
        d1_dataset = loader.load_all(data_dir, 'train')
        d1_dataset.include_regions = True
        dataloaders['d1_raw'] = loader.get_dataloader(
            d1_dataset,
            batch_size=1,
            shuffle=False
        )
    
    # D3: Training dataset for Stage B1
    d3_path = os.path.join(data_dir, 'processed', 'd3_train.json')
    if os.path.exists(d3_path):
        with open(d3_path, 'r') as f:
            d3_annotations = json.load(f)
        d3_dataset = DocVQADataset(
            data_dir=data_dir,
            split='train',
            annotations=d3_annotations,
            include_regions=False
        )
        dataloaders['d3_train'] = loader.get_dataloader(
            d3_dataset,
            batch_size=config.training.b1_batch_size,
            shuffle=True
        )
    
    # D4: Validation dataset for Stage B2
    d4_path = os.path.join(data_dir, 'processed', 'd4_val.json')
    if os.path.exists(d4_path):
        with open(d4_path, 'r') as f:
            d4_annotations = json.load(f)
        d4_dataset = DocVQADataset(
            data_dir=data_dir,
            split='val',
            annotations=d4_annotations,
            include_regions=True
        )
        dataloaders['d4_val'] = loader.get_dataloader(
            d4_dataset,
            batch_size=config.training.b2_batch_size,
            shuffle=False
        )
    
    # Test dataset
    test_path = os.path.join(data_dir, 'processed', 'dtest.json')
    if os.path.exists(test_path):
        with open(test_path, 'r') as f:
            test_annotations = json.load(f)
        test_dataset = DocVQADataset(
            data_dir=data_dir,
            split='test',
            annotations=test_annotations,
            include_regions=False
        )
        dataloaders['test'] = loader.get_dataloader(
            test_dataset,
            batch_size=config.training.b1_batch_size,
            shuffle=False
        )
    
    return dataloaders

