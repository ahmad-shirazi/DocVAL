"""
Data processing utilities for DocVAL pipeline.
Handles dataset flow: D1 -> D2 -> D3/D4/Dtest
"""
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset
import random

class DataProcessor:
    """
    Manages data flow through DocVAL pipeline:
    - D1: Raw 102,447 examples
    - D2: Teacher-generated (102K with CoT)
    - D3: Filtered training set (76K)
    - D4: Filtered validation set (9.5K)
    - Dtest: Held-out test set (9.5K)
    """
    
    def __init__(self, config):
        self.config = config
    
    def create_d2_from_d1(
        self,
        d1_dataset: Dataset,
        teacher_outputs: List[Dict]
    ) -> List[Dict]:
        """
        Augment D1 with teacher CoT traces to create D2.
        
        Args:
            d1_dataset: Original dataset with (image, question, answer_gt, bbox_gt)
            teacher_outputs: List of teacher predictions with CoT
        
        Returns:
            D2 dataset with added teacher CoT traces
        """
        d2_data = []
        
        for i, output in enumerate(teacher_outputs):
            # Get original data
            original = d1_dataset[i] if i < len(d1_dataset) else {}
            
            # Merge teacher output with original
            example = {
                'image_id': output.get('image_id', f'doc_{i}'),
                'image_path': original.get('image_path', output.get('image_path', '')),
                'question': output['question'],
                'answer': output['answer_gt'],
                'bbox': output['bbox_gt'],
                'cot_steps': output['cot_steps'],
                'answer_pred': output['answer_pred'],
                'bbox_pred': output['bbox_pred'],
                'regions': output.get('regions', [])
            }
            
            d2_data.append(example)
        
        return d2_data
    
    def apply_val_filter(
        self,
        d2_dataset: List[Dict],
        quality_scores: List[float],
        q_threshold: float = 0.85
    ) -> Tuple[List[Dict], List[float]]:
        """
        Filter D2 using VAL Filter to create high-quality dataset.
        
        Returns:
            - Filtered dataset (95K examples)
            - Quality scores for each example
        """
        filtered_data = []
        filtered_scores = []
        
        for example, score in zip(d2_dataset, quality_scores):
            if score >= q_threshold:
                filtered_data.append(example)
                filtered_scores.append(score)
        
        return filtered_data, filtered_scores
    
    def split_filtered_data(
        self,
        filtered_dataset: List[Dict],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Split filtered data into D3/D4/Dtest (76K/9.5K/9.5K).
        
        Returns:
            (d3_train, d4_val, dtest)
        """
        # Shuffle with seed for reproducibility
        random.seed(seed)
        shuffled_data = filtered_dataset.copy()
        random.shuffle(shuffled_data)
        
        n_total = len(shuffled_data)
        n_train = int(train_ratio * n_total)
        n_val = int(val_ratio * n_total)
        
        d3_train = shuffled_data[:n_train]
        d4_val = shuffled_data[n_train:n_train + n_val]
        dtest = shuffled_data[n_train + n_val:]
        
        return d3_train, d4_val, dtest
    
    def prepare_training_format(
        self,
        examples: List[Dict],
        include_regions: bool = False
    ) -> List[Dict]:
        """
        Format examples for training.
        Remove regions for student training if specified.
        """
        formatted = []
        
        for ex in examples:
            sample = {
                'image_id': ex['image_id'],
                'image_path': ex.get('image_path', ''),
                'question': ex['question'],
                'answer': ex['answer'],
                'bbox': ex['bbox'],
                'cot_steps': ex.get('cot_steps', [])
            }
            
            if include_regions and 'regions' in ex:
                sample['regions'] = ex['regions']
            
            formatted.append(sample)
        
        return formatted

