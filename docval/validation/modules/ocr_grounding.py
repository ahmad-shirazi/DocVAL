"""
Module 1: OCR Grounding Engine.
Verifies bbox targets actual text and checks semantic correctness.
"""
import torch
from typing import Dict, List, Tuple, Optional

class OCRGroundingModule:
    """
    Validates spatial grounding using detected text regions.
    
    Key checks:
    1. Does predicted bbox target an actual text region?
    2. Does it target the CORRECT region (semantic matching)?
    3. Region mismatch detection (e.g., Subtotal vs Total)
    """
    
    def validate(
        self,
        pred_bbox: List[float],
        gt_bbox: List[float],
        regions: List[Dict]
    ) -> Dict:
        """
        Args:
            pred_bbox: [x1, y1, x2, y2]
            gt_bbox: [x1, y1, x2, y2]
            regions: Detected text regions
        
        Returns:
            {
                'matched_region': Dict or None,
                'gt_region': Dict,
                'is_correct_region': bool,
                'semantic_error': str or None,
                'score': float
            }
        """
        # Find region with highest IoU for prediction
        matched_region = self._find_best_match(pred_bbox, regions)
        
        # Find ground truth region
        gt_region = self._find_best_match(gt_bbox, regions)
        
        # Check if same region
        is_correct = False
        if matched_region and gt_region:
            is_correct = (matched_region['id'] == gt_region['id'])
        
        semantic_error = None
        if not is_correct and matched_region and gt_region:
            semantic_error = (
                f"Your bbox targets Region #{matched_region['id']} "
                f"(\"{matched_region.get('text', 'N/A')}\") but should target "
                f"Region #{gt_region['id']} (\"{gt_region.get('text', 'N/A')}\")"
            )
        
        return {
            'matched_region': matched_region,
            'gt_region': gt_region,
            'is_correct_region': is_correct,
            'semantic_error': semantic_error,
            'score': 1.0 if is_correct else 0.0
        }
    
    def _find_best_match(
        self,
        bbox: List[float],
        regions: List[Dict]
    ) -> Optional[Dict]:
        """Find region with highest IoU to bbox"""
        if not regions:
            return None
        
        best_iou = 0
        best_region = None
        
        for region in regions:
            iou = self._compute_iou(bbox, region['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_region = region
        
        return best_region if best_iou > 0.1 else None
    
    def _compute_iou(
        self,
        bbox1: List[float],
        bbox2: List[float]
    ) -> float:
        """Compute Intersection over Union"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

