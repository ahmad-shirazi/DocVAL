"""
Module 3: BBox Validator.
Validates spatial localization accuracy.
"""
import torch
from typing import Dict, List, Optional

class BBoxValidatorModule:
    """
    Validates bounding box accuracy.
    
    Metrics:
    - IoU (Intersection over Union)
    - Region correctness
    - Pixel-level error vector
    """
    
    def validate(
        self,
        pred_bbox: List[float],
        gt_bbox: List[float],
        pred_region: Optional[Dict],
        gt_region: Optional[Dict]
    ) -> Dict:
        """
        Compute Q_bbox = 0.8 * IoU(pred, gt) + 0.2 * 𝟙[region_match]
        
        Returns:
            {
                'iou': float,
                'region_match': bool,
                'pixel_delta': [dx1, dy1, dx2, dy2],
                'score': float,
                'correction_text': str
            }
        """
        # Compute IoU
        iou = self._compute_iou(pred_bbox, gt_bbox)
        
        # Check region match
        region_match = (
            pred_region is not None and
            gt_region is not None and
            pred_region['id'] == gt_region['id']
        )
        
        # Compute pixel-level corrections
        pixel_delta = [
            gt_bbox[0] - pred_bbox[0],  # dx1
            gt_bbox[1] - pred_bbox[1],  # dy1
            gt_bbox[2] - pred_bbox[2],  # dx2
            gt_bbox[3] - pred_bbox[3]   # dy2
        ]
        
        # Overall score
        score = 0.8 * iou + 0.2 * float(region_match)
        
        # Generate correction text
        correction_text = self._generate_correction(
            pred_bbox,
            gt_bbox,
            pixel_delta,
            iou
        )
        
        return {
            'iou': iou,
            'region_match': region_match,
            'pixel_delta': pixel_delta,
            'score': score,
            'correction_text': correction_text
        }
    
    def _compute_iou(
        self,
        bbox1: List[float],
        bbox2: List[float]
    ) -> float:
        """Compute Intersection over Union"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _generate_correction(
        self,
        pred_bbox: List[float],
        gt_bbox: List[float],
        delta: List[float],
        iou: float
    ) -> str:
        """Generate human-readable correction guidance"""
        if iou > 0.9:
            return "BBox nearly correct (IoU > 0.9)"
        
        dx1, dy1, dx2, dy2 = delta
        
        corrections = []
        if abs(dx1) > 5:
            direction = "RIGHT" if dx1 > 0 else "LEFT"
            corrections.append(f"Move {direction} by {abs(dx1):.0f}px")
        
        if abs(dy1) > 5:
            direction = "DOWN" if dy1 > 0 else "UP"
            corrections.append(f"Move {direction} by {abs(dy1):.0f}px")
        
        if abs(dx2 - dx1) > 5:
            action = "EXPAND" if (dx2 - dx1) > 0 else "SHRINK"
            corrections.append(f"{action} width")
        
        if abs(dy2 - dy1) > 5:
            action = "EXPAND" if (dy2 - dy1) > 0 else "SHRINK"
            corrections.append(f"{action} height")
        
        correction_text = (
            f"BBox IoU: {iou:.3f}. "
            f"Predicted: {[int(x) for x in pred_bbox]}, Target: {[int(x) for x in gt_bbox]}. "
            f"Corrections: {'; '.join(corrections) if corrections else 'None'}"
        )
        
        return correction_text

