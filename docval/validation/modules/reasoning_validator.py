"""
Module 4: Reasoning Validator.
Validates CoT reasoning quality and consistency.
"""
from typing import Dict, List
import re

class ReasoningValidatorModule:
    """
    Validates chain-of-thought reasoning.
    
    Checks:
    1. Structural completeness
    2. Coordinate consistency
    3. Spatial consistency
    """
    
    def validate(
        self,
        cot_steps: List[str],
        pred_bbox: List[float],
        regions: List[Dict]
    ) -> Dict:
        """
        Compute Q_reason = (S_struct + S_coord + S_spatial) / 3
        
        Returns:
            {
                'structural_score': float,
                'coordinate_score': float,
                'spatial_score': float,
                'score': float,
                'issues': List[str]
            }
        """
        # Check 1: Structural completeness
        s_struct = self._check_structure(cot_steps)
        
        # Check 2: Coordinate consistency
        s_coord = self._check_coordinates(cot_steps, pred_bbox)
        
        # Check 3: Spatial consistency
        s_spatial = self._check_spatial_language(cot_steps, pred_bbox, regions)
        
        # Overall score
        score = (s_struct + s_coord + s_spatial) / 3.0
        
        # Collect issues
        issues = []
        if s_struct < 1.0:
            issues.append("Incomplete reasoning structure")
        if s_coord < 1.0:
            issues.append("Coordinate inconsistency in reasoning")
        if s_spatial < 1.0:
            issues.append("Spatial description doesn't match bbox")
        
        return {
            'structural_score': s_struct,
            'coordinate_score': s_coord,
            'spatial_score': s_spatial,
            'score': score,
            'issues': issues
        }
    
    def _check_structure(self, cot_steps: List[str]) -> float:
        """
        Check if CoT has required components:
        - Region identification
        - Spatial analysis
        - Answer extraction
        - BBox determination
        """
        if not cot_steps:
            return 0.0
        
        required_keywords = [
            ['region', 'area', 'section', 'text'],
            ['locate', 'position', 'spatial', 'where'],
            ['answer', 'extract', 'found', 'value'],
            ['bbox', 'bounding', 'coordinates', 'box']
        ]
        
        cot_text = " ".join(cot_steps).lower()
        
        matches = 0
        for keyword_group in required_keywords:
            if any(kw in cot_text for kw in keyword_group):
                matches += 1
        
        return matches / len(required_keywords)
    
    def _check_coordinates(
        self,
        cot_steps: List[str],
        pred_bbox: List[float]
    ) -> float:
        """
        Check if coordinates mentioned in CoT match final bbox.
        """
        cot_text = " ".join(cot_steps)
        
        # Extract any coordinate mentions
        coord_pattern = r'\[?\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]?'
        mentioned_coords = re.findall(coord_pattern, cot_text)
        
        if not mentioned_coords:
            # No explicit coordinates mentioned - acceptable
            return 1.0
        
        # Check if any mentioned coords match predicted bbox
        for coord_tuple in mentioned_coords:
            try:
                coords = [float(x) for x in coord_tuple]
                if len(coords) == 4:
                    # Check approximate match
                    if all(abs(coords[i] - pred_bbox[i]) < 20 for i in range(4)):
                        return 1.0
            except:
                continue
        
        return 0.5  # Mentioned coords but don't match
    
    def _check_spatial_language(
        self,
        cot_steps: List[str],
        pred_bbox: List[float],
        regions: List[Dict]
    ) -> float:
        """
        Check if spatial descriptions (top/bottom/left/right) match actual bbox position.
        """
        cot_text = " ".join(cot_steps).lower()
        
        # Spatial keywords
        spatial_keywords = {
            'top': ['top', 'upper', 'above'],
            'bottom': ['bottom', 'lower', 'below'],
            'left': ['left', 'leftmost'],
            'right': ['right', 'rightmost'],
            'center': ['center', 'middle']
        }
        
        # Simple heuristic check
        # This would be more sophisticated in practice
        score = 1.0
        
        # If mentions "top" but bbox is in bottom half, penalize
        if any(kw in cot_text for kw in spatial_keywords['top']):
            # Assuming normalized coords or checking relative position
            if pred_bbox[1] > 300:  # Simple heuristic
                score -= 0.3
        
        if any(kw in cot_text for kw in spatial_keywords['bottom']):
            if pred_bbox[1] < 300:
                score -= 0.3
        
        return max(0.0, score)

