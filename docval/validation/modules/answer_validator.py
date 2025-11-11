"""
Module 2: Answer Validator.
Validates textual correctness using ANLS metric.
"""
try:
    import Levenshtein
    LEVENSHTEIN_AVAILABLE = True
except ImportError:
    LEVENSHTEIN_AVAILABLE = False
    
from typing import Dict, List

class AnswerValidatorModule:
    """
    Validates answer correctness.
    
    Metrics:
    - ANLS (Average Normalized Levenshtein Similarity)
    - Text presence in document OCR
    """
    
    def validate(
        self,
        pred_answer: str,
        gt_answer: str,
        regions: List[Dict]
    ) -> Dict:
        """
        Compute Q_ans = 0.7 * ANLS(pred, gt) + 0.3 * 𝟙[pred ∈ OCR(R)]
        
        Returns:
            {
                'anls': float,
                'in_document': bool,
                'score': float,
                'error_message': str or None
            }
        """
        # Compute ANLS
        anls = self._compute_anls(pred_answer, gt_answer)
        
        # Check if answer exists in document
        ocr_texts = [r.get('text', '') for r in regions]
        in_document = self._text_in_ocr(pred_answer, ocr_texts)
        
        # Overall score
        score = 0.7 * anls + 0.3 * float(in_document)
        
        error_message = None
        if score < 0.9:
            if not in_document:
                error_message = (
                    f"Answer '{pred_answer}' not found in document text. "
                    f"Expected: '{gt_answer}'"
                )
            else:
                error_message = (
                    f"Answer mismatch: got '{pred_answer}', "
                    f"expected '{gt_answer}' (ANLS: {anls:.3f})"
                )
        
        return {
            'anls': anls,
            'in_document': in_document,
            'score': score,
            'error_message': error_message
        }
    
    def _compute_anls(self, pred: str, gt: str) -> float:
        """
        Average Normalized Levenshtein Similarity.
        ANLS = 1 - (edit_distance / max_length)
        """
        pred = pred.lower().strip()
        gt = gt.lower().strip()
        
        if pred == gt:
            return 1.0
        
        max_len = max(len(pred), len(gt))
        if max_len == 0:
            return 1.0
        
        if LEVENSHTEIN_AVAILABLE:
            edit_dist = Levenshtein.distance(pred, gt)
        else:
            # Fallback to simple edit distance
            edit_dist = self._simple_edit_distance(pred, gt)
        
        return max(0.0, 1.0 - edit_dist / max_len)
    
    def _simple_edit_distance(self, s1: str, s2: str) -> int:
        """Simple Levenshtein distance implementation"""
        if len(s1) < len(s2):
            return self._simple_edit_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _text_in_ocr(self, text: str, ocr_texts: List[str]) -> bool:
        """Check if text appears in OCR results"""
        text_lower = text.lower().strip()
        for ocr_text in ocr_texts:
            if text_lower in ocr_text.lower():
                return True
        return False

