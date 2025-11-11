"""
Module 5: Feedback Generator.
Generates output based on validation mode.
"""
from typing import Dict, List, Optional

class FeedbackGeneratorModule:
    """
    Generates feedback based on validation mode.
    
    Filter mode: Binary accept/reject
    Verifier mode: Detailed, actionable feedback
    """
    
    def __init__(self, mode: str = "filter"):
        self.mode = mode
    
    def generate(
        self,
        quality_score: float,
        ocr_result: Dict,
        answer_result: Dict,
        bbox_result: Dict,
        reasoning_result: Dict,
        prediction: Dict,
        ground_truth: Dict,
        q_threshold: float = 0.85
    ) -> Dict:
        """
        Generate appropriate feedback based on mode.
        """
        if self.mode == "filter":
            return self._generate_filter_output(
                quality_score,
                q_threshold
            )
        else:  # verifier
            return self._generate_verifier_output(
                quality_score,
                ocr_result,
                answer_result,
                bbox_result,
                reasoning_result,
                prediction,
                ground_truth,
                q_threshold
            )
    
    def _generate_filter_output(
        self,
        quality_score: float,
        q_threshold: float
    ) -> Dict:
        """
        Binary filter output for Phase A.
        
        Returns:
            {
                'decision': 'ACCEPT' or 'REJECT'
            }
        """
        decision = 'ACCEPT' if quality_score >= q_threshold else 'REJECT'
        
        return {
            'decision': decision
        }
    
    def _generate_verifier_output(
        self,
        quality_score: float,
        ocr_result: Dict,
        answer_result: Dict,
        bbox_result: Dict,
        reasoning_result: Dict,
        prediction: Dict,
        ground_truth: Dict,
        q_threshold: float
    ) -> Dict:
        """
        Detailed feedback for Phase B2.
        
        Returns comprehensive error analysis and corrections.
        """
        status = 'VALID' if quality_score >= q_threshold else 'INVALID'
        
        feedback = {
            'status': status,
            'errors': [],
            'corrections': []
        }
        
        # Answer feedback
        if answer_result.get('error_message'):
            feedback['errors'].append({
                'type': 'answer',
                'message': answer_result['error_message'],
                'severity': 'high' if answer_result['score'] < 0.5 else 'medium'
            })
            feedback['corrections'].append(
                f"Expected answer: '{ground_truth['answer']}'"
            )
        
        # BBox feedback
        if bbox_result['iou'] < 0.75:
            feedback['errors'].append({
                'type': 'bbox',
                'message': bbox_result['correction_text'],
                'severity': 'high' if bbox_result['iou'] < 0.5 else 'medium',
                'pixel_delta': bbox_result['pixel_delta']
            })
        
        # OCR grounding feedback
        if ocr_result.get('semantic_error'):
            feedback['errors'].append({
                'type': 'semantic',
                'message': ocr_result['semantic_error'],
                'severity': 'critical'
            })
        
        # Reasoning feedback
        if reasoning_result.get('issues'):
            feedback['errors'].append({
                'type': 'reasoning',
                'message': '; '.join(reasoning_result['issues']),
                'severity': 'low'
            })
        
        # Generate priority-ordered fixes
        feedback['priority_fixes'] = self._prioritize_fixes(feedback['errors'])
        
        # Generate corrected output
        feedback['corrected_output'] = {
            'answer': ground_truth['answer'],
            'bbox': ground_truth['bbox'],
            'suggested_cot': self._generate_corrected_cot(
                prediction,
                ground_truth,
                ocr_result
            )
        }
        
        return feedback
    
    def _prioritize_fixes(self, errors: List[Dict]) -> List[str]:
        """
        Order fixes by severity and dependency.
        
        Priority order:
        1. Semantic errors (wrong region)
        2. Answer errors
        3. BBox position errors
        4. Reasoning issues
        """
        severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        
        sorted_errors = sorted(
            errors,
            key=lambda x: severity_order.get(x['severity'], 99)
        )
        
        fixes = []
        for error in sorted_errors:
            fixes.append(f"[{error['severity'].upper()}] {error['message']}")
        
        return fixes
    
    def _generate_corrected_cot(
        self,
        prediction: Dict,
        ground_truth: Dict,
        ocr_result: Dict
    ) -> List[str]:
        """
        Generate suggested correct reasoning steps.
        """
        gt_region = ocr_result.get('gt_region')
        
        if gt_region:
            corrected_steps = [
                f"Step 1: Identify the relevant text region containing '{ground_truth['answer']}'",
                f"Step 2: This appears in Region #{gt_region['id']} with text '{gt_region.get('text', 'N/A')}'",
                f"Step 3: Extract answer: '{ground_truth['answer']}'",
                f"Step 4: Determine bounding box: {ground_truth['bbox']}"
            ]
        else:
            corrected_steps = [
                f"Step 1: Identify the relevant text region",
                f"Step 2: Locate the answer text",
                f"Step 3: Extract answer: '{ground_truth['answer']}'",
                f"Step 4: Determine bounding box: {ground_truth['bbox']}"
            ]
        
        return corrected_steps

