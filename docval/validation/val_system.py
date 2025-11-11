"""
Main VAL (Validator for Answer Localization) system.
Dual-mode operation: Filter (Phase A) and Verifier (Phase B2).
"""
from typing import Dict, List, Tuple, Optional
import torch
from .modules import (
    OCRGroundingModule,
    AnswerValidatorModule,
    BBoxValidatorModule,
    ReasoningValidatorModule,
    FeedbackGeneratorModule
)

class VALSystem:
    """
    Validator for Answer Localization.
    
    Two operational modes:
    1. Filter Mode (Phase A): Binary accept/reject at 50 ex/sec
    2. Verifier Mode (Phase B2): Detailed feedback at 12 ex/sec
    
    Five validation modules:
    1. OCR Grounding Engine
    2. Answer Validator
    3. BBox Validator
    4. Reasoning Validator
    5. Feedback Generator
    """
    
    def __init__(
        self,
        config,
        mode: str = "filter"  # "filter" or "verifier"
    ):
        self.config = config
        self.mode = mode
        
        # Initialize modules
        self.ocr_grounding = OCRGroundingModule()
        self.answer_validator = AnswerValidatorModule()
        self.bbox_validator = BBoxValidatorModule()
        self.reasoning_validator = ReasoningValidatorModule()
        self.feedback_generator = FeedbackGeneratorModule(mode=mode)
        
        # Module weights
        self.alpha_ans = config.alpha_ans
        self.alpha_bbox = config.alpha_bbox
        self.alpha_reason = config.alpha_reason
    
    def validate(
        self,
        image: torch.Tensor,
        regions: List[Dict],
        prediction: Dict,
        ground_truth: Dict,
        iteration: Optional[int] = None
    ) -> Dict:
        """
        Validate prediction and generate feedback.
        
        Args:
            image: Document image
            regions: Detected text regions (from text detector)
            prediction: Student/teacher output
                {
                    'cot_steps': List[str],
                    'answer': str,
                    'bbox': [x1, y1, x2, y2]
                }
            ground_truth: Ground truth
                {
                    'answer': str,
                    'bbox': [x1, y1, x2, y2]
                }
            iteration: Current iteration (for B2 tracking)
        
        Returns:
            If mode='filter':
                {
                    'decision': 'ACCEPT' or 'REJECT',
                    'quality_score': float,
                    'module_scores': Dict
                }
            
            If mode='verifier':
                {
                    'status': 'VALID' or 'INVALID',
                    'quality_score': float,
                    'module_scores': Dict,
                    'feedback': {
                        'answer_error': str,
                        'bbox_error': str,
                        'reasoning_issues': List[str],
                        'pixel_corrections': [dx1, dy1, dx2, dy2],
                        'priority_fixes': List[str],
                        'corrected_output': Dict
                    }
                }
        """
        # Module 1: OCR Grounding
        ocr_result = self.ocr_grounding.validate(
            prediction['bbox'],
            ground_truth['bbox'],
            regions
        )
        
        # Module 2: Answer Validation
        answer_result = self.answer_validator.validate(
            prediction['answer'],
            ground_truth['answer'],
            regions
        )
        
        # Module 3: BBox Validation
        bbox_result = self.bbox_validator.validate(
            prediction['bbox'],
            ground_truth['bbox'],
            ocr_result.get('matched_region'),
            ocr_result.get('gt_region')
        )
        
        # Module 4: Reasoning Validation
        reasoning_result = self.reasoning_validator.validate(
            prediction.get('cot_steps', []),
            prediction['bbox'],
            regions
        )
        
        # Compute overall quality score
        Q_ans = answer_result['score']
        Q_bbox = bbox_result['score']
        Q_reason = reasoning_result['score']
        
        Q = (self.alpha_ans * Q_ans +
             self.alpha_bbox * Q_bbox +
             self.alpha_reason * Q_reason)
        
        # Module 5: Generate output based on mode
        output = self.feedback_generator.generate(
            quality_score=Q,
            ocr_result=ocr_result,
            answer_result=answer_result,
            bbox_result=bbox_result,
            reasoning_result=reasoning_result,
            prediction=prediction,
            ground_truth=ground_truth,
            q_threshold=self.config.q_min
        )
        
        output['quality_score'] = Q
        output['module_scores'] = {
            'answer': Q_ans,
            'bbox': Q_bbox,
            'reasoning': Q_reason
        }
        
        return output
    
    def batch_validate(
        self,
        batch: List[Dict],
        batch_size: int = 32
    ) -> List[Dict]:
        """Batch validation for efficiency"""
        results = []
        for item in batch:
            result = self.validate(
                item['image'],
                item['regions'],
                item['prediction'],
                item['ground_truth']
            )
            results.append(result)
        return results


class VALFilter(VALSystem):
    """
    Filter mode for Phase A.
    Binary accept/reject at 50 examples/sec.
    """
    def __init__(self, config):
        super().__init__(config, mode="filter")


class VALVerifier(VALSystem):
    """
    Verifier mode for Phase B2.
    Detailed feedback at 12 examples/sec.
    """
    def __init__(self, config):
        super().__init__(config, mode="verifier")

