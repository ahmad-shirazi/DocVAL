"""
DocVAL: Validated Chain-of-Thought Distillation for Document VQA

A framework for training compact Vision-Language Models to perform
precise answer localization without requiring text detection at inference time.
"""

__version__ = "1.0.0"
__author__ = "DocVAL Team"

from .config import DocVALConfig
from .models import TeacherVLM, StudentVLM, TextDetector
from .validation import VALSystem, VALFilter, VALVerifier

__all__ = [
    'DocVALConfig',
    'TeacherVLM',
    'StudentVLM',
    'TextDetector',
    'VALSystem',
    'VALFilter',
    'VALVerifier',
]

