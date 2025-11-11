"""Models module for DocVAL"""
from .teacher_vlm import TeacherVLM
from .student_vlm import StudentVLM
from .text_detector import TextDetector

__all__ = [
    'TeacherVLM',
    'StudentVLM',
    'TextDetector'
]

