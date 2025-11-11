"""
Centralized hyperparameter management with validation and overrides.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
import yaml
import os

@dataclass
class TeacherConfig:
    model_name: str = "gemini-2.5-pro"  # Default: Gemini 2.5 Pro (thinking model)
    # Supported: gpt-5, claude-4.5-sonnet, gemini-2.5-flash, gpt-4o
    # Open-source: qwen3-vl-235b-a22b-thinking, llama4-400b-a17b
    temperature: Optional[float] = None  # None for thinking models, 0.7 for others
    max_tokens: int = 8192  # Output tokens: 8K recommended (up to 65K for Gemini)
    cost_per_102k: float = 510.0
    
    @property
    def is_thinking_model(self) -> bool:
        """Check if model is a thinking/reasoning model"""
        thinking_keywords = ['thinking', '2.5-pro', 'o1', 'o3']
        return any(kw in self.model_name.lower() for kw in thinking_keywords)
    
    def get_max_tokens_for_model(self, model_name: str = None) -> int:
        """Get recommended max_tokens for specific model"""
        model = model_name or self.model_name
        model_lower = model.lower()
        
        # Model-specific token limits
        if 'gpt-4o' in model_lower or 'gpt-5' in model_lower:
            return 16384
        elif 'gemini' in model_lower:
            return 8192
        elif 'claude' in model_lower:
            return 8192
        elif 'qwen' in model_lower or 'llama' in model_lower:
            return 8192
        else:
            return 8192  # Default

@dataclass
class StudentConfig:
    model_name: str = "google/gemma-3-12b"  # Default: Gemma3-12B (12B)
    # Supported: google/gemma-3-4b (4B), qwen/qwen3-vl-8b-thinking (8B),
    # OpenGVLab/internvl3.5-8b (8B), meta-llama/Llama-3.2-11B-Vision (11B),
    # OpenGVLab/internvl3.5-14b (14B)
    max_tokens: int = 8192  # Output generation tokens
    sequence_length: int = 8192  # Total sequence length (context + generation)
    hidden_size: int = 4096
    
    def get_max_tokens_for_model(self, model_name: str = None) -> int:
        """Get recommended max_tokens for specific student model"""
        model = model_name or self.model_name
        model_lower = model.lower()
        
        # Model-specific recommendations
        if 'gemma-3-4b' in model_lower:
            return 4096  # Smaller model, shorter outputs
        elif 'gemma-3-12b' in model_lower:
            return 8192
        elif 'qwen3-vl' in model_lower:
            return 8192  # Can handle 32K context
        elif 'internvl' in model_lower:
            return 8192
        elif 'llama-3.2' in model_lower:
            return 8192  # 128K context, but 8K output recommended
        else:
            return 8192  # Safe default
    
@dataclass
class VALConfig:
    q_min: float = 0.85
    alpha_ans: float = 0.4
    alpha_bbox: float = 0.4
    alpha_reason: float = 0.2
    filter_throughput: int = 50
    verifier_throughput: int = 12

@dataclass
class TrainingConfig:
    # Stage B1
    b1_lr: float = 2e-4
    b1_batch_size: int = 32
    b1_gradient_accumulation: int = 4
    b1_epochs: int = 3
    b1_warmup_steps: int = 500
    
    # Stage B2
    b2_lr: float = 2e-4
    b2_batch_size: int = 32
    b2_gradient_accumulation: int = 4
    b2_epochs_per_iter: int = 2
    b2_max_iterations: int = 20
    b2_convergence_window: int = 3
    b2_convergence_threshold: float = 0.2
    
    # Common
    weight_decay: float = 0.01
    optimizer: str = "adamw"

@dataclass
class DataConfig:
    total_examples: int = 102447
    filtered_examples: int = 95000
    train_split: int = 76000
    val_split: int = 9500
    test_split: int = 9500
    datasets: List[str] = field(default_factory=lambda: ["docvqa", "visualmrc", "funsd", "cord", "sroie"])

@dataclass
class DocVALConfig:
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    student: StudentConfig = field(default_factory=StudentConfig)
    val: VALConfig = field(default_factory=VALConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    @classmethod
    def from_yaml(cls, path: str):
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Parse teacher config
        teacher_dict = config_dict.get('teacher', {})
        teacher = TeacherConfig(**teacher_dict)
        
        # Parse student config
        student_dict = config_dict.get('student', {})
        student = StudentConfig(**student_dict)
        
        # Parse VAL config
        val_dict = config_dict.get('val', {})
        # Flatten nested structure
        val_params = {
            'q_min': val_dict.get('filter', {}).get('q_min', 0.85),
            'alpha_ans': val_dict.get('weights', {}).get('alpha_ans', 0.4),
            'alpha_bbox': val_dict.get('weights', {}).get('alpha_bbox', 0.4),
            'alpha_reason': val_dict.get('weights', {}).get('alpha_reason', 0.2),
            'filter_throughput': val_dict.get('filter', {}).get('throughput', 50),
            'verifier_throughput': val_dict.get('verifier', {}).get('throughput', 12)
        }
        val = VALConfig(**val_params)
        
        # Parse training config
        training_dict = config_dict.get('training', {})
        b1_dict = training_dict.get('stage_b1', {})
        b2_dict = training_dict.get('stage_b2', {})
        
        training_params = {
            'b1_lr': b1_dict.get('learning_rate', 2e-4),
            'b1_batch_size': b1_dict.get('batch_size', 32),
            'b1_gradient_accumulation': b1_dict.get('gradient_accumulation', 4),
            'b1_epochs': b1_dict.get('epochs', 3),
            'b1_warmup_steps': b1_dict.get('warmup_steps', 500),
            'b2_lr': b2_dict.get('learning_rate', 2e-4),
            'b2_batch_size': b2_dict.get('batch_size', 32),
            'b2_gradient_accumulation': b2_dict.get('gradient_accumulation', 4),
            'b2_epochs_per_iter': b2_dict.get('epochs_per_iteration', 2),
            'b2_max_iterations': b2_dict.get('max_iterations', 20),
            'b2_convergence_window': b2_dict.get('convergence_window', 3),
            'b2_convergence_threshold': b2_dict.get('convergence_threshold', 0.2),
            'weight_decay': b1_dict.get('weight_decay', 0.01),
            'optimizer': b1_dict.get('optimizer', 'adamw')
        }
        training = TrainingConfig(**training_params)
        
        # Parse data config
        data_dict = config_dict.get('data', {})
        data = DataConfig(
            total_examples=data_dict.get('total_examples', 102447),
            filtered_examples=data_dict.get('filtered_examples', 95000),
            train_split=data_dict.get('splits', {}).get('train', 76000),
            val_split=data_dict.get('splits', {}).get('val', 9500),
            test_split=data_dict.get('splits', {}).get('test', 9500),
            datasets=data_dict.get('datasets', ["docvqa", "visualmrc", "funsd", "cord", "sroie"])
        )
        
        return cls(
            teacher=teacher,
            student=student,
            val=val,
            training=training,
            data=data
        )
    
    def save(self, path: str):
        """Save configuration to YAML"""
        config_dict = {
            'teacher': asdict(self.teacher),
            'student': asdict(self.student),
            'val': {
                'filter': {
                    'q_min': self.val.q_min,
                    'throughput': self.val.filter_throughput
                },
                'verifier': {
                    'throughput': self.val.verifier_throughput
                },
                'weights': {
                    'alpha_ans': self.val.alpha_ans,
                    'alpha_bbox': self.val.alpha_bbox,
                    'alpha_reason': self.val.alpha_reason
                }
            },
            'training': {
                'stage_b1': {
                    'learning_rate': self.training.b1_lr,
                    'batch_size': self.training.b1_batch_size,
                    'gradient_accumulation': self.training.b1_gradient_accumulation,
                    'epochs': self.training.b1_epochs,
                    'warmup_steps': self.training.b1_warmup_steps,
                    'weight_decay': self.training.weight_decay,
                    'optimizer': self.training.optimizer
                },
                'stage_b2': {
                    'learning_rate': self.training.b2_lr,
                    'batch_size': self.training.b2_batch_size,
                    'gradient_accumulation': self.training.b2_gradient_accumulation,
                    'epochs_per_iteration': self.training.b2_epochs_per_iter,
                    'max_iterations': self.training.b2_max_iterations,
                    'convergence_window': self.training.b2_convergence_window,
                    'convergence_threshold': self.training.b2_convergence_threshold
                }
            },
            'data': {
                'total_examples': self.data.total_examples,
                'filtered_examples': self.data.filtered_examples,
                'splits': {
                    'train': self.data.train_split,
                    'val': self.data.val_split,
                    'test': self.data.test_split
                },
                'datasets': self.data.datasets
            }
        }
        
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

