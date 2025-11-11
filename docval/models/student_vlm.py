"""
Student VLM (Gemma 3 12B) for fine-tuning and inference.
Learns pure visual-to-bbox mapping without region inputs.
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from typing import Dict, List, Tuple, Optional
from PIL import Image
import re

class StudentVLM(nn.Module):
    """
    Student Vision-Language Model.
    
    Supported Models (4B-14B):
    - Gemma3-4B (4B parameters)
    - Gemma3-12B (12B parameters) - DEFAULT
    - Qwen3-VL-8B-Thinking (8B)
    - InternVL3.5-8B (8B)
    - Llama-3.2-11B-Vision (11B)
    - InternVL3.5-14B (14B)
    
    Architecture:
    - Input: (image, question) only - NO regions
    - Output: (CoT, answer, bbox)
    """
    
    def __init__(
        self,
        model_name: str = "google/gemma-3-12b",
        pretrained: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.model_name = model_name
        self.device = device
        
        # Load base VLM
        print(f"Loading student model: {model_name}")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            # Try to load processor for vision models
            try:
                self.processor = AutoProcessor.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
            except:
                self.processor = None
                print("No processor found, using tokenizer only")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using dummy model for testing")
            self.model = None
            self.tokenizer = None
            self.processor = None
        
        # Add special tokens for structured output
        if self.tokenizer:
            self._add_special_tokens()
    
    def _add_special_tokens(self):
        """Add tokens for CoT structure and bbox coordinates"""
        special_tokens = {
            'additional_special_tokens': [
                '<COT>', '</COT>',
                '<STEP>', '</STEP>',
                '<ANSWER>', '</ANSWER>',
                '<BBOX>', '</BBOX>'
            ]
        }
        
        num_added = self.tokenizer.add_special_tokens(special_tokens)
        if num_added > 0 and self.model:
            self.model.resize_token_embeddings(len(self.tokenizer))
    
    def forward(
        self,
        image: torch.Tensor,
        question: str,
        target_sequence: Optional[str] = None
    ) -> Dict:
        """
        Forward pass for training.
        
        Args:
            image: Image tensor (B, C, H, W) or PIL Image
            question: Question text or list of questions
            target_sequence: Target CoT+answer+bbox sequence (training)
        
        Returns:
            {
                'loss': torch.Tensor (if target provided),
                'logits': torch.Tensor
            }
        """
        if self.model is None:
            # Return dummy output if model not loaded
            return {
                'loss': torch.tensor(0.0, device=self.device),
                'logits': torch.zeros((1, 10, 100), device=self.device)
            }
        
        # Encode inputs
        inputs = self._encode_inputs(image, question, target_sequence)
        
        # Forward through model
        outputs = self.model(**inputs)
        
        result = {'logits': outputs.logits}
        
        if target_sequence is not None and 'labels' in inputs:
            # Compute cross-entropy loss
            result['loss'] = outputs.loss if hasattr(outputs, 'loss') else self._compute_loss(
                outputs.logits,
                inputs['labels']
            )
        
        return result
    
    def generate(
        self,
        image: torch.Tensor,
        question: str,
        max_length: int = 8192,
        temperature: float = 0.7
    ) -> Dict:
        """
        Generate CoT output at inference.
        
        Returns:
            {
                'cot_steps': List[str],
                'answer': str,
                'bbox': [x1, y1, x2, y2]
            }
        """
        if self.model is None:
            # Return dummy output
            return {
                'cot_steps': ['Step 1: Analyzing document', 'Step 2: Finding answer'],
                'answer': 'N/A',
                'bbox': [0, 0, 100, 100]
            }
        
        # Encode input
        inputs = self._encode_inputs(image, question)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=False,  # Greedy for deterministic output
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode and parse
        generated_text = self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=False
        )
        
        parsed = self._parse_generation(generated_text)
        
        return parsed
    
    def _encode_inputs(
        self,
        image: torch.Tensor,
        question: str,
        target: Optional[str] = None
    ) -> Dict:
        """Encode image and text for model input"""
        # Construct prompt
        prompt = f"""Analyze this document image and answer the question with step-by-step reasoning.

Question: {question}

Provide your response in this format:
<COT>
<STEP>Step 1: [First reasoning step]</STEP>
<STEP>Step 2: [Second reasoning step]</STEP>
...
</COT>
<ANSWER>[Your answer]</ANSWER>
<BBOX>[x1, y1, x2, y2]</BBOX>
"""
        
        # Handle both single and batch inputs
        if isinstance(question, list):
            prompts = [prompt.replace(question, q) for q in question]
        else:
            prompts = prompt
        
        # Tokenize
        if self.processor and hasattr(image, 'size'):
            # Use processor for vision models
            inputs = self.processor(
                text=prompts,
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)
        else:
            # Text-only encoding
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
        
        # Add labels if target provided
        if target is not None:
            if isinstance(target, list):
                target_text = target
            else:
                target_text = [target]
            
            labels = self.tokenizer(
                target_text,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).input_ids.to(self.device)
            
            inputs['labels'] = labels
        
        return inputs
    
    def _parse_generation(self, text: str) -> Dict:
        """Parse generated text into structured output"""
        # Extract CoT steps
        cot_steps = []
        cot_pattern = r'<COT>(.*?)</COT>'
        cot_match = re.search(cot_pattern, text, re.DOTALL)
        if cot_match:
            cot_text = cot_match.group(1)
            step_pattern = r'<STEP>(.*?)</STEP>'
            steps = re.findall(step_pattern, cot_text, re.DOTALL)
            cot_steps = [s.strip() for s in steps]
        
        # Extract answer
        answer = 'N/A'
        answer_pattern = r'<ANSWER>(.*?)</ANSWER>'
        answer_match = re.search(answer_pattern, text, re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()
        
        # Extract bbox
        bbox = [0, 0, 100, 100]
        bbox_pattern = r'<BBOX>\[?\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]?</BBOX>'
        bbox_match = re.search(bbox_pattern, text)
        if bbox_match:
            bbox = [float(bbox_match.group(i)) for i in range(1, 5)]
        
        return {
            'cot_steps': cot_steps,
            'answer': answer,
            'bbox': bbox
        }
    
    def _compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss for sequence generation.
        
        Loss formula from paper:
        L_SFT = -Σ_t log P_θ(c_t | I, q, c_<t)
        """
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        return loss
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        if self.model:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'model_name': self.model_name
            }, path)
            
            # Save tokenizer
            if self.tokenizer:
                self.tokenizer.save_pretrained(path + "_tokenizer")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        if self.model:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load tokenizer if exists
            tokenizer_path = path + "_tokenizer"
            if os.path.exists(tokenizer_path):
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)


import os

