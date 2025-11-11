"""
Teacher VLM wrapper supporting multiple model backends.
Generates CoT traces with region awareness.
Handles both thinking models (no temperature) and standard models.
"""
from typing import Dict, List, Tuple, Optional
import torch
from PIL import Image
import os
import base64
import io
import json
import re

# Google Gemini imports
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

class TeacherVLM:
    """
    Teacher model that generates structured CoT outputs.
    
    Supported Models:
    - Gemini 2.5 Pro (default)
    - GPT-5
    - Claude 4.5 Sonnet
    - Gemini 2.5 Flash
    - GPT-4o
    - Open-source: Qwen3-VL-235B-A22B-Thinking, Llama4-400B-A17B
    """
    
    def __init__(
        self,
        model_name: str = "gemini-2.5-pro",
        temperature: Optional[float] = None,
        max_tokens: int = 8192
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        
        # Auto-detect if thinking model and set temperature accordingly
        self.is_thinking_model = self._is_thinking_model(model_name)
        if temperature is None:
            # Use None for thinking models, default 0.7 for others
            self.temperature = None if self.is_thinking_model else 0.7
        else:
            self.temperature = temperature
        
        self.model = None
        
        # Initialize API client or load model
        self._init_model()
    
    def _is_thinking_model(self, model_name: str) -> bool:
        """Check if model is a thinking/reasoning model"""
        thinking_keywords = ['thinking', '2.5-pro', 'o1', 'o3', 'deepthink']
        return any(kw in model_name.lower() for kw in thinking_keywords)
    
    def _init_model(self):
        """Initialize the specific teacher model"""
        if "gemini" in self.model_name.lower():
            if not GEMINI_AVAILABLE:
                raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")
            
            # Load API key from .env file or environment
            from dotenv import load_dotenv
            load_dotenv()  # Load from .env file
            
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found. Please set it in .env file or environment variables.")
            
            # Configure Gemini
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(self.model_name)
            print(f"✓ Initialized Gemini model: {self.model_name}")
            print(f"✓ API Key loaded: ...{api_key[-8:]}")
            
        elif "gpt" in self.model_name.lower():
            # Initialize OpenAI API
            try:
                import openai
                api_key = os.getenv('OPENAI_API_KEY')
                if api_key:
                    openai.api_key = api_key
                    self.model = openai
                    print(f"Initialized OpenAI model: {self.model_name}")
            except ImportError:
                print("Warning: openai not installed.")
            
        elif "claude" in self.model_name.lower():
            # Initialize Anthropic API
            try:
                import anthropic
                api_key = os.getenv('ANTHROPIC_API_KEY')
                if api_key:
                    self.model = anthropic.Anthropic(api_key=api_key)
                    print(f"Initialized Claude model: {self.model_name}")
            except ImportError:
                print("Warning: anthropic not installed.")
        else:
            # Load open-source model
            print(f"Warning: Model {self.model_name} not yet implemented.")
    
    def generate_cot(
        self,
        image: Image.Image,
        question: str,
        regions: List[Dict],
        answer_gt: Optional[str] = None,
        bbox_gt: Optional[List[float]] = None
    ) -> Dict:
        """
        Generate structured CoT output.
        
        Args:
            image: Document image
            question: Question text
            regions: Detected text regions from detector
                     [{'bbox': [x1,y1,x2,y2], 'text': str, 'id': int}, ...]
            answer_gt: Ground truth answer (for context)
            bbox_gt: Ground truth bbox (for context)
        
        Returns:
            {
                'cot_steps': List[str],  # Step-by-step reasoning
                'answer': str,            # Predicted answer
                'bbox': [x1, y1, x2, y2], # Predicted bounding box
                'raw_output': str         # Full model output
            }
        """
        # Construct prompt with regions and question
        prompt = self._construct_prompt(question, regions, answer_gt, bbox_gt)
        
        # Generate response
        response = self._generate(image, prompt)
        
        # Parse structured output
        parsed = self._parse_output(response, regions)
        
        return parsed
    
    def _construct_prompt(
        self,
        question: str,
        regions: List[Dict],
        answer_gt: Optional[str],
        bbox_gt: Optional[List[float]]
    ) -> str:
        """
        Construct prompt for teacher generation.
        
        Prompt structure:
        - Document image understanding
        - Available regions with IDs and text
        - Question
        - Request for step-by-step reasoning
        - Output format specification
        """
        regions_text = self._format_regions(regions)
        
        prompt = f"""You are an expert at analyzing document images to answer questions with precise spatial localization.

**DETECTED TEXT REGIONS IN DOCUMENT:**
{regions_text}

**QUESTION:** {question}

Please provide a detailed 7-step chain-of-thought analysis following this EXACT structure:

**Step 1: Document Understanding**
- Identify the document type and overall structure
- Example: "This is a restaurant receipt with itemized charges and summary fields"

**Step 2: Question Interpretation**
- Parse what specific information is being requested
- Example: "Question asks for 'total amount' - need the final payment total, not subtotal"

**Step 3: Visual Region Localization**
- Describe where to look in the document spatially
- Example: "Scanning lower portion of receipt where totals typically appear"

**Step 4: Field/Label Identification**
- Identify the specific label or field containing the answer
- Example: "Located 'TOTAL' label in bold text at bottom-right, distinct from 'SUBTOTAL' above"

**Step 5: Answer Extraction**
- Extract and validate the answer text from the identified region
- Example: "Amount '$108.51' positioned immediately to right of TOTAL label"

**Step 6: Spatial Verification**
- Confirm spatial relationships and positioning to ensure correctness
- Example: "This is the lowest/final line item, confirming it's the grand total rather than intermediate calculations"

**Step 7: Bbox Determination**
- Specify bounding box coordinates [x1, y1, x2, y2] with reasoning
- Example: "Bbox encompasses the numerical value: approximately [542, 876, 618, 904] in pixel coordinates"

**OUTPUT FORMAT:**
REASONING:
Step 1: [Your document understanding]
Step 2: [Your question interpretation]
Step 3: [Your visual region localization]
Step 4: [Your field/label identification]
Step 5: [Your answer extraction]
Step 6: [Your spatial verification]
Step 7: [Your bbox determination]

ANSWER: [extracted answer text]
BBOX: [x1, y1, x2, y2]
"""
        
        return prompt
    
    def _format_regions(self, regions: List[Dict]) -> str:
        """Format regions for prompt"""
        if not regions:
            return "No text regions detected."
        
        formatted = []
        for r in regions[:20]:  # Limit to top 20 regions
            bbox_str = f"[{r['bbox'][0]:.0f}, {r['bbox'][1]:.0f}, {r['bbox'][2]:.0f}, {r['bbox'][3]:.0f}]"
            text_str = r.get('text', 'N/A')
            formatted.append(f"Region #{r['id']}: {bbox_str} → \"{text_str}\"")
        
        return "\n".join(formatted)
    
    def _generate(self, image: Image.Image, prompt: str) -> str:
        """Call model API"""
        if self.model is None:
            # Return dummy response if model not initialized
            return """REASONING:
Step 1: Looking at the document regions
Step 2: Identifying the answer location
Step 3: Extracting the answer
Step 4: Determining bounding box

ANSWER: $45.99
BBOX: [100, 200, 150, 220]"""
        
        if "gemini" in self.model_name.lower():
            try:
                # Generate with Gemini
                # Build generation config based on model type
                gen_config_params = {
                    'max_output_tokens': self.max_tokens
                }
                
                # Only add temperature for non-thinking models
                if not self.is_thinking_model and self.temperature is not None:
                    gen_config_params['temperature'] = self.temperature
                
                response = self.model.generate_content(
                    [prompt, image],
                    generation_config=genai.GenerationConfig(**gen_config_params)
                )
                return response.text
            except Exception as e:
                print(f"Error generating with Gemini: {e}")
                return self._get_fallback_response()
        
        elif "gpt" in self.model_name.lower():
            # OpenAI implementation
            try:
                # Convert image to base64
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                response = self.model.ChatCompletion.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": f"data:image/png;base64,{img_str}"
                                }
                            ]
                        }
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"Error generating with OpenAI: {e}")
                return self._get_fallback_response()
        
        else:
            return self._get_fallback_response()
    
    def _get_fallback_response(self) -> str:
        """Fallback response when model fails"""
        return """REASONING:
Step 1: Analyzing document regions
Step 2: Identifying relevant information
Step 3: Extracting answer
Step 4: Determining coordinates

ANSWER: N/A
BBOX: [0, 0, 100, 100]"""
    
    def _parse_output(self, response: str, regions: List[Dict]) -> Dict:
        """Parse model response into structured 7-step format"""
        # Extract reasoning steps
        cot_steps = []
        reasoning_match = re.search(r'REASONING:(.*?)(?=ANSWER:|$)', response, re.DOTALL | re.IGNORECASE)
        if reasoning_match:
            reasoning_text = reasoning_match.group(1).strip()
            # Split into steps - improved pattern to handle multi-line steps
            # Match "Step N:" followed by content until the next "Step N:" or end
            steps = re.findall(r'Step \d+:\s*(.*?)(?=\n\nStep \d+:|\nStep \d+:|$)', reasoning_text, re.DOTALL)
            cot_steps = [step.strip() for step in steps if step.strip()]
        
        # Extract answer
        answer = "N/A"
        answer_match = re.search(r'ANSWER:\s*(.+?)(?=\n+BBOX:|$)', response, re.IGNORECASE)
        if answer_match:
            answer = answer_match.group(1).strip()
        
        # Extract bbox
        bbox = [0, 0, 100, 100]
        bbox_match = re.search(r'BBOX:\s*\[?\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]?', response, re.IGNORECASE)
        if bbox_match:
            bbox = [float(bbox_match.group(i)) for i in range(1, 5)]
        
        return {
            'cot_steps': cot_steps,
            'answer': answer,
            'bbox': bbox,
            'raw_output': response
        }
    
    def batch_generate(
        self,
        batch: List[Dict],
        batch_size: int = 8
    ) -> List[Dict]:
        """Generate CoT for batch of examples"""
        results = []
        
        for item in batch:
            result = self.generate_cot(
                image=item['image'],
                question=item['question'],
                regions=item.get('regions', []),
                answer_gt=item.get('answer_gt'),
                bbox_gt=item.get('bbox_gt')
            )
            results.append(result)
        
        return results

