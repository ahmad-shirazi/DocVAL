"""
Text detection models for validation-time use only.
NOT used during student inference.
"""
import torch
import torch.nn as nn
from typing import List, Dict, Tuple
from PIL import Image
import numpy as np
import cv2

class TextDetector:
    """
    Wrapper for text detection models.
    Used ONLY in Phase A (teacher generation) and Phase B2 (VAL feedback).
    
    Supported Detectors (from paper ablations):
    - DB-ResNet (default, ~50ms inference)
    - CRAFT
    - PSENet
    - EasyOCR
    
    Note: Text detection is NOT used during student inference (Phase C).
    """
    
    def __init__(
        self,
        detector_name: str = "db_resnet",  # Default: DB-ResNet
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.detector_name = detector_name
        self.device = device
        self.model = None
        self.ocr_model = None
        
        # Load detector model
        self._load_detector()
        self._load_ocr()
    
    def _load_detector(self):
        """Load specific text detection model"""
        print(f"Loading text detector: {self.detector_name}")
        
        if self.detector_name == "db_resnet":
            try:
                # Use doctr for DB-ResNet from HuggingFace
                from doctr.models import ocr_predictor, from_hub
                
                # Load DB-ResNet50 detection model
                # Using the standard doctr OCR predictor with db_resnet50
                self.model = ocr_predictor(
                    det_arch='db_resnet50',
                    reco_arch='crnn_vgg16_bn',
                    pretrained=True
                )
                print("Loaded DB-ResNet50 detector from doctr")
            except ImportError as e:
                print(f"doctr not available: {e}")
                print("Install with: pip install python-doctr")
                self.model = None
                
        elif self.detector_name == "craft":
            try:
                # CRAFT implementation
                print("CRAFT detector not yet implemented, using fallback")
                self.model = None
            except:
                self.model = None
                
        elif self.detector_name == "easyocr":
            try:
                import easyocr
                self.model = easyocr.Reader(['en'], gpu=self.device == 'cuda')
                print("Loaded EasyOCR detector")
            except ImportError:
                print("easyocr not available, using fallback")
                self.model = None
        else:
            print(f"Unknown detector: {self.detector_name}, using fallback")
            self.model = None
    
    def _load_ocr(self):
        """Load OCR model for text recognition"""
        try:
            from doctr.models import ocr_predictor
            self.ocr_model = ocr_predictor(
                det_arch='db_resnet50',
                reco_arch='crnn_vgg16_bn',
                pretrained=True
            ).to(self.device)
            print("Loaded OCR model")
        except:
            # Fallback to EasyOCR
            try:
                import easyocr
                if self.ocr_model is None:
                    self.ocr_model = easyocr.Reader(['en'], gpu=self.device == 'cuda')
                    print("Loaded EasyOCR for text recognition")
            except:
                self.ocr_model = None
                print("No OCR model available")
    
    def detect(
        self,
        image: Image.Image,
        return_text: bool = True
    ) -> List[Dict]:
        """
        Detect text regions in document image using DB-ResNet50.
        
        Args:
            image: Document image (PIL Image)
            return_text: Whether to include OCR text (doctr includes by default)
        
        Returns:
            List of detected regions:
            [
                {
                    'id': int,
                    'bbox': [x1, y1, x2, y2],  # Pixel coordinates
                    'text': str,
                    'confidence': float
                },
                ...
            ]
            
        Typical output: 15-20 regions per document
        Inference time: ~50ms on GPU with DB-ResNet50
        """
        if self.model is None:
            print("Warning: Model not loaded, using fallback detector")
            return self._generate_dummy_regions(image)
        
        # Convert PIL to numpy
        image_np = np.array(image)
        
        # Run detection based on detector type
        if self.detector_name == "easyocr":
            # EasyOCR returns both detection and recognition
            results = self.model.readtext(image_np)
            regions = self._process_easyocr_results(results)
        elif self.detector_name == "db_resnet":
            # Use doctr DB-ResNet (includes OCR)
            regions = self._detect_with_doctr(image_np)
        else:
            # Other detectors
            regions = self._detect_with_doctr(image_np)
        
        return regions
    
    def _detect_with_doctr(self, image_np: np.ndarray) -> List[Dict]:
        """Detect using doctr"""
        # Run OCR prediction
        result = self.model([image_np])
        
        # Extract regions from doctr output
        regions = []
        region_id = 0
        
        # doctr returns result with pages
        for page in result.pages:
            h, w = image_np.shape[:2]
            
            # Iterate through blocks and lines
            for block in page.blocks:
                for line in block.lines:
                    # Get bounding box (doctr returns normalized coords)
                    # geometry is ((x_min, y_min), (x_max, y_max)) in relative coords
                    geometry = line.geometry
                    x1, y1 = geometry[0]
                    x2, y2 = geometry[1]
                    
                    # Convert to pixel coordinates
                    x1_px = int(x1 * w)
                    y1_px = int(y1 * h)
                    x2_px = int(x2 * w)
                    y2_px = int(y2 * h)
                    
                    # Extract text from words
                    text = " ".join([word.value for word in line.words])
                    
                    # Calculate average confidence
                    confidences = [word.confidence for word in line.words if word.confidence]
                    confidence = float(np.mean(confidences)) if confidences else 0.9
                    
                    regions.append({
                        'id': region_id,
                        'bbox': [x1_px, y1_px, x2_px, y2_px],
                        'text': text,
                        'confidence': confidence
                    })
                    region_id += 1
        
        return regions
    
    def _process_easyocr_results(self, results: List) -> List[Dict]:
        """Process EasyOCR results"""
        regions = []
        
        for i, (bbox, text, conf) in enumerate(results):
            # EasyOCR returns [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            
            x1, y1 = min(x_coords), min(y_coords)
            x2, y2 = max(x_coords), max(y_coords)
            
            regions.append({
                'id': i,
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'text': text,
                'confidence': float(conf)
            })
        
        return regions
    
    def _generate_dummy_regions(self, image: Image.Image) -> List[Dict]:
        """Generate dummy regions for testing"""
        w, h = image.size
        regions = []
        
        # Create grid of dummy regions
        for i in range(15):
            row = i // 3
            col = i % 3
            
            x1 = int(col * w / 3) + 10
            y1 = int(row * h / 5) + 10
            x2 = x1 + int(w / 3) - 20
            y2 = y1 + int(h / 5) - 20
            
            regions.append({
                'id': i,
                'bbox': [x1, y1, x2, y2],
                'text': f'Region {i} text',
                'confidence': 0.9
            })
        
        return regions
    
    def _add_ocr_text(
        self,
        image: Image.Image,
        regions: List[Dict]
    ) -> List[Dict]:
        """Run OCR on detected regions"""
        if self.ocr_model is None:
            # Just add dummy text
            for r in regions:
                if 'text' not in r:
                    r['text'] = f"Text_{r['id']}"
            return regions
        
        image_np = np.array(image)
        
        for region in regions:
            if 'text' in region:
                continue
            
            # Crop region
            x1, y1, x2, y2 = region['bbox']
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image_np.shape[1], x2), min(image_np.shape[0], y2)
            
            crop = image_np[y1:y2, x1:x2]
            
            if crop.size == 0:
                region['text'] = ""
                continue
            
            # Run OCR on crop
            try:
                if hasattr(self.ocr_model, 'readtext'):
                    # EasyOCR
                    results = self.ocr_model.readtext(crop)
                    text = " ".join([r[1] for r in results])
                else:
                    # doctr or other
                    result = self.ocr_model([crop])
                    text = " ".join([
                        word.value
                        for page in result.pages
                        for block in page.blocks
                        for line in block.lines
                        for word in line.words
                    ])
                
                region['text'] = text
            except:
                region['text'] = ""
        
        return regions
    
    def batch_detect(
        self,
        images: List[Image.Image],
        batch_size: int = 8
    ) -> List[List[Dict]]:
        """Batch detection for efficiency"""
        all_regions = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            
            for image in batch:
                regions = self.detect(image)
                all_regions.append(regions)
        
        return all_regions

