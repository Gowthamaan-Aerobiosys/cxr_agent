"""
CXR Classifier
Provides classification capabilities for chest X-ray images including disease detection
and abnormality classification
"""

import torch
import torch.nn as nn
import torchvision.models as models
from transformers import AutoModel, AutoProcessor
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
import logging
from pathlib import Path
import json

from .image_processor import CXRImageProcessor

logger = logging.getLogger(__name__)

class CXRClassifier:
    """Multi-task CXR classifier for disease detection and abnormality classification"""
    
    def __init__(self, model_type: str = "densenet121", num_classes: int = 14):
        self.model_type = model_type
        self.num_classes = num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_processor = CXRImageProcessor()
        
        # CXR pathology labels (based on NIH ChestX-ray14 dataset)
        self.pathology_labels = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
            'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
            'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
            'Pleural_Thickening', 'Hernia'
        ]
        
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        """Load the classification model"""
        try:
            if self.model_type == "densenet121":
                self._load_densenet()
            elif self.model_type == "vision_transformer":
                self._load_vision_transformer()
            elif self.model_type == "resnet50":
                self._load_resnet()
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            logger.info(f"Loaded {self.model_type} classifier with {self.num_classes} classes")
        
        except Exception as e:
            logger.error(f"Error loading classifier model: {str(e)}")
            raise
    
    def _load_densenet(self):
        """Load DenseNet-121 model (commonly used for CXR classification)"""
        self.model = models.densenet121(pretrained=True)
        
        # Modify classifier for CXR pathologies
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, self.num_classes),
            nn.Sigmoid()  # Multi-label classification
        )
        
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def _load_resnet(self):
        """Load ResNet-50 model"""
        self.model = models.resnet50(pretrained=True)
        
        # Modify final layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, self.num_classes),
            nn.Sigmoid()
        )
        
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def _load_vision_transformer(self):
        """Load Vision Transformer for CXR classification"""
        try:
            # Using a medical imaging specific model if available
            model_name = "microsoft/BiomedVLP-CXR-BERT-specialized"  # Example medical model
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            
            # Add classification head
            self.classifier_head = nn.Sequential(
                nn.Linear(self.model.config.hidden_size, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, self.num_classes),
                nn.Sigmoid()
            )
            
            self.model = self.model.to(self.device)
            self.classifier_head = self.classifier_head.to(self.device)
            
        except Exception:
            # Fallback to standard ViT
            logger.warning("Medical ViT not available, using standard ViT")
            self.model = models.vit_b_16(pretrained=True)
            self.model.heads.head = nn.Sequential(
                nn.Linear(768, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, self.num_classes),
                nn.Sigmoid()
            )
            self.model = self.model.to(self.device)
    
    def classify_image(self, image_path: Union[str, Path, np.ndarray]) -> Dict[str, float]:
        """Classify a single CXR image"""
        try:
            # Load and preprocess image
            if isinstance(image_path, np.ndarray):
                image = image_path
            else:
                image = self.image_processor.load_image(image_path)
            
            # Preprocess for model
            input_tensor = self.image_processor.preprocess_for_model(image)
            
            # Inference
            with torch.no_grad():
                if self.model_type == "vision_transformer" and hasattr(self, 'classifier_head'):
                    features = self.model(input_tensor).last_hidden_state.mean(dim=1)
                    outputs = self.classifier_head(features)
                else:
                    outputs = self.model(input_tensor)
                
                probabilities = torch.sigmoid(outputs).cpu().numpy()[0]
            
            # Create results dictionary
            results = {}
            for i, label in enumerate(self.pathology_labels[:len(probabilities)]):
                results[label] = float(probabilities[i])
            
            logger.info(f"Classification completed for image")
            return results
        
        except Exception as e:
            logger.error(f"Error in image classification: {str(e)}")
            raise
    
    def classify_batch(self, image_paths: List[Union[str, Path]]) -> List[Dict[str, float]]:
        """Classify multiple CXR images"""
        results = []
        
        for image_path in image_paths:
            try:
                result = self.classify_image(image_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Error classifying {image_path}: {str(e)}")
                results.append({})
        
        return results
    
    def get_top_predictions(self, classification_results: Dict[str, float], top_k: int = 5) -> List[Tuple[str, float]]:
        """Get top K predictions from classification results"""
        sorted_results = sorted(classification_results.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]
    
    def filter_predictions(self, classification_results: Dict[str, float], threshold: float = 0.5) -> Dict[str, float]:
        """Filter predictions above threshold"""
        return {k: v for k, v in classification_results.items() if v >= threshold}
    
    def generate_report(self, classification_results: Dict[str, float], threshold: float = 0.3) -> str:
        """Generate a clinical report from classification results"""
        high_confidence = {k: v for k, v in classification_results.items() if v >= threshold}
        
        if not high_confidence:
            return "No significant abnormalities detected above the confidence threshold."
        
        report_lines = ["CHEST X-RAY CLASSIFICATION RESULTS:", ""]
        report_lines.append("Detected Abnormalities:")
        
        for pathology, confidence in sorted(high_confidence.items(), key=lambda x: x[1], reverse=True):
            confidence_level = "High" if confidence >= 0.7 else "Moderate" if confidence >= 0.5 else "Low"
            report_lines.append(f"- {pathology}: {confidence:.3f} ({confidence_level} confidence)")
        
        report_lines.extend(["", "Note: This is an AI-generated classification and should be reviewed by a qualified radiologist."])
        
        return "\n".join(report_lines)
    
    def compare_classifications(self, results1: Dict[str, float], results2: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Compare two classification results"""
        comparison = {}
        all_labels = set(results1.keys()) | set(results2.keys())
        
        for label in all_labels:
            score1 = results1.get(label, 0.0)
            score2 = results2.get(label, 0.0)
            comparison[label] = {
                'image1': score1,
                'image2': score2,
                'difference': score2 - score1
            }
        
        return comparison
    
    def save_results(self, results: Dict[str, float], output_path: Union[str, Path]) -> None:
        """Save classification results to JSON file"""
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved classification results to {output_path}")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise
    
    def load_results(self, input_path: Union[str, Path]) -> Dict[str, float]:
        """Load classification results from JSON file"""
        try:
            with open(input_path, 'r') as f:
                results = json.load(f)
            logger.info(f"Loaded classification results from {input_path}")
            return results
        except Exception as e:
            logger.error(f"Error loading results: {str(e)}")
            raise
