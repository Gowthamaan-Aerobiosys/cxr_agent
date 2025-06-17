"""
Lung Segmentation
Provides segmentation capabilities for chest X-ray images including lung field segmentation
and anatomical structure identification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import cv2
from typing import Dict, List, Tuple, Union, Optional
import logging
from pathlib import Path
import json
from scipy import ndimage
from skimage import measure, morphology
from skimage.segmentation import watershed
from skimage.feature import peak_local_maxima

from .image_processor import CXRImageProcessor

logger = logging.getLogger(__name__)

class UNet(nn.Module):
    """U-Net architecture for lung segmentation"""
    
    def __init__(self, in_channels: int = 1, out_channels: int = 3, features: List[int] = [64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Down part of UNET (encoder)
        for feature in features:
            self.downs.append(self._double_conv(in_channels, feature))
            in_channels = feature
        
        # Up part of UNET (decoder)
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(self._double_conv(feature*2, feature))
        
        self.bottleneck = self._double_conv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def _double_conv(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Double convolution block"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        skip_connections = []
        
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            
            if x.shape != skip_connection.shape:
                x = transforms.functional.resize(x, size=skip_connection.shape[2:])
            
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        
        return self.final_conv(x)

class LungSegmenter:
    """Lung segmentation class for CXR images"""
    
    def __init__(self, model_type: str = "unet"):
        self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_processor = CXRImageProcessor()
        
        # Segmentation classes
        self.segment_labels = {
            0: 'background',
            1: 'left_lung',
            2: 'right_lung'
        }
        
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the segmentation model"""
        try:
            if self.model_type == "unet":
                self.model = UNet(in_channels=1, out_channels=3)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Loaded {self.model_type} segmentation model")
        
        except Exception as e:
            logger.error(f"Error loading segmentation model: {str(e)}")
            raise
    
    def segment_lungs(self, image_path: Union[str, Path, np.ndarray]) -> Dict[str, np.ndarray]:
        """Segment lung fields in CXR image"""
        try:
            # Load and preprocess image
            if isinstance(image_path, np.ndarray):
                image = image_path
            else:
                image = self.image_processor.load_image(image_path)
            
            original_shape = image.shape
            
            # Preprocess for model
            input_tensor = self.image_processor.preprocess_for_model(image)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                predictions = torch.softmax(outputs, dim=1)
                predicted_mask = torch.argmax(predictions, dim=1).cpu().numpy()[0]
            
            # Resize back to original shape
            predicted_mask = cv2.resize(predicted_mask.astype(np.uint8), 
                                      (original_shape[1], original_shape[0]), 
                                      interpolation=cv2.INTER_NEAREST)
            
            # Extract individual lung masks
            masks = {}
            for class_id, label in self.segment_labels.items():
                mask = (predicted_mask == class_id).astype(np.uint8)
                masks[label] = mask
            
            # Post-process masks
            masks = self._post_process_masks(masks)
            
            logger.info("Lung segmentation completed")
            return masks
        
        except Exception as e:
            logger.error(f"Error in lung segmentation: {str(e)}")
            raise
    
    def _post_process_masks(self, masks: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Post-process segmentation masks"""
        processed_masks = {}
        
        for label, mask in masks.items():
            if label == 'background':
                processed_masks[label] = mask
                continue
            
            # Remove small components
            mask = morphology.remove_small_objects(mask.astype(bool), min_size=1000)
            
            # Fill holes
            mask = ndimage.binary_fill_holes(mask)
            
            # Smooth boundaries
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            processed_masks[label] = mask
        
        return processed_masks
    
    def extract_lung_features(self, image: np.ndarray, masks: Dict[str, np.ndarray]) -> Dict[str, Dict]:
        """Extract features from segmented lung regions"""
        features = {}
        
        for label, mask in masks.items():
            if label == 'background':
                continue
            
            # Apply mask to image
            masked_image = image * mask
            
            # Calculate features
            lung_features = {
                'area': np.sum(mask),
                'mean_intensity': np.mean(masked_image[mask > 0]) if np.sum(mask) > 0 else 0,
                'std_intensity': np.std(masked_image[mask > 0]) if np.sum(mask) > 0 else 0,
                'min_intensity': np.min(masked_image[mask > 0]) if np.sum(mask) > 0 else 0,
                'max_intensity': np.max(masked_image[mask > 0]) if np.sum(mask) > 0 else 0,
                'perimeter': self._calculate_perimeter(mask),
                'circularity': self._calculate_circularity(mask),
                'solidity': self._calculate_solidity(mask)
            }
            
            features[label] = lung_features
        
        return features
    
    def _calculate_perimeter(self, mask: np.ndarray) -> float:
        """Calculate perimeter of mask"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            return cv2.arcLength(contours[0], True)
        return 0.0
    
    def _calculate_circularity(self, mask: np.ndarray) -> float:
        """Calculate circularity of mask"""
        area = np.sum(mask)
        perimeter = self._calculate_perimeter(mask)
        if perimeter > 0:
            return 4 * np.pi * area / (perimeter ** 2)
        return 0.0
    
    def _calculate_solidity(self, mask: np.ndarray) -> float:
        """Calculate solidity of mask"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            area = cv2.contourArea(contours[0])
            hull = cv2.convexHull(contours[0])
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                return area / hull_area
        return 0.0
    
    def visualize_segmentation(self, image: np.ndarray, masks: Dict[str, np.ndarray]) -> np.ndarray:
        """Create visualization of segmentation results"""
        # Create colored overlay
        overlay = np.zeros((*image.shape, 3), dtype=np.uint8)
        
        # Color mapping
        colors = {
            'left_lung': (255, 0, 0),   # Red
            'right_lung': (0, 255, 0),  # Green
            'background': (0, 0, 0)     # Black
        }
        
        for label, mask in masks.items():
            if label in colors:
                color = colors[label]
                overlay[mask > 0] = color
        
        # Convert original image to 3-channel
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image
        
        # Blend images
        result = cv2.addWeighted(image_rgb, 0.7, overlay, 0.3, 0)
        
        return result
    
    def calculate_lung_volume_ratio(self, masks: Dict[str, np.ndarray]) -> float:
        """Calculate the ratio of left to right lung volume"""
        left_area = np.sum(masks.get('left_lung', np.zeros((1, 1))))
        right_area = np.sum(masks.get('right_lung', np.zeros((1, 1))))
        
        if right_area > 0:
            return left_area / right_area
        return 0.0
    
    def detect_pneumothorax(self, image: np.ndarray, masks: Dict[str, np.ndarray], threshold: float = 0.15) -> Dict[str, bool]:
        """Simple pneumothorax detection based on lung area reduction"""
        results = {}
        
        # Expected normal lung areas (rough estimates)
        expected_left_area = 0.22 * image.shape[0] * image.shape[1]  # ~22% of image
        expected_right_area = 0.25 * image.shape[0] * image.shape[1]  # ~25% of image
        
        left_area = np.sum(masks.get('left_lung', np.zeros((1, 1))))
        right_area = np.sum(masks.get('right_lung', np.zeros((1, 1))))
        
        left_reduction = (expected_left_area - left_area) / expected_left_area
        right_reduction = (expected_right_area - right_area) / expected_right_area
        
        results['left_pneumothorax'] = left_reduction > threshold
        results['right_pneumothorax'] = right_reduction > threshold
        results['left_area_reduction'] = left_reduction
        results['right_area_reduction'] = right_reduction
        
        return results
    
    def save_masks(self, masks: Dict[str, np.ndarray], output_dir: Union[str, Path]) -> None:
        """Save segmentation masks to files"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for label, mask in masks.items():
            mask_path = output_dir / f"{label}_mask.png"
            cv2.imwrite(str(mask_path), (mask * 255).astype(np.uint8))
        
        logger.info(f"Saved segmentation masks to {output_dir}")
    
    def save_features(self, features: Dict[str, Dict], output_path: Union[str, Path]) -> None:
        """Save extracted features to JSON file"""
        try:
            with open(output_path, 'w') as f:
                json.dump(features, f, indent=2)
            logger.info(f"Saved lung features to {output_path}")
        except Exception as e:
            logger.error(f"Error saving features: {str(e)}")
            raise
