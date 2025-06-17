"""
CXR Image Processor
Handles loading, preprocessing, and basic operations on chest X-ray images
"""

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from typing import Union, Tuple, Optional, List
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class CXRImageProcessor:
    """Base class for CXR image processing operations"""
    
    def __init__(self, target_size: Tuple[int, int] = (512, 512)):
        self.target_size = target_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Standard preprocessing pipeline for medical images
        self.transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])  # Grayscale normalization
        ])
        
        # Transform for visualization
        self.viz_transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor()
        ])
    
    def load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """Load and preprocess CXR image"""
        try:
            # Load image
            if isinstance(image_path, str):
                image_path = Path(image_path)
            
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            logger.info(f"Loaded CXR image: {image_path} with shape {image.shape}")
            return image
        
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            raise
    
    def preprocess_for_model(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model inference"""
        # Convert to PIL Image
        pil_image = Image.fromarray(image)
        
        # Apply transforms
        tensor = self.transform(pil_image)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor.to(self.device)
    
    def preprocess_for_visualization(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for visualization"""
        pil_image = Image.fromarray(image)
        tensor = self.viz_transform(pil_image)
        return tensor
    
    def enhance_contrast(self, image: np.ndarray, alpha: float = 1.2, beta: int = 20) -> np.ndarray:
        """Enhance image contrast using linear transformation"""
        enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return enhanced
    
    def apply_clahe(self, image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """Apply Contrast Limited Adaptive Histogram Equalization"""
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced = clahe.apply(image)
        return enhanced
    
    def denoise_image(self, image: np.ndarray, h: int = 10, template_window_size: int = 7, search_window_size: int = 21) -> np.ndarray:
        """Denoise image using Non-local Means Denoising"""
        denoised = cv2.fastNlMeansDenoising(image, None, h, template_window_size, search_window_size)
        return denoised
    
    def resize_image(self, image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Resize image to target size"""
        resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        return resized
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to 0-255 range"""
        normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        return normalized.astype(np.uint8)
    
    def extract_roi(self, image: np.ndarray, roi_coords: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract region of interest from image"""
        x, y, w, h = roi_coords
        roi = image[y:y+h, x:x+w]
        return roi
    
    def get_image_stats(self, image: np.ndarray) -> dict:
        """Get basic statistics of the image"""
        stats = {
            'shape': image.shape,
            'dtype': str(image.dtype),
            'min': float(np.min(image)),
            'max': float(np.max(image)),
            'mean': float(np.mean(image)),
            'std': float(np.std(image)),
            'size_mb': image.nbytes / (1024 * 1024)
        }
        return stats
    
    def save_image(self, image: np.ndarray, output_path: Union[str, Path]) -> None:
        """Save processed image"""
        try:
            cv2.imwrite(str(output_path), image)
            logger.info(f"Saved processed image to {output_path}")
        except Exception as e:
            logger.error(f"Error saving image to {output_path}: {str(e)}")
            raise
    
    def batch_preprocess(self, image_paths: List[Union[str, Path]]) -> List[torch.Tensor]:
        """Preprocess multiple images in batch"""
        processed_images = []
        
        for image_path in image_paths:
            try:
                image = self.load_image(image_path)
                processed = self.preprocess_for_model(image)
                processed_images.append(processed)
            except Exception as e:
                logger.error(f"Error processing image {image_path}: {str(e)}")
                continue
        
        return processed_images
    
    def create_image_grid(self, images: List[np.ndarray], grid_size: Tuple[int, int]) -> np.ndarray:
        """Create a grid of images for visualization"""
        rows, cols = grid_size
        if len(images) > rows * cols:
            images = images[:rows * cols]
        
        # Ensure all images have the same size
        target_h, target_w = self.target_size
        resized_images = [self.resize_image(img, (target_w, target_h)) for img in images]
        
        # Create grid
        grid_rows = []
        for i in range(0, len(resized_images), cols):
            row_images = resized_images[i:i+cols]
            # Pad row if necessary
            while len(row_images) < cols:
                row_images.append(np.zeros((target_h, target_w), dtype=np.uint8))
            
            row = np.hstack(row_images)
            grid_rows.append(row)
        
        # Create final grid
        grid = np.vstack(grid_rows)
        return grid
