"""
CXR Image Processor
Handles loading, preprocessing, and enhancement of chest X-ray images
"""

import numpy as np
import cv2
from PIL import Image
from typing import Union, Tuple, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class CXRImageProcessor:
    """Image processor for chest X-ray images"""

    def __init__(self, target_size: Tuple[int, int] = (512, 512)):
        self.target_size = target_size

    def load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Load image from path

        Args:
            image_path: Path to image file

        Returns:
            Grayscale image as numpy array
        """
        try:
            if isinstance(image_path, str):
                image_path = Path(image_path)

            # Load with PIL
            img = Image.open(image_path)

            # Convert to grayscale
            if img.mode != "L":
                img = img.convert("L")

            # Convert to numpy array
            img_array = np.array(img, dtype=np.uint8)

            logger.debug(f"Loaded image: {image_path}, shape: {img_array.shape}")
            return img_array

        except Exception as e:
            logger.error(f"Error loading image: {e}")
            raise

    def preprocess(
        self,
        image: np.ndarray,
        resize: bool = True,
        normalize: bool = True,
        enhance: bool = True,
    ) -> np.ndarray:
        """
        Preprocess CXR image

        Args:
            image: Input image
            resize: Whether to resize
            normalize: Whether to normalize
            enhance: Whether to enhance contrast

        Returns:
            Preprocessed image
        """
        processed = image.copy()

        # Resize
        if resize and processed.shape[:2] != self.target_size:
            processed = cv2.resize(
                processed, self.target_size, interpolation=cv2.INTER_LINEAR
            )

        # Normalize
        if normalize:
            processed = self.normalize_image(processed)

        # Enhance
        if enhance:
            processed = self.enhance_contrast(processed)

        return processed

    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 255] range"""
        img_min = np.min(image)
        img_max = np.max(image)

        if img_max > img_min:
            normalized = ((image - img_min) / (img_max - img_min) * 255).astype(
                np.uint8
            )
        else:
            normalized = image.astype(np.uint8)

        return normalized

    def enhance_contrast(
        self, image: np.ndarray, clip_limit: float = 2.0
    ) -> np.ndarray:
        """
        Enhance image contrast using CLAHE

        Args:
            image: Input image
            clip_limit: Contrast limit

        Returns:
            Enhanced image
        """
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        return enhanced

    def denoise(self, image: np.ndarray, strength: int = 10) -> np.ndarray:
        """
        Denoise image

        Args:
            image: Input image
            strength: Denoising strength

        Returns:
            Denoised image
        """
        denoised = cv2.fastNlMeansDenoising(image, None, strength, 7, 21)
        return denoised

    def equalize_histogram(self, image: np.ndarray) -> np.ndarray:
        """Equalize image histogram"""
        equalized = cv2.equalizeHist(image)
        return equalized

    def resize_image(self, image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Resize image to specified size"""
        resized = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        return resized
