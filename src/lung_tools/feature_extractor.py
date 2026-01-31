"""
Feature Extractor
Extracts radiological and quantitative features from CXR images
"""

import numpy as np
import cv2
from typing import Dict, Optional
import logging
from scipy import ndimage
from skimage import measure

logger = logging.getLogger(__name__)


class CXRFeatureExtractor:
    """Feature extraction for CXR analysis"""

    def __init__(self):
        pass

    def extract_all_features(
        self, image: np.ndarray, masks: Dict[str, np.ndarray]
    ) -> Dict[str, Dict]:
        """
        Extract all features from CXR image

        Args:
            image: Input CXR image
            masks: Segmentation masks

        Returns:
            Dictionary of feature categories
        """
        features = {}

        features["radiological"] = self.extract_radiological_features(image, masks)
        features["texture"] = self.extract_texture_features(image, masks)
        features["shape"] = self.extract_shape_features(masks)
        features["intensity"] = self.extract_intensity_features(image, masks)

        return features

    def extract_radiological_features(
        self, image: np.ndarray, masks: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Extract radiological features"""
        features = {}

        # Lung field symmetry
        left_area = np.sum(masks.get("left_lung", 0))
        right_area = np.sum(masks.get("right_lung", 0))

        if max(left_area, right_area) > 0:
            features["lung_field_symmetry"] = float(
                min(left_area, right_area) / max(left_area, right_area)
            )
        else:
            features["lung_field_symmetry"] = 0.0

        # Costophrenic angle clarity (simplified)
        features["costophrenic_angle_clarity"] = self._estimate_costophrenic_clarity(
            image, masks
        )

        # Hemidiaphragm level
        features["hemidiaphragm_symmetry"] = self._estimate_hemidiaphragm_symmetry(
            masks
        )

        return features

    def extract_texture_features(
        self, image: np.ndarray, masks: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Extract texture features"""
        features = {}

        lung_mask = masks.get("both_lungs", np.zeros_like(image))

        if np.sum(lung_mask) > 0:
            lung_pixels = image[lung_mask > 0]

            # Statistical features
            features["mean_intensity"] = float(np.mean(lung_pixels))
            features["std_intensity"] = float(np.std(lung_pixels))
            features["entropy"] = float(self._calculate_entropy(lung_pixels))
            features["contrast"] = float(
                np.std(lung_pixels) / (np.mean(lung_pixels) + 1e-6)
            )
        else:
            features["mean_intensity"] = 0.0
            features["std_intensity"] = 0.0
            features["entropy"] = 0.0
            features["contrast"] = 0.0

        return features

    def extract_shape_features(self, masks: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Extract shape features"""
        features = {}

        for side in ["left_lung", "right_lung"]:
            mask = masks.get(side, np.zeros((1, 1)))

            if np.sum(mask) > 0:
                # Area
                area = float(np.sum(mask))

                # Perimeter
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if contours:
                    perimeter = float(cv2.arcLength(contours[0], True))

                    # Compactness
                    if perimeter > 0:
                        compactness = (4 * np.pi * area) / (perimeter**2)
                    else:
                        compactness = 0.0

                    features[f"{side}_area"] = area
                    features[f"{side}_perimeter"] = perimeter
                    features[f"{side}_compactness"] = compactness
                else:
                    features[f"{side}_area"] = area
                    features[f"{side}_perimeter"] = 0.0
                    features[f"{side}_compactness"] = 0.0
            else:
                features[f"{side}_area"] = 0.0
                features[f"{side}_perimeter"] = 0.0
                features[f"{side}_compactness"] = 0.0

        return features

    def extract_intensity_features(
        self, image: np.ndarray, masks: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Extract intensity-based features"""
        features = {}

        for side in ["left_lung", "right_lung"]:
            mask = masks.get(side, np.zeros_like(image))

            if np.sum(mask) > 0:
                lung_pixels = image[mask > 0]

                features[f"{side}_mean"] = float(np.mean(lung_pixels))
                features[f"{side}_median"] = float(np.median(lung_pixels))
                features[f"{side}_std"] = float(np.std(lung_pixels))
                features[f"{side}_min"] = float(np.min(lung_pixels))
                features[f"{side}_max"] = float(np.max(lung_pixels))
            else:
                features[f"{side}_mean"] = 0.0
                features[f"{side}_median"] = 0.0
                features[f"{side}_std"] = 0.0
                features[f"{side}_min"] = 0.0
                features[f"{side}_max"] = 0.0

        return features

    def _estimate_costophrenic_clarity(
        self, image: np.ndarray, masks: Dict[str, np.ndarray]
    ) -> float:
        """Estimate costophrenic angle clarity (simplified)"""
        h, w = image.shape

        # Look at lower corners of lung fields
        lower_region = image[int(h * 0.7) :, :]

        # Edge detection
        edges = cv2.Canny(lower_region, 50, 150)

        # Count edges in costophrenic angle regions
        left_corner = edges[:, : int(w * 0.3)]
        right_corner = edges[:, int(w * 0.7) :]

        left_clarity = np.sum(left_corner) / (left_corner.size + 1e-6)
        right_clarity = np.sum(right_corner) / (right_corner.size + 1e-6)

        # Average clarity (normalized)
        clarity = (left_clarity + right_clarity) / 2.0 * 1000  # Scale up

        return float(clarity)

    def _estimate_hemidiaphragm_symmetry(self, masks: Dict[str, np.ndarray]) -> float:
        """Estimate hemidiaphragm symmetry"""
        left_mask = masks.get("left_lung", np.zeros((1, 1)))
        right_mask = masks.get("right_lung", np.zeros((1, 1)))

        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return 0.0

        # Find lowest points of each lung (highest y-coordinate)
        left_points = np.where(left_mask > 0)
        right_points = np.where(right_mask > 0)

        left_lowest = np.max(left_points[0]) if len(left_points[0]) > 0 else 0
        right_lowest = np.max(right_points[0]) if len(right_points[0]) > 0 else 0

        # Calculate symmetry
        if max(left_lowest, right_lowest) > 0:
            symmetry = min(left_lowest, right_lowest) / max(left_lowest, right_lowest)
        else:
            symmetry = 0.0

        return float(symmetry)

    def _calculate_entropy(self, pixels: np.ndarray) -> float:
        """Calculate Shannon entropy of pixel intensities"""
        hist, _ = np.histogram(pixels, bins=256, range=(0, 256))
        hist = hist / np.sum(hist)  # Normalize

        # Remove zeros to avoid log(0)
        hist = hist[hist > 0]

        entropy = -np.sum(hist * np.log2(hist))
        return entropy
