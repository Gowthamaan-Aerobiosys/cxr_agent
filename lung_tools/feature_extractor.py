"""
CXR Feature Extractor
Extracts various features from chest X-ray images including radiological and anatomical features
"""

import numpy as np
import cv2
from scipy import ndimage
from skimage import feature, filters, measure
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
import logging
from pathlib import Path
import json

from .image_processor import CXRImageProcessor

logger = logging.getLogger(__name__)

class CXRFeatureExtractor:
    """Extract comprehensive features from CXR images"""
    
    def __init__(self):
        self.image_processor = CXRImageProcessor()
    
    def extract_all_features(self, image: np.ndarray, masks: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Dict]:
        """Extract all available features from CXR image"""
        features = {}
        
        # Basic image features
        features['basic'] = self.extract_basic_features(image)
        
        # Texture features
        features['texture'] = self.extract_texture_features(image)
        
        # Shape features (if masks provided)
        if masks:
            features['shape'] = self.extract_shape_features(masks)
        
        # Radiological features
        features['radiological'] = self.extract_radiological_features(image)
        
        # Histogram features
        features['histogram'] = self.extract_histogram_features(image)
        
        # Edge features
        features['edge'] = self.extract_edge_features(image)
        
        # Regional features (if masks provided)
        if masks:
            features['regional'] = self.extract_regional_features(image, masks)
        
        logger.info("Feature extraction completed")
        return features
    
    def extract_basic_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract basic statistical features from image"""
        features = {
            'mean_intensity': float(np.mean(image)),
            'std_intensity': float(np.std(image)),
            'min_intensity': float(np.min(image)),
            'max_intensity': float(np.max(image)),
            'median_intensity': float(np.median(image)),
            'intensity_range': float(np.max(image) - np.min(image)),
            'skewness': float(self._calculate_skewness(image)),
            'kurtosis': float(self._calculate_kurtosis(image)),
            'energy': float(np.sum(image ** 2)),
            'entropy': float(self._calculate_entropy(image))
        }
        return features
    
    def extract_texture_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract texture features using various methods"""
        features = {}
        
        # Local Binary Patterns
        lbp = feature.local_binary_pattern(image, P=8, R=1, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
        lbp_hist = lbp_hist.astype(float)
        lbp_hist /= (lbp_hist.sum() + 1e-6)
        
        for i, val in enumerate(lbp_hist):
            features[f'lbp_bin_{i}'] = float(val)
        
        # Gray Level Co-occurrence Matrix (GLCM) approximation
        glcm_features = self._calculate_glcm_features(image)
        features.update(glcm_features)
        
        # Gabor filter responses
        gabor_features = self._calculate_gabor_features(image)
        features.update(gabor_features)
        
        return features
    
    def extract_shape_features(self, masks: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Extract shape features from segmentation masks"""
        shape_features = {}
        
        for label, mask in masks.items():
            if label == 'background':
                continue
            
            features = {
                'area': float(np.sum(mask)),
                'perimeter': self._calculate_perimeter(mask),
                'circularity': self._calculate_circularity(mask),
                'eccentricity': self._calculate_eccentricity(mask),
                'solidity': self._calculate_solidity(mask),
                'extent': self._calculate_extent(mask),
                'aspect_ratio': self._calculate_aspect_ratio(mask),
                'compactness': self._calculate_compactness(mask)
            }
            
            shape_features[label] = features
        
        return shape_features
    
    def extract_radiological_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract radiological-specific features"""
        features = {}
        
        # Rib shadow detection (simplified)
        features['rib_shadow_intensity'] = self._detect_rib_shadows(image)
        
        # Cardiac shadow analysis
        features['cardiac_region_intensity'] = self._analyze_cardiac_region(image)
        
        # Diaphragm analysis
        features['diaphragm_sharpness'] = self._analyze_diaphragm_sharpness(image)
        
        # Costophrenic angle analysis
        features['costophrenic_angle_clarity'] = self._analyze_costophrenic_angles(image)
        
        # Lung field symmetry
        features['lung_field_symmetry'] = self._calculate_lung_symmetry(image)
        
        return features
    
    def extract_histogram_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract histogram-based features"""
        hist, _ = np.histogram(image, bins=256, range=(0, 256))
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-6)
        
        features = {
            'hist_mean': float(np.sum(hist * np.arange(256))),
            'hist_std': float(np.sqrt(np.sum(((np.arange(256) - np.sum(hist * np.arange(256))) ** 2) * hist))),
            'hist_skewness': float(self._calculate_hist_skewness(hist)),
            'hist_kurtosis': float(self._calculate_hist_kurtosis(hist)),
            'hist_energy': float(np.sum(hist ** 2)),
            'hist_entropy': float(-np.sum(hist * np.log(hist + 1e-6)))
        }
        
        # Percentile features
        for p in [10, 25, 50, 75, 90]:
            features[f'percentile_{p}'] = float(np.percentile(image, p))
        
        return features
    
    def extract_edge_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract edge-based features"""
        # Sobel edge detection
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Canny edge detection
        canny_edges = cv2.Canny(image, 50, 150)
        
        features = {
            'edge_density': float(np.sum(canny_edges > 0) / canny_edges.size),
            'mean_edge_strength': float(np.mean(sobel_magnitude)),
            'max_edge_strength': float(np.max(sobel_magnitude)),
            'edge_contrast': float(np.std(sobel_magnitude)),
            'horizontal_edges': float(np.sum(np.abs(sobel_x))),
            'vertical_edges': float(np.sum(np.abs(sobel_y)))
        }
        
        return features
    
    def extract_regional_features(self, image: np.ndarray, masks: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Extract features from specific regions defined by masks"""
        regional_features = {}
        
        for label, mask in masks.items():
            if label == 'background':
                continue
            
            # Apply mask to image
            masked_image = image * mask
            region_pixels = masked_image[mask > 0]
            
            if len(region_pixels) == 0:
                regional_features[label] = {}
                continue
            
            features = {
                'mean_intensity': float(np.mean(region_pixels)),
                'std_intensity': float(np.std(region_pixels)),
                'min_intensity': float(np.min(region_pixels)),
                'max_intensity': float(np.max(region_pixels)),
                'intensity_range': float(np.max(region_pixels) - np.min(region_pixels)),
                'region_contrast': float(np.std(region_pixels) / (np.mean(region_pixels) + 1e-6)),
                'region_homogeneity': float(1.0 / (1.0 + np.var(region_pixels))),
                'region_energy': float(np.sum(region_pixels ** 2) / len(region_pixels))
            }
            
            regional_features[label] = features
        
        return regional_features
    
    def _calculate_skewness(self, image: np.ndarray) -> float:
        """Calculate skewness of image intensities"""
        mean_val = np.mean(image)
        std_val = np.std(image)
        if std_val == 0:
            return 0.0
        return np.mean(((image - mean_val) / std_val) ** 3)
    
    def _calculate_kurtosis(self, image: np.ndarray) -> float:
        """Calculate kurtosis of image intensities"""
        mean_val = np.mean(image)
        std_val = np.std(image)
        if std_val == 0:
            return 0.0
        return np.mean(((image - mean_val) / std_val) ** 4) - 3
    
    def _calculate_entropy(self, image: np.ndarray) -> float:
        """Calculate entropy of image"""
        hist, _ = np.histogram(image, bins=256, range=(0, 256))
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-6)
        return -np.sum(hist * np.log(hist + 1e-6))
    
    def _calculate_glcm_features(self, image: np.ndarray) -> Dict[str, float]:
        """Calculate GLCM-based texture features (simplified)"""
        # Simplified GLCM calculation
        # In practice, you might want to use skimage.feature.greycomatrix
        
        # Quantize image to reduce computational complexity
        quantized = (image / 32).astype(np.int32)
        
        # Calculate co-occurrence in different directions
        features = {}
        
        # Horizontal co-occurrence
        h_cooc = self._simple_cooccurrence(quantized, dx=1, dy=0)
        features['glcm_contrast_h'] = float(self._glcm_contrast(h_cooc))
        features['glcm_homogeneity_h'] = float(self._glcm_homogeneity(h_cooc))
        
        # Vertical co-occurrence
        v_cooc = self._simple_cooccurrence(quantized, dx=0, dy=1)
        features['glcm_contrast_v'] = float(self._glcm_contrast(v_cooc))
        features['glcm_homogeneity_v'] = float(self._glcm_homogeneity(v_cooc))
        
        return features
    
    def _simple_cooccurrence(self, image: np.ndarray, dx: int, dy: int) -> np.ndarray:
        """Simple co-occurrence matrix calculation"""
        max_val = int(np.max(image)) + 1
        cooc = np.zeros((max_val, max_val))
        
        for i in range(image.shape[0] - abs(dy)):
            for j in range(image.shape[1] - abs(dx)):
                val1 = image[i, j]
                val2 = image[i + dy, j + dx]
                cooc[val1, val2] += 1
        
        return cooc / (cooc.sum() + 1e-6)
    
    def _glcm_contrast(self, cooc: np.ndarray) -> float:
        """Calculate contrast from co-occurrence matrix"""
        contrast = 0.0
        for i in range(cooc.shape[0]):
            for j in range(cooc.shape[1]):
                contrast += cooc[i, j] * (i - j) ** 2
        return contrast
    
    def _glcm_homogeneity(self, cooc: np.ndarray) -> float:
        """Calculate homogeneity from co-occurrence matrix"""
        homogeneity = 0.0
        for i in range(cooc.shape[0]):
            for j in range(cooc.shape[1]):
                homogeneity += cooc[i, j] / (1 + abs(i - j))
        return homogeneity
    
    def _calculate_gabor_features(self, image: np.ndarray) -> Dict[str, float]:
        """Calculate Gabor filter responses"""
        features = {}
        
        # Apply Gabor filters with different orientations
        for angle in [0, 45, 90, 135]:
            gabor_response = filters.gabor(image, frequency=0.1, theta=np.radians(angle))
            features[f'gabor_mean_{angle}'] = float(np.mean(np.abs(gabor_response[0])))
            features[f'gabor_std_{angle}'] = float(np.std(gabor_response[0]))
        
        return features
    
    def _detect_rib_shadows(self, image: np.ndarray) -> float:
        """Detect rib shadow intensity (simplified)"""
        # This is a simplified approach
        # Rib shadows typically appear as horizontal dark lines
        horizontal_filter = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]])
        filtered = cv2.filter2D(image, -1, horizontal_filter)
        return float(np.mean(np.abs(filtered)))
    
    def _analyze_cardiac_region(self, image: np.ndarray) -> float:
        """Analyze cardiac region (simplified)"""
        # Cardiac region is typically in the lower-left portion
        h, w = image.shape
        cardiac_region = image[int(h*0.4):int(h*0.8), int(w*0.3):int(w*0.7)]
        return float(np.mean(cardiac_region))
    
    def _analyze_diaphragm_sharpness(self, image: np.ndarray) -> float:
        """Analyze diaphragm sharpness (simplified)"""
        # Diaphragm is typically in the lower portion
        h, w = image.shape
        diaphragm_region = image[int(h*0.7):, :]
        
        # Calculate gradient to measure sharpness
        gradient = np.gradient(diaphragm_region.astype(float))
        sharpness = np.mean(np.abs(gradient[0]))  # Vertical gradient
        return float(sharpness)
    
    def _analyze_costophrenic_angles(self, image: np.ndarray) -> float:
        """Analyze costophrenic angle clarity (simplified)"""
        # Costophrenic angles are at the bottom corners
        h, w = image.shape
        left_angle = image[int(h*0.8):, :int(w*0.2)]
        right_angle = image[int(h*0.8):, int(w*0.8):]
        
        # Measure local contrast as a proxy for clarity
        left_contrast = np.std(left_angle)
        right_contrast = np.std(right_angle)
        
        return float((left_contrast + right_contrast) / 2)
    
    def _calculate_lung_symmetry(self, image: np.ndarray) -> float:
        """Calculate lung field symmetry"""
        h, w = image.shape
        left_lung = image[:, :w//2]
        right_lung = np.fliplr(image[:, w//2:])  # Flip for comparison
        
        # Resize to same size if needed
        min_w = min(left_lung.shape[1], right_lung.shape[1])
        left_lung = left_lung[:, :min_w]
        right_lung = right_lung[:, :min_w]
        
        # Calculate correlation as symmetry measure
        correlation = np.corrcoef(left_lung.flatten(), right_lung.flatten())[0, 1]
        return float(correlation if not np.isnan(correlation) else 0.0)
    
    def _calculate_hist_skewness(self, hist: np.ndarray) -> float:
        """Calculate skewness of histogram"""
        bins = np.arange(len(hist))
        mean_val = np.sum(hist * bins)
        std_val = np.sqrt(np.sum(hist * (bins - mean_val) ** 2))
        if std_val == 0:
            return 0.0
        return np.sum(hist * ((bins - mean_val) / std_val) ** 3)
    
    def _calculate_hist_kurtosis(self, hist: np.ndarray) -> float:
        """Calculate kurtosis of histogram"""
        bins = np.arange(len(hist))
        mean_val = np.sum(hist * bins)
        std_val = np.sqrt(np.sum(hist * (bins - mean_val) ** 2))
        if std_val == 0:
            return 0.0
        return np.sum(hist * ((bins - mean_val) / std_val) ** 4) - 3
    
    # Reuse shape calculation methods from segmentation module
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
    
    def _calculate_eccentricity(self, mask: np.ndarray) -> float:
        """Calculate eccentricity of mask"""
        moments = cv2.moments(mask)
        if moments['m00'] != 0:
            x = moments['m10'] / moments['m00']
            y = moments['m01'] / moments['m00']
            
            mu20 = moments['m20'] / moments['m00'] - x * x
            mu02 = moments['m02'] / moments['m00'] - y * y
            mu11 = moments['m11'] / moments['m00'] - x * y
            
            # Calculate eigenvalues
            theta = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
            a = (mu20 + mu02 + np.sqrt((mu20 - mu02)**2 + 4*mu11**2)) / 2
            b = (mu20 + mu02 - np.sqrt((mu20 - mu02)**2 + 4*mu11**2)) / 2
            
            if a > 0:
                return np.sqrt(1 - b/a)
        
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
    
    def _calculate_extent(self, mask: np.ndarray) -> float:
        """Calculate extent of mask"""
        area = np.sum(mask)
        y_coords, x_coords = np.where(mask)
        if len(y_coords) > 0:
            bbox_area = (np.max(x_coords) - np.min(x_coords)) * (np.max(y_coords) - np.min(y_coords))
            if bbox_area > 0:
                return area / bbox_area
        return 0.0
    
    def _calculate_aspect_ratio(self, mask: np.ndarray) -> float:
        """Calculate aspect ratio of mask"""
        y_coords, x_coords = np.where(mask)
        if len(y_coords) > 0:
            width = np.max(x_coords) - np.min(x_coords)
            height = np.max(y_coords) - np.min(y_coords)
            if height > 0:
                return width / height
        return 0.0
    
    def _calculate_compactness(self, mask: np.ndarray) -> float:
        """Calculate compactness of mask"""
        area = np.sum(mask)
        perimeter = self._calculate_perimeter(mask)
        if perimeter > 0:
            return (perimeter ** 2) / (4 * np.pi * area)
        return 0.0
    
    def save_features(self, features: Dict, output_path: Union[str, Path]) -> None:
        """Save extracted features to JSON file"""
        try:
            with open(output_path, 'w') as f:
                json.dump(features, f, indent=2)
            logger.info(f"Saved features to {output_path}")
        except Exception as e:
            logger.error(f"Error saving features: {str(e)}")
            raise
    
    def features_to_dataframe(self, features: Dict) -> pd.DataFrame:
        """Convert features to pandas DataFrame"""
        flattened_features = {}
        
        def flatten_dict(d, parent_key='', sep='_'):
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    flatten_dict(v, new_key, sep=sep)
                else:
                    flattened_features[new_key] = v
        
        flatten_dict(features)
        return pd.DataFrame([flattened_features])
