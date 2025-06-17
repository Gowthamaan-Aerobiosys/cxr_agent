"""
Pathology Detector
Advanced pathology detection for CXR images using multiple detection strategies
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Union, Optional
import logging
from pathlib import Path
import json
from scipy import ndimage
from skimage import measure, morphology, filters

from .image_processor import CXRImageProcessor
from .classifier import CXRClassifier
from .segmentation import LungSegmenter
from .feature_extractor import CXRFeatureExtractor

logger = logging.getLogger(__name__)

class PathologyDetector:
    """Advanced pathology detection for CXR images"""
    
    def __init__(self):
        self.image_processor = CXRImageProcessor()
        self.classifier = CXRClassifier()
        self.segmenter = LungSegmenter()
        self.feature_extractor = CXRFeatureExtractor()
        
        # Pathology-specific thresholds and parameters
        self.pathology_params = {
            'pneumothorax': {
                'area_reduction_threshold': 0.15,
                'edge_intensity_threshold': 0.3,
                'symmetry_threshold': 0.8
            },
            'pleural_effusion': {
                'opacity_threshold': 0.4,
                'lower_region_ratio': 0.3,
                'gradient_threshold': 0.2
            },
            'pneumonia': {
                'opacity_threshold': 0.35,
                'heterogeneity_threshold': 0.25,
                'regional_contrast_threshold': 0.15
            },
            'cardiomegaly': {
                'ctr_threshold': 0.5,  # Cardiothoracic ratio
                'cardiac_area_threshold': 0.25
            },
            'atelectasis': {
                'volume_loss_threshold': 0.2,
                'density_increase_threshold': 0.3,
                'shift_threshold': 0.1
            }
        }
    
    def detect_all_pathologies(self, image_path: Union[str, Path, np.ndarray]) -> Dict[str, Dict]:
        """Detect all pathologies in CXR image"""
        try:
            # Load and preprocess image
            if isinstance(image_path, np.ndarray):
                image = image_path
            else:
                image = self.image_processor.load_image(image_path)
            
            # Get segmentation masks
            masks = self.segmenter.segment_lungs(image)
            
            # Get classification results
            classification_results = self.classifier.classify_image(image)
            
            # Extract features
            features = self.feature_extractor.extract_all_features(image, masks)
            
            # Detect specific pathologies
            pathology_results = {}
            
            pathology_results['pneumothorax'] = self.detect_pneumothorax(image, masks, features)
            pathology_results['pleural_effusion'] = self.detect_pleural_effusion(image, masks, features)
            pathology_results['pneumonia'] = self.detect_pneumonia(image, masks, features)
            pathology_results['cardiomegaly'] = self.detect_cardiomegaly(image, masks, features)
            pathology_results['atelectasis'] = self.detect_atelectasis(image, masks, features)
            pathology_results['pulmonary_edema'] = self.detect_pulmonary_edema(image, masks, features)
            pathology_results['consolidation'] = self.detect_consolidation(image, masks, features)
            
            # Combine with classification results
            combined_results = {
                'classification': classification_results,
                'rule_based_detection': pathology_results,
                'confidence_scores': self._calculate_confidence_scores(classification_results, pathology_results),
                'final_diagnosis': self._generate_final_diagnosis(classification_results, pathology_results)
            }
            
            logger.info("Pathology detection completed")
            return combined_results
        
        except Exception as e:
            logger.error(f"Error in pathology detection: {str(e)}")
            raise
    
    def detect_pneumothorax(self, image: np.ndarray, masks: Dict[str, np.ndarray], features: Dict) -> Dict[str, Union[bool, float]]:
        """Detect pneumothorax using multiple indicators"""
        results = {
            'detected': False,
            'confidence': 0.0,
            'indicators': {}
        }
        
        # Check lung area reduction
        total_image_area = image.shape[0] * image.shape[1]
        left_lung_area = np.sum(masks.get('left_lung', np.zeros((1,1))))
        right_lung_area = np.sum(masks.get('right_lung', np.zeros((1,1))))
        
        expected_lung_area = 0.45 * total_image_area  # Normal lung area ~45% of image
        actual_lung_area = left_lung_area + right_lung_area
        area_reduction = (expected_lung_area - actual_lung_area) / expected_lung_area
        
        results['indicators']['area_reduction'] = float(area_reduction)
        
        # Check lung field asymmetry
        lung_symmetry = features.get('radiological', {}).get('lung_field_symmetry', 1.0)
        results['indicators']['asymmetry'] = float(1.0 - lung_symmetry)
        
        # Check for pleural line (simplified detection)
        pleural_line_score = self._detect_pleural_line(image, masks)
        results['indicators']['pleural_line'] = float(pleural_line_score)
        
        # Calculate confidence
        confidence_factors = []
        
        if area_reduction > self.pathology_params['pneumothorax']['area_reduction_threshold']:
            confidence_factors.append(0.4)
        
        if lung_symmetry < self.pathology_params['pneumothorax']['symmetry_threshold']:
            confidence_factors.append(0.3)
        
        if pleural_line_score > self.pathology_params['pneumothorax']['edge_intensity_threshold']:
            confidence_factors.append(0.3)
        
        results['confidence'] = sum(confidence_factors)
        results['detected'] = results['confidence'] > 0.5
        
        return results
    
    def detect_pleural_effusion(self, image: np.ndarray, masks: Dict[str, np.ndarray], features: Dict) -> Dict[str, Union[bool, float]]:
        """Detect pleural effusion"""
        results = {
            'detected': False,
            'confidence': 0.0,
            'indicators': {}
        }
        
        # Check for increased opacity in lower lung fields
        h = image.shape[0]
        lower_region = image[int(h*0.6):, :]
        upper_region = image[:int(h*0.4), :]
        
        lower_opacity = np.mean(lower_region)
        upper_opacity = np.mean(upper_region)
        opacity_ratio = lower_opacity / (upper_opacity + 1e-6)
        
        results['indicators']['lower_opacity_ratio'] = float(opacity_ratio)
        
        # Check for blunted costophrenic angles
        costophrenic_clarity = features.get('radiological', {}).get('costophrenic_angle_clarity', 0.0)
        results['indicators']['costophrenic_blunting'] = float(1.0 / (costophrenic_clarity + 1.0))
        
        # Check for meniscus sign (simplified)
        meniscus_score = self._detect_meniscus_sign(image)
        results['indicators']['meniscus_sign'] = float(meniscus_score)
        
        # Calculate confidence
        confidence_factors = []
        
        if opacity_ratio > 1.2:  # Lower region significantly more opaque
            confidence_factors.append(0.4)
        
        if costophrenic_clarity < 20:  # Blunted angles
            confidence_factors.append(0.3)
        
        if meniscus_score > 0.3:
            confidence_factors.append(0.3)
        
        results['confidence'] = sum(confidence_factors)
        results['detected'] = results['confidence'] > 0.5
        
        return results
    
    def detect_pneumonia(self, image: np.ndarray, masks: Dict[str, np.ndarray], features: Dict) -> Dict[str, Union[bool, float]]:
        """Detect pneumonia/consolidation"""
        results = {
            'detected': False,
            'confidence': 0.0,
            'indicators': {}
        }
        
        # Check for focal opacity
        focal_opacity_score = self._detect_focal_opacity(image, masks)
        results['indicators']['focal_opacity'] = float(focal_opacity_score)
        
        # Check air bronchogram (simplified)
        air_bronchogram_score = self._detect_air_bronchogram(image, masks)
        results['indicators']['air_bronchogram'] = float(air_bronchogram_score)
        
        # Check regional heterogeneity
        heterogeneity_score = self._calculate_regional_heterogeneity(image, masks)
        results['indicators']['heterogeneity'] = float(heterogeneity_score)
        
        # Calculate confidence
        confidence_factors = []
        
        if focal_opacity_score > self.pathology_params['pneumonia']['opacity_threshold']:
            confidence_factors.append(0.4)
        
        if air_bronchogram_score > 0.2:
            confidence_factors.append(0.3)
        
        if heterogeneity_score > self.pathology_params['pneumonia']['heterogeneity_threshold']:
            confidence_factors.append(0.3)
        
        results['confidence'] = sum(confidence_factors)
        results['detected'] = results['confidence'] > 0.5
        
        return results
    
    def detect_cardiomegaly(self, image: np.ndarray, masks: Dict[str, np.ndarray], features: Dict) -> Dict[str, Union[bool, float]]:
        """Detect cardiomegaly"""
        results = {
            'detected': False,
            'confidence': 0.0,
            'indicators': {}
        }
        
        # Calculate cardiothoracic ratio (simplified)
        ctr = self._calculate_cardiothoracic_ratio(image)
        results['indicators']['cardiothoracic_ratio'] = float(ctr)
        
        # Check cardiac region size
        cardiac_area_ratio = self._calculate_cardiac_area_ratio(image)
        results['indicators']['cardiac_area_ratio'] = float(cardiac_area_ratio)
        
        # Calculate confidence
        confidence_factors = []
        
        if ctr > self.pathology_params['cardiomegaly']['ctr_threshold']:
            confidence_factors.append(0.6)
        
        if cardiac_area_ratio > self.pathology_params['cardiomegaly']['cardiac_area_threshold']:
            confidence_factors.append(0.4)
        
        results['confidence'] = sum(confidence_factors)
        results['detected'] = results['confidence'] > 0.5
        
        return results
    
    def detect_atelectasis(self, image: np.ndarray, masks: Dict[str, np.ndarray], features: Dict) -> Dict[str, Union[bool, float]]:
        """Detect atelectasis"""
        results = {
            'detected': False,
            'confidence': 0.0,
            'indicators': {}
        }
        
        # Check for volume loss
        volume_loss_score = self._detect_volume_loss(image, masks)
        results['indicators']['volume_loss'] = float(volume_loss_score)
        
        # Check for increased density
        density_increase_score = self._detect_density_increase(image, masks)
        results['indicators']['density_increase'] = float(density_increase_score)
        
        # Check for mediastinal shift (simplified)
        shift_score = self._detect_mediastinal_shift(image)
        results['indicators']['mediastinal_shift'] = float(shift_score)
        
        # Calculate confidence
        confidence_factors = []
        
        if volume_loss_score > self.pathology_params['atelectasis']['volume_loss_threshold']:
            confidence_factors.append(0.4)
        
        if density_increase_score > self.pathology_params['atelectasis']['density_increase_threshold']:
            confidence_factors.append(0.3)
        
        if shift_score > self.pathology_params['atelectasis']['shift_threshold']:
            confidence_factors.append(0.3)
        
        results['confidence'] = sum(confidence_factors)
        results['detected'] = results['confidence'] > 0.5
        
        return results
    
    def detect_pulmonary_edema(self, image: np.ndarray, masks: Dict[str, np.ndarray], features: Dict) -> Dict[str, Union[bool, float]]:
        """Detect pulmonary edema"""
        results = {
            'detected': False,
            'confidence': 0.0,
            'indicators': {}
        }
        
        # Check for bilateral hazy opacities
        bilateral_haze_score = self._detect_bilateral_haze(image, masks)
        results['indicators']['bilateral_haze'] = float(bilateral_haze_score)
        
        # Check for bat wing pattern (simplified)
        bat_wing_score = self._detect_bat_wing_pattern(image, masks)
        results['indicators']['bat_wing_pattern'] = float(bat_wing_score)
        
        # Check for Kerley lines (simplified)
        kerley_lines_score = self._detect_kerley_lines(image)
        results['indicators']['kerley_lines'] = float(kerley_lines_score)
        
        # Calculate confidence
        confidence_factors = []
        
        if bilateral_haze_score > 0.3:
            confidence_factors.append(0.4)
        
        if bat_wing_score > 0.2:
            confidence_factors.append(0.3)
        
        if kerley_lines_score > 0.2:
            confidence_factors.append(0.3)
        
        results['confidence'] = sum(confidence_factors)
        results['detected'] = results['confidence'] > 0.5
        
        return results
    
    def detect_consolidation(self, image: np.ndarray, masks: Dict[str, np.ndarray], features: Dict) -> Dict[str, Union[bool, float]]:
        """Detect consolidation"""
        results = {
            'detected': False,
            'confidence': 0.0,
            'indicators': {}
        }
        
        # Check for homogeneous opacity
        homogeneous_opacity_score = self._detect_homogeneous_opacity(image, masks)
        results['indicators']['homogeneous_opacity'] = float(homogeneous_opacity_score)
        
        # Check for silhouette sign
        silhouette_sign_score = self._detect_silhouette_sign(image)
        results['indicators']['silhouette_sign'] = float(silhouette_sign_score)
        
        # Calculate confidence
        confidence_factors = []
        
        if homogeneous_opacity_score > 0.4:
            confidence_factors.append(0.6)
        
        if silhouette_sign_score > 0.3:
            confidence_factors.append(0.4)
        
        results['confidence'] = sum(confidence_factors)
        results['detected'] = results['confidence'] > 0.5
        
        return results
    
    # Helper methods for specific pathology detection
    
    def _detect_pleural_line(self, image: np.ndarray, masks: Dict[str, np.ndarray]) -> float:
        """Detect pleural line for pneumothorax"""
        # Edge detection in lung periphery
        edges = cv2.Canny(image, 50, 150)
        
        # Focus on lung periphery
        lung_mask = masks.get('left_lung', np.zeros_like(image)) + masks.get('right_lung', np.zeros_like(image))
        lung_boundary = cv2.dilate(lung_mask, np.ones((5,5)), iterations=1) - lung_mask
        
        peripheral_edges = edges * lung_boundary
        return np.sum(peripheral_edges) / (np.sum(lung_boundary) + 1e-6)
    
    def _detect_meniscus_sign(self, image: np.ndarray) -> float:
        """Detect meniscus sign for pleural effusion"""
        h = image.shape[0]
        lower_region = image[int(h*0.7):, :]
        
        # Look for curved interface in lower regions
        edges = cv2.Canny(lower_region, 30, 100)
        
        # Apply horizontal line detection
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)
        
        if lines is not None:
            # Count curved/horizontal lines
            horizontal_lines = 0
            for line in lines:
                rho, theta = line[0]
                if abs(theta - np.pi/2) < 0.3:  # Nearly horizontal
                    horizontal_lines += 1
            
            return min(horizontal_lines / 10.0, 1.0)  # Normalize
        
        return 0.0
    
    def _detect_focal_opacity(self, image: np.ndarray, masks: Dict[str, np.ndarray]) -> float:
        """Detect focal opacity for pneumonia"""
        lung_mask = masks.get('left_lung', np.zeros_like(image)) + masks.get('right_lung', np.zeros_like(image))
        
        if np.sum(lung_mask) == 0:
            return 0.0
        
        lung_image = image * lung_mask
        
        # Find regions significantly darker than lung average
        lung_pixels = lung_image[lung_mask > 0]
        lung_mean = np.mean(lung_pixels)
        lung_std = np.std(lung_pixels)
        
        # Threshold for opacity
        opacity_threshold = lung_mean - 1.5 * lung_std
        opaque_regions = (lung_image < opacity_threshold) & (lung_mask > 0)
        
        return np.sum(opaque_regions) / np.sum(lung_mask)
    
    def _detect_air_bronchogram(self, image: np.ndarray, masks: Dict[str, np.ndarray]) -> float:
        """Detect air bronchogram sign"""
        lung_mask = masks.get('left_lung', np.zeros_like(image)) + masks.get('right_lung', np.zeros_like(image))
        
        # Look for bright tubular structures within opaque regions
        # This is a simplified implementation
        enhanced = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        air_bronchogram_regions = enhanced * lung_mask
        
        return np.sum(air_bronchogram_regions > np.percentile(air_bronchogram_regions[lung_mask > 0], 90)) / np.sum(lung_mask) if np.sum(lung_mask) > 0 else 0.0
    
    def _calculate_regional_heterogeneity(self, image: np.ndarray, masks: Dict[str, np.ndarray]) -> float:
        """Calculate regional heterogeneity"""
        lung_mask = masks.get('left_lung', np.zeros_like(image)) + masks.get('right_lung', np.zeros_like(image))
        
        if np.sum(lung_mask) == 0:
            return 0.0
        
        # Divide lung into regions and calculate variance
        h, w = image.shape
        grid_size = 8
        heterogeneity_scores = []
        
        for i in range(grid_size):
            for j in range(grid_size):
                y_start = int(i * h / grid_size)
                y_end = int((i + 1) * h / grid_size)
                x_start = int(j * w / grid_size)
                x_end = int((j + 1) * w / grid_size)
                
                region_mask = lung_mask[y_start:y_end, x_start:x_end]
                region_image = image[y_start:y_end, x_start:x_end]
                
                if np.sum(region_mask) > 0:
                    region_pixels = region_image[region_mask > 0]
                    heterogeneity_scores.append(np.std(region_pixels))
        
        return np.mean(heterogeneity_scores) if heterogeneity_scores else 0.0
    
    def _calculate_cardiothoracic_ratio(self, image: np.ndarray) -> float:
        """Calculate cardiothoracic ratio (simplified)"""
        h, w = image.shape
        
        # Find cardiac silhouette (simplified - use intensity-based segmentation)
        middle_region = image[:, int(w*0.2):int(w*0.8)]
        
        # Threshold to find heart region (typically darker)
        heart_threshold = np.percentile(middle_region, 30)
        heart_mask = middle_region < heart_threshold
        
        # Find widest part of heart
        heart_widths = np.sum(heart_mask, axis=1)
        max_heart_width = np.max(heart_widths) if len(heart_widths) > 0 else 0
        
        # Chest width (approximate)
        chest_width = w * 0.8  # Assuming 80% of image width is chest
        
        ctr = max_heart_width / chest_width if chest_width > 0 else 0
        return min(ctr, 1.0)  # Cap at 1.0
    
    def _calculate_cardiac_area_ratio(self, image: np.ndarray) -> float:
        """Calculate cardiac area ratio"""
        h, w = image.shape
        total_area = h * w
        
        # Cardiac region (simplified)
        cardiac_region = image[int(h*0.3):int(h*0.8), int(w*0.3):int(w*0.7)]
        
        # Threshold for cardiac silhouette
        threshold = np.percentile(cardiac_region, 25)
        cardiac_mask = cardiac_region < threshold
        
        cardiac_area = np.sum(cardiac_mask)
        return cardiac_area / total_area
    
    def _detect_volume_loss(self, image: np.ndarray, masks: Dict[str, np.ndarray]) -> float:
        """Detect volume loss for atelectasis"""
        left_lung_area = np.sum(masks.get('left_lung', np.zeros((1,1))))
        right_lung_area = np.sum(masks.get('right_lung', np.zeros((1,1))))
        
        # Compare left and right lung areas
        if left_lung_area > 0 and right_lung_area > 0:
            area_ratio = min(left_lung_area, right_lung_area) / max(left_lung_area, right_lung_area)
            return 1.0 - area_ratio  # Volume loss score
        
        return 0.0
    
    def _detect_density_increase(self, image: np.ndarray, masks: Dict[str, np.ndarray]) -> float:
        """Detect density increase in lung regions"""
        left_lung = masks.get('left_lung', np.zeros_like(image))
        right_lung = masks.get('right_lung', np.zeros_like(image))
        
        if np.sum(left_lung) > 0 and np.sum(right_lung) > 0:
            left_density = np.mean(image[left_lung > 0])
            right_density = np.mean(image[right_lung > 0])
            
            # Higher density difference suggests atelectasis
            density_diff = abs(left_density - right_density) / (max(left_density, right_density) + 1e-6)
            return density_diff
        
        return 0.0
    
    def _detect_mediastinal_shift(self, image: np.ndarray) -> float:
        """Detect mediastinal shift (simplified)"""
        h, w = image.shape
        
        # Find mediastinal structures (simplified)
        central_region = image[:, int(w*0.4):int(w*0.6)]
        
        # Calculate center of mass of dark structures (mediastinum)
        threshold = np.percentile(central_region, 20)
        mediastinal_mask = central_region < threshold
        
        if np.sum(mediastinal_mask) > 0:
            moments = cv2.moments(mediastinal_mask.astype(np.uint8))
            if moments['m00'] != 0:
                centroid_x = moments['m10'] / moments['m00']
                expected_center = central_region.shape[1] / 2
                shift = abs(centroid_x - expected_center) / expected_center
                return shift
        
        return 0.0
    
    def _detect_bilateral_haze(self, image: np.ndarray, masks: Dict[str, np.ndarray]) -> float:
        """Detect bilateral hazy opacities"""
        left_lung = masks.get('left_lung', np.zeros_like(image))
        right_lung = masks.get('right_lung', np.zeros_like(image))
        
        # Calculate haziness in both lungs
        left_haze = self._calculate_haziness(image, left_lung)
        right_haze = self._calculate_haziness(image, right_lung)
        
        # Both lungs should be hazy for pulmonary edema
        return min(left_haze, right_haze)
    
    def _calculate_haziness(self, image: np.ndarray, mask: np.ndarray) -> float:
        """Calculate haziness in a region"""
        if np.sum(mask) == 0:
            return 0.0
        
        masked_image = image * mask
        
        # Calculate local variance to detect hazy appearance
        kernel = np.ones((5, 5)) / 25
        local_mean = cv2.filter2D(masked_image.astype(np.float32), -1, kernel)
        local_variance = cv2.filter2D((masked_image.astype(np.float32) - local_mean) ** 2, -1, kernel)
        
        # Low variance suggests haziness
        avg_variance = np.mean(local_variance[mask > 0])
        max_possible_variance = np.var(masked_image[mask > 0])
        
        if max_possible_variance > 0:
            haziness = 1.0 - (avg_variance / max_possible_variance)
            return haziness
        
        return 0.0
    
    def _detect_bat_wing_pattern(self, image: np.ndarray, masks: Dict[str, np.ndarray]) -> float:
        """Detect bat wing pattern for pulmonary edema"""
        # Simplified detection of central opacity with peripheral sparing
        h, w = image.shape
        
        # Central region
        central_opacity = np.mean(image[int(h*0.3):int(h*0.7), int(w*0.3):int(w*0.7)])
        
        # Peripheral regions
        peripheral_regions = [
            image[:int(h*0.3), :],  # Upper
            image[int(h*0.7):, :],  # Lower
            image[:, :int(w*0.3)],  # Left
            image[:, int(w*0.7):]   # Right
        ]
        
        peripheral_opacity = np.mean([np.mean(region) for region in peripheral_regions])
        
        # Bat wing pattern: central > peripheral
        if peripheral_opacity > 0:
            ratio = central_opacity / peripheral_opacity
            return max(0, (ratio - 1.0) / 2.0)  # Normalize
        
        return 0.0
    
    def _detect_kerley_lines(self, image: np.ndarray) -> float:
        """Detect Kerley lines (simplified)"""
        # Look for horizontal linear opacities in lower lungs
        h = image.shape[0]
        lower_region = image[int(h*0.6):, :]
        
        # Edge detection
        edges = cv2.Canny(lower_region, 30, 100)
        
        # Detect horizontal lines
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=30)
        
        if lines is not None:
            horizontal_lines = 0
            for line in lines:
                rho, theta = line[0]
                if abs(theta - np.pi/2) < 0.2:  # Nearly horizontal
                    horizontal_lines += 1
            
            return min(horizontal_lines / 20.0, 1.0)
        
        return 0.0
    
    def _detect_homogeneous_opacity(self, image: np.ndarray, masks: Dict[str, np.ndarray]) -> float:
        """Detect homogeneous opacity for consolidation"""
        lung_mask = masks.get('left_lung', np.zeros_like(image)) + masks.get('right_lung', np.zeros_like(image))
        
        if np.sum(lung_mask) == 0:
            return 0.0
        
        # Find opaque regions
        lung_pixels = image[lung_mask > 0]
        opacity_threshold = np.percentile(lung_pixels, 25)
        opaque_regions = (image < opacity_threshold) & (lung_mask > 0)
        
        # Check homogeneity of opaque regions
        if np.sum(opaque_regions) > 0:
            opaque_pixels = image[opaque_regions]
            homogeneity = 1.0 - (np.std(opaque_pixels) / (np.mean(opaque_pixels) + 1e-6))
            return max(0, homogeneity)
        
        return 0.0
    
    def _detect_silhouette_sign(self, image: np.ndarray) -> float:
        """Detect silhouette sign"""
        # Simplified: look for loss of cardiac/diaphragmatic borders
        h, w = image.shape
        
        # Cardiac border region
        cardiac_border = image[int(h*0.4):int(h*0.7), int(w*0.3):int(w*0.5)]
        
        # Calculate edge strength at cardiac border
        edges = cv2.Canny(cardiac_border, 50, 150)
        edge_density = np.sum(edges) / edges.size
        
        # Low edge density suggests silhouette sign
        return max(0, 1.0 - edge_density * 10)  # Normalize
    
    def _calculate_confidence_scores(self, classification_results: Dict[str, float], rule_based_results: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate combined confidence scores"""
        combined_scores = {}
        
        for pathology in rule_based_results.keys():
            # Get classification confidence
            class_confidence = classification_results.get(pathology.capitalize(), 0.0)
            
            # Get rule-based confidence
            rule_confidence = rule_based_results[pathology].get('confidence', 0.0)
            
            # Combine using weighted average
            combined_confidence = 0.6 * class_confidence + 0.4 * rule_confidence
            combined_scores[pathology] = combined_confidence
        
        return combined_scores
    
    def _generate_final_diagnosis(self, classification_results: Dict[str, float], rule_based_results: Dict[str, Dict]) -> Dict[str, str]:
        """Generate final diagnosis with reasoning"""
        diagnosis = {}
        
        combined_scores = self._calculate_confidence_scores(classification_results, rule_based_results)
        
        for pathology, confidence in combined_scores.items():
            if confidence > 0.7:
                diagnosis[pathology] = "High probability - further clinical correlation recommended"
            elif confidence > 0.5:
                diagnosis[pathology] = "Moderate probability - consider clinical context"
            elif confidence > 0.3:
                diagnosis[pathology] = "Low probability - findings inconclusive"
            else:
                diagnosis[pathology] = "Not detected"
        
        return diagnosis
    
    def save_results(self, results: Dict, output_path: Union[str, Path]) -> None:
        """Save pathology detection results"""
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved pathology detection results to {output_path}")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise
