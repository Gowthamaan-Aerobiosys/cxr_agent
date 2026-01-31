"""
Lung Segmenter using torchxrayvision PSPNet
Segments lung regions and anatomical structures
"""

import numpy as np
import cv2
import torch
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class LungSegmenter:
    """Lung segmentation using torchxrayvision PSPNet"""

    def __init__(self, device: Optional[str] = None):
        self.model = None
        self.is_loaded = False
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.targets = [
            "Left Clavicle",
            "Right Clavicle",
            "Left Scapula",
            "Right Scapula",
            "Left Lung",
            "Right Lung",
            "Left Hilus Pulmonis",
            "Right Hilus Pulmonis",
            "Heart",
            "Aorta",
            "Facies Diaphragmatica",
            "Mediastinum",
            "Weasand",
            "Spine",
        ]

    def _load_model(self):
        """Lazy load PSPNet model"""
        if not self.is_loaded:
            try:
                import torchxrayvision as xrv

                self.model = xrv.baseline_models.chestx_det.PSPNet()
                self.model = self.model.to(self.device)
                self.model.eval()
                self.is_loaded = True
                logger.info("PSPNet segmentation model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load PSPNet model: {e}")
                raise

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for PSPNet following torchxrayvision guidelines
        Reference: https://github.com/mlmed/torchxrayvision/blob/main/scripts/segmentation.ipynb
        """
        import torchxrayvision as xrv
        import torchvision

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = image.mean(2)  # Average channels
        else:
            gray = image

        # Normalize to [-1024, 1024] range (torchxrayvision standard)
        # Assumes input is 8-bit (0-255) or similar range
        maxval = 255 if image.max() <= 255 else image.max()
        img_normalized = xrv.datasets.normalize(gray, maxval)

        # Add channel dimension [H, W] -> [1, H, W]
        img_normalized = img_normalized[None, ...]

        # Apply transforms: center crop and resize to 512x512
        transform = torchvision.transforms.Compose(
            [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(512)]
        )

        img_transformed = transform(img_normalized)

        # Convert to tensor and add batch dimension [1, H, W] -> [1, 1, H, W]
        img_tensor = torch.from_numpy(img_transformed).unsqueeze(0)

        return img_tensor

    def segment_lungs(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Segment lung regions using PSPNet

        Args:
            image: Input CXR image

        Returns:
            Dictionary of segmentation masks for all 14 anatomical structures
        """
        try:
            # Load model if not already loaded
            self._load_model()

            # Store original shape for resizing masks back
            original_shape = image.shape[:2]

            # Preprocess image
            image_tensor = self._preprocess_image(image)
            image_tensor = image_tensor.to(self.device)

            # Run segmentation
            with torch.no_grad():
                output = self.model(image_tensor)

            # output shape: [1, 14, 512, 512]
            masks_tensor = output.cpu().numpy()[0]  # Shape: [14, 512, 512]

            # Convert to binary masks and resize to original dimensions
            masks = {}
            for idx, target_name in enumerate(self.targets):
                mask = masks_tensor[idx]
                # Threshold at 0.5
                binary_mask = (mask > 0.5).astype(np.uint8)
                # Resize back to original dimensions
                if original_shape != (512, 512):
                    binary_mask = cv2.resize(
                        binary_mask,
                        (original_shape[1], original_shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                masks[target_name.lower().replace(" ", "_")] = binary_mask

            # Extract specific lung masks for backward compatibility
            masks["left_lung"] = masks["left_lung"]
            masks["right_lung"] = masks["right_lung"]
            masks["both_lungs"] = masks["left_lung"] + masks["right_lung"]

            logger.debug(
                f"PSPNet segmented lungs: left={np.sum(masks['left_lung'])}, "
                f"right={np.sum(masks['right_lung'])} pixels"
            )

            return masks

        except Exception as e:
            logger.error(f"Error in PSPNet lung segmentation: {e}")
            # Return empty masks on error
            h, w = image.shape[:2]
            return {
                "left_lung": np.zeros((h, w), dtype=np.uint8),
                "right_lung": np.zeros((h, w), dtype=np.uint8),
                "both_lungs": np.zeros((h, w), dtype=np.uint8),
            }

    def get_lung_metrics(self, masks: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate lung metrics from masks"""
        metrics = {}

        left_area = float(np.sum(masks.get("left_lung", 0)))
        right_area = float(np.sum(masks.get("right_lung", 0)))
        total_area = left_area + right_area

        metrics["left_lung_area"] = left_area
        metrics["right_lung_area"] = right_area
        metrics["total_lung_area"] = total_area

        if total_area > 0:
            metrics["lung_symmetry"] = min(left_area, right_area) / max(
                left_area, right_area
            )
        else:
            metrics["lung_symmetry"] = 0.0

        return metrics
