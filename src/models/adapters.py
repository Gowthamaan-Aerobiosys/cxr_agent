import torch
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import numpy as np
from PIL import Image
import time

logger = logging.getLogger(__name__)


class BaseModelAdapter(ABC):
    """
    Abstract base class for all model adapters

    All model adapters must implement:
    - load(): Load the model
    - predict() or appropriate inference method
    - cleanup(): Clean up resources
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.model = None
        self.is_loaded = False
        self.load_time = None

    @abstractmethod
    async def load(self):
        """Load the model asynchronously"""
        pass

    @abstractmethod
    async def predict(self, *args, **kwargs):
        """Run inference"""
        pass

    async def cleanup(self):
        """Clean up model resources"""
        if self.model is not None:
            del self.model
            self.model = None
        self.is_loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class BinaryClassifierAdapter(BaseModelAdapter):
    """
    Adapter for binary classification (Normal vs Abnormal)
    """

    def __init__(
        self,
        checkpoint_path: str,
        model_type: str = "swin_transformer",
        device: Optional[torch.device] = None,
    ):
        super().__init__(
            device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type
        self.labels = ["Normal", "Abnormal"]

    async def load(self):
        """Load the binary classification model"""
        start_time = time.time()

        try:
            checkpoint_path = Path(self.checkpoint_path)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

            # Load checkpoint
            checkpoint = torch.load(
                str(checkpoint_path), map_location=self.device, weights_only=False
            )

            # Extract model (handle different checkpoint formats)
            if isinstance(checkpoint, dict):
                if "model" in checkpoint:
                    self.model = checkpoint["model"]
                elif "model_state_dict" in checkpoint:
                    # Need to instantiate model first, then load state dict
                    self.model = self._create_model_architecture()
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    raise ValueError("Checkpoint format not recognized")
            else:
                self.model = checkpoint

            self.model = self.model.to(self.device)
            self.model.eval()

            self.is_loaded = True
            self.load_time = time.time() - start_time

            logger.info(f"Binary classifier loaded in {self.load_time:.2f}s")

        except Exception as e:
            logger.error(f"Failed to load binary classifier: {e}")
            raise

    def _create_model_architecture(self):
        """Create model architecture based on model_type"""
        if self.model_type == "swin_transformer":
            import timm

            # Use timm Swin Large model for binary classification
            model = timm.create_model(
                "swin_large_patch4_window7_224.ms_in22k_ft_in1k",
                pretrained=True,
                num_classes=1,
            )
            return model
        elif self.model_type == "resnet":
            from torchvision.models import resnet50

            model = resnet50(pretrained=False)
            model.fc = torch.nn.Linear(model.fc.in_features, 2)
            return model
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    async def predict(self, image_path: str, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Predict if CXR is normal or abnormal

        Args:
            image_path: Path to the CXR image
            threshold: Classification threshold

        Returns:
            Dictionary with predictions and probabilities
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        try:
            # Preprocess image
            image_tensor = await self._preprocess_image(image_path)
            image_tensor = image_tensor.to(self.device)

            # Run inference
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model(image_tensor)
                # Use sigmoid for binary classification from single logit
                prob = torch.sigmoid(outputs).cpu().numpy()[0][0]

            inference_time = time.time() - start_time
            if prob > threshold:
                predicted_class = "Abnormal"
                confidence = prob
            else:
                predicted_class = "Normal"
                confidence = 1 - prob

            return {
                "prediction": predicted_class,
                "confidence": float(confidence),
                "probabilities": {"Normal": float(1 - prob), "Abnormal": float(prob)},
                "threshold": threshold,
                "inference_time_ms": inference_time * 1000,
                "image_path": image_path,
            }

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise

    async def _preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess image for model input"""
        from torchvision import transforms

        # Standard ImageNet normalization
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)
        return image_tensor


class MultiClassClassifierAdapter(BaseModelAdapter):
    """
    Adapter for 14-class disease classification using Swin Transformer
    Detects multiple pathologies: Atelectasis, Cardiomegaly, etc.
    """

    def __init__(
        self,
        checkpoint_path: str,
        num_classes: int = 14,
        model_type: str = "swin_transformer",
        device: Optional[torch.device] = None,
    ):
        super().__init__(
            device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.checkpoint_path = checkpoint_path
        self.num_classes = num_classes
        self.model_type = model_type

        # NIH ChestX-ray14 labels
        self.labels = [
            "Atelectasis",
            "Cardiomegaly",
            "Effusion",
            "Infiltration",
            "Mass",
            "Nodule",
            "Pneumonia",
            "Pneumothorax",
            "Consolidation",
            "Edema",
            "Emphysema",
            "Fibrosis",
            "Pleural_Thickening",
            "Hernia",
        ]

    async def load(self):
        """Load the multi-class classification model"""
        start_time = time.time()

        try:
            checkpoint_path = Path(self.checkpoint_path)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

            checkpoint = torch.load(
                str(checkpoint_path), map_location=self.device, weights_only=False
            )

            if isinstance(checkpoint, dict):
                if "model" in checkpoint:
                    self.model = checkpoint["model"]
                elif "state_dict" in checkpoint:
                    self.model = self._create_model_architecture()
                    self.model.load_state_dict(checkpoint["state_dict"])
            else:
                self.model = checkpoint

            self.model = self.model.to(self.device)
            self.model.eval()

            self.is_loaded = True
            self.load_time = time.time() - start_time

            logger.info(f"Multi-class classifier loaded in {self.load_time:.2f}s")

        except Exception as e:
            logger.error(f"Failed to load multi-class classifier: {e}")
            raise

    def _create_model_architecture(self):
        """Create Swin Transformer architecture for multi-label CXR classification"""
        import timm

        if self.model_type == "swin_transformer":
            # Use timm Swin Large model for multi-class classification
            model = timm.create_model(
                "swin_large_patch4_window7_224.ms_in22k_ft_in1k",
                pretrained=True,
                num_classes=self.num_classes,
            )
            return model
        else:
            # Fallback to DenseNet if specified
            from torchvision.models import densenet121
            import torch.nn as nn

            model = densenet121(pretrained=False)
            num_features = model.classifier.in_features

            model.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, self.num_classes),
                nn.Sigmoid(),
            )
            return model

    async def predict(
        self, image_path: str, threshold: float = 0.3, top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Predict multiple diseases from CXR

        Args:
            image_path: Path to CXR image
            threshold: Minimum confidence threshold
            top_k: Return top K predictions

        Returns:
            Dictionary with disease predictions
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        try:
            # Preprocess
            image_tensor = await self._preprocess_image(image_path)
            image_tensor = image_tensor.to(self.device)

            # Inference
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model(image_tensor)
                # Apply softmax for multi-class classification
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

            inference_time = time.time() - start_time

            # Create predictions dictionary
            predictions = {
                label: float(prob) for label, prob in zip(self.labels, probs)
            }

            # Filter by threshold
            detected_diseases = {
                label: prob for label, prob in predictions.items() if prob >= threshold
            }

            # Sort by confidence
            sorted_predictions = sorted(
                predictions.items(), key=lambda x: x[1], reverse=True
            )[:top_k]

            return {
                "all_predictions": predictions,
                "detected_diseases": detected_diseases,
                "top_predictions": dict(sorted_predictions),
                "num_detected": len(detected_diseases),
                "threshold": threshold,
                "inference_time_ms": inference_time * 1000,
                "image_path": image_path,
            }

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise

    async def _preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess image for model input"""
        from torchvision import transforms

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        image = Image.open(image_path).convert("RGB")
        return transform(image).unsqueeze(0)


class SegmentationAdapter(BaseModelAdapter):
    """
    Adapter for chest X-ray segmentation using torchxrayvision PSPNet
    Performs semantic segmentation of 14 anatomical structures
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model_type: str = "pspnet",
        device: Optional[torch.device] = None,
    ):
        super().__init__(
            device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.checkpoint_path = checkpoint_path
        self.model_type = model_type
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

    async def load(self):
        """Load torchxrayvision PSPNet segmentation model"""
        start_time = time.time()

        try:
            import torchxrayvision as xrv

            self.model = xrv.baseline_models.chestx_det.PSPNet()
            self.model = self.model.to(self.device)
            self.model.eval()

            self.is_loaded = True
            self.load_time = time.time() - start_time

            logger.info(f"Segmentation model loaded in {self.load_time:.2f}s")

        except Exception as e:
            logger.error(f"Failed to load segmentation model: {e}")
            raise

    async def predict(
        self,
        image_path: str,
        save_mask: bool = False,
        output_path: Optional[str] = None,
        target_structures: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Segment anatomical structures in CXR

        Args:
            image_path: Path to CXR image
            save_mask: Whether to save segmentation masks
            output_path: Path to save mask (directory if multiple, file if single)
            target_structures: List of specific structures to segment (None = all)

        Returns:
            Segmentation results with metrics for each structure
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        try:
            start_time = time.time()

            # Preprocess image for torchxrayvision
            image_tensor = await self._preprocess_image(image_path)
            image_tensor = image_tensor.to(self.device)

            # Run segmentation
            with torch.no_grad():
                output = self.model(image_tensor)

            # output shape: [1, 14, 512, 512]
            masks = output.cpu().numpy()[0]  # Shape: [14, 512, 512]

            inference_time = time.time() - start_time

            # Calculate metrics for each structure
            structure_metrics = {}
            for idx, target_name in enumerate(self.targets):
                if target_structures is None or target_name in target_structures:
                    mask = masks[idx]
                    structure_metrics[target_name] = {
                        "area_pixels": float(np.sum(mask > 0.5)),
                        "mean_confidence": float(np.mean(mask)),
                        "max_confidence": float(np.max(mask)),
                    }

            # Calculate lung-specific metrics
            left_lung_idx = self.targets.index("Left Lung")
            right_lung_idx = self.targets.index("Right Lung")
            left_lung_mask = masks[left_lung_idx] > 0.5
            right_lung_mask = masks[right_lung_idx] > 0.5

            total_lung_area = float(np.sum(left_lung_mask) + np.sum(right_lung_mask))
            total_image_area = masks.shape[1] * masks.shape[2]

            # Save masks if requested
            saved_paths = []
            if save_mask and output_path:
                output_dir = Path(output_path)
                if target_structures:
                    # Save only requested structures
                    for target in target_structures:
                        if target in self.targets:
                            idx = self.targets.index(target)
                            mask_to_save = (masks[idx] * 255).astype(np.uint8)
                            save_path = output_dir / f"{target.replace(' ', '_')}.png"
                            output_dir.mkdir(parents=True, exist_ok=True)
                            Image.fromarray(mask_to_save).save(save_path)
                            saved_paths.append(str(save_path))
                else:
                    # Save all structures
                    output_dir.mkdir(parents=True, exist_ok=True)
                    for idx, target in enumerate(self.targets):
                        mask_to_save = (masks[idx] * 255).astype(np.uint8)
                        save_path = output_dir / f"{target.replace(' ', '_')}.png"
                        Image.fromarray(mask_to_save).save(save_path)
                        saved_paths.append(str(save_path))

            return {
                "mask_shape": masks.shape,
                "structure_metrics": structure_metrics,
                "lung_area_pixels": total_lung_area,
                "lung_ratio": total_lung_area / total_image_area,
                "left_lung_area": float(np.sum(left_lung_mask)),
                "right_lung_area": float(np.sum(right_lung_mask)),
                "available_structures": self.targets,
                "inference_time_ms": inference_time * 1000,
                "mask_saved": save_mask and len(saved_paths) > 0,
                "saved_paths": saved_paths if save_mask else [],
                "image_path": image_path,
            }

        except Exception as e:
            logger.error(f"Segmentation error: {e}")
            raise

    async def _preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess image for torchxrayvision model input"""
        import torchxrayvision as xrv
        from torchvision import transforms

        # Load image
        image = Image.open(image_path).convert("L")  # Convert to grayscale
        image = np.array(image)

        # Normalize to [-1024, 1024] range (standard for CXR)
        image = image.astype(np.float32)
        image = (image / 255.0) * 2048 - 1024

        # Add channel dimension and convert to tensor
        image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

        # Resize to 512x512 if needed
        if image_tensor.shape[2] != 512 or image_tensor.shape[3] != 512:
            resize = transforms.Resize((512, 512))
            image_tensor = resize(image_tensor)

        return image_tensor


class RAGAdapter(BaseModelAdapter):
    """
    Adapter for RAG (Retrieval-Augmented Generation)
    Handles medical knowledge queries
    """

    def __init__(
        self,
        model_name: str,
        vector_db_path: str,
        documents_path: str,
        device: Optional[torch.device] = None,
    ):
        super().__init__(
            device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model_name = model_name
        self.vector_db_path = vector_db_path
        self.documents_path = documents_path
        self.agent = None

    async def load(self):
        """Load RAG agent"""
        start_time = time.time()

        try:
            from src.rag.llm_engine import AgenticRAG
            from src.rag.document_processor import VectorStore

            # Initialize vector store
            self.vector_store = VectorStore(persist_directory=self.vector_db_path)

            # Initialize agent
            self.agent = AgenticRAG(
                model_name=self.model_name, vector_store=self.vector_store
            )

            self.is_loaded = True
            self.load_time = time.time() - start_time

            logger.info(f"RAG agent loaded in {self.load_time:.2f}s")

        except Exception as e:
            logger.error(f"Failed to load RAG agent: {e}")
            raise

    async def predict(self, *args, **kwargs):
        """Alias for query method"""
        return await self.query(*args, **kwargs)

    async def query(
        self, query: str, top_k: int = 5, include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Query medical knowledge base

        Args:
            query: Medical question
            top_k: Number of documents to retrieve
            include_sources: Include source references

        Returns:
            Response with answer and sources
        """
        if not self.is_loaded:
            raise RuntimeError("RAG agent not loaded. Call load() first.")

        try:
            start_time = time.time()

            # Run RAG query
            response = await self.agent.answer_question(query=query, top_k=top_k)

            query_time = time.time() - start_time

            result = {
                "query": query,
                "answer": response.get("answer", ""),
                "query_time_ms": query_time * 1000,
            }

            if include_sources:
                result["sources"] = response.get("sources", [])
                result["retrieved_docs"] = response.get("num_docs", 0)

            return result

        except Exception as e:
            logger.error(f"RAG query error: {e}")
            raise


class ReportGeneratorAdapter(BaseModelAdapter):
    """
    Adapter for radiology report generation
    Generates structured reports from CXR findings
    """

    def __init__(
        self,
        model_name: str,
        checkpoint_path: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(
            device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path

    async def load(self):
        """Load report generation model"""
        start_time = time.time()

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=(
                    torch.float16 if self.device.type == "cuda" else torch.float32
                ),
                device_map="auto" if self.device.type == "cuda" else None,
            )

            if self.device.type == "cpu":
                self.model = self.model.to(self.device)

            self.model.eval()

            self.is_loaded = True
            self.load_time = time.time() - start_time

            logger.info(f"Report generator loaded in {self.load_time:.2f}s")

        except Exception as e:
            logger.error(f"Failed to load report generator: {e}")
            raise

    async def predict(self, *args, **kwargs):
        """Alias for generate method"""
        return await self.generate(*args, **kwargs)

    async def generate(
        self,
        image_path: str,
        findings: Optional[Dict] = None,
        clinical_context: str = "",
        style: str = "structured",
    ) -> Dict[str, Any]:
        """
        Generate radiology report

        Args:
            image_path: Path to CXR image
            findings: Structured findings from classification
            clinical_context: Patient information
            style: Report style (structured/narrative/brief)

        Returns:
            Generated report
        """
        if not self.is_loaded:
            raise RuntimeError("Report generator not loaded. Call load() first.")

        try:
            start_time = time.time()

            # Construct prompt
            prompt = self._build_prompt(findings, clinical_context, style)

            # Generate report
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                )

            report = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            generation_time = time.time() - start_time

            return {
                "report": report,
                "style": style,
                "findings_included": findings is not None,
                "clinical_context_included": bool(clinical_context),
                "generation_time_ms": generation_time * 1000,
                "image_path": image_path,
            }

        except Exception as e:
            logger.error(f"Report generation error: {e}")
            raise

    def _build_prompt(
        self, findings: Optional[Dict], clinical_context: str, style: str
    ) -> str:
        """Build prompt for report generation"""
        prompt = "Generate a chest X-ray radiology report.\n\n"

        if clinical_context:
            prompt += f"Clinical Context:\n{clinical_context}\n\n"

        if findings:
            prompt += "Findings:\n"
            if "binary_classification" in findings:
                bc = findings["binary_classification"]
                prompt += f"- Overall: {bc.get('prediction', 'Unknown')}\n"

            if "disease_classification" in findings:
                dc = findings["disease_classification"]
                detected = dc.get("detected_diseases", {})
                if detected:
                    prompt += "- Detected pathologies:\n"
                    for disease, prob in detected.items():
                        prompt += f"  * {disease}: {prob:.1%}\n"

            prompt += "\n"

        prompt += f"Report Style: {style}\n\nReport:\n"

        return prompt


class FeatureExtractorAdapter(BaseModelAdapter):
    """
    Adapter for medical image feature extraction using MedSigLIP
    Extracts multimodal embeddings from CXR images and text descriptions
    """

    def __init__(
        self,
        model_name: str = "google/medsiglip-448",
        hf_token: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(
            device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model_name = model_name
        self.hf_token = hf_token or self._get_hf_token()
        self.processor = None

    def _get_hf_token(self) -> Optional[str]:
        """Get Hugging Face token from environment variables or .env file"""
        import os
        from pathlib import Path

        # Try loading from .env file first
        try:
            from dotenv import load_dotenv

            # Look for .env in project root
            env_path = Path(__file__).parent.parent.parent / ".env"
            if env_path.exists():
                load_dotenv(env_path)
                logger.debug(f"Loaded .env file from {env_path}")
        except ImportError:
            logger.debug(
                "python-dotenv not installed, relying on existing environment variables"
            )
        except Exception as e:
            logger.debug(f"Could not load .env file: {e}")

        # Try multiple common environment variable names
        token = (
            os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACE_TOKEN")
            or os.getenv("HF_ACCESS_TOKEN")
        )

        if token:
            logger.info(
                f"Hugging Face token loaded successfully (length: {len(token)})"
            )
        else:
            logger.warning(
                "No Hugging Face token found. Set HF_TOKEN in .env file or environment variables."
            )

        return token

    async def load(self):
        """Load MedSigLIP model and processor"""
        start_time = time.time()

        try:
            from transformers import AutoProcessor, AutoModel

            # Load processor and model
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                token=self.hf_token,
            )
            self.model = AutoModel.from_pretrained(
                self.model_name,
                token=self.hf_token,
            )

            self.model = self.model.to(self.device)
            self.model.eval()

            self.is_loaded = True
            self.load_time = time.time() - start_time

            logger.info(f"MedSigLIP feature extractor loaded in {self.load_time:.2f}s")

        except Exception as e:
            logger.error(f"Failed to load MedSigLIP model: {e}")
            raise

    async def predict(
        self,
        image_path: Union[str, List[str]],
        texts: Optional[List[str]] = None,
        return_embeddings: bool = True,
        return_similarities: bool = False,
    ) -> Dict[str, Any]:
        """
        Extract features from CXR image(s) and optionally compute text similarities

        Args:
            image_path: Path to single image or list of image paths
            texts: Optional list of text descriptions for similarity computation
            return_embeddings: Whether to return image/text embeddings
            return_similarities: Whether to compute and return image-text similarities

        Returns:
            Dictionary with embeddings and/or similarity scores
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        try:
            start_time = time.time()

            # Handle single or multiple images
            image_paths = [image_path] if isinstance(image_path, str) else image_path

            # Load and resize images
            images = []
            for img_path in image_paths:
                img = Image.open(img_path).convert("RGB")
                resized_img = self._resize_image(img)
                images.append(resized_img)

            # Prepare inputs
            if texts:
                inputs = self.processor(
                    text=texts, images=images, padding="max_length", return_tensors="pt"
                ).to(self.device)
            else:
                inputs = self.processor(images=images, return_tensors="pt").to(
                    self.device
                )

            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)

            extraction_time = time.time() - start_time

            result = {
                "num_images": len(images),
                "extraction_time_ms": extraction_time * 1000,
                "image_paths": image_paths,
            }

            # Add embeddings if requested
            if return_embeddings:
                result["image_embeddings"] = outputs.image_embeds.cpu().numpy().tolist()
                result["image_embedding_shape"] = list(outputs.image_embeds.shape)

                if texts and hasattr(outputs, "text_embeds"):
                    result["text_embeddings"] = (
                        outputs.text_embeds.cpu().numpy().tolist()
                    )
                    result["text_embedding_shape"] = list(outputs.text_embeds.shape)
                    result["num_texts"] = len(texts)

            # Add similarity scores if requested
            if return_similarities and texts and hasattr(outputs, "logits_per_image"):
                logits_per_image = outputs.logits_per_image
                probs = torch.softmax(logits_per_image, dim=1).cpu().numpy()

                # Create detailed similarity results
                similarities = []
                for img_idx, img_path in enumerate(image_paths):
                    img_similarities = {}
                    for text_idx, text in enumerate(texts):
                        img_similarities[text] = float(probs[img_idx][text_idx])
                    similarities.append(
                        {
                            "image_path": img_path,
                            "similarities": img_similarities,
                            "top_match": max(
                                img_similarities.items(), key=lambda x: x[1]
                            ),
                        }
                    )

                result["similarities"] = similarities
                result["texts"] = texts

            return result

        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            raise

    def _resize_image(self, image: Image.Image) -> Image.Image:
        """
        Resize image to 448x448 using Pillow's bilinear interpolation
        This provides similar results to the Big Vision library implementation
        """
        return image.resize((448, 448), Image.Resampling.BILINEAR)

    async def extract_embeddings_batch(
        self, image_paths: List[str], batch_size: int = 8
    ) -> Dict[str, Any]:
        """
        Extract embeddings from a batch of images efficiently

        Args:
            image_paths: List of image paths
            batch_size: Number of images to process at once

        Returns:
            Dictionary with all embeddings
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        all_embeddings = []
        start_time = time.time()

        try:
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i : i + batch_size]

                # Load and resize images
                images = []
                for img_path in batch_paths:
                    img = Image.open(img_path).convert("RGB")
                    resized_img = self._resize_image(img)
                    images.append(resized_img)

                # Process batch
                inputs = self.processor(images=images, return_tensors="pt").to(
                    self.device
                )

                with torch.no_grad():
                    outputs = self.model(**inputs)

                batch_embeddings = outputs.image_embeds.cpu().numpy()
                all_embeddings.append(batch_embeddings)

            # Concatenate all batches
            all_embeddings = np.concatenate(all_embeddings, axis=0)

            extraction_time = time.time() - start_time

            return {
                "embeddings": all_embeddings.tolist(),
                "embedding_shape": list(all_embeddings.shape),
                "num_images": len(image_paths),
                "batch_size": batch_size,
                "extraction_time_ms": extraction_time * 1000,
                "image_paths": image_paths,
            }

        except Exception as e:
            logger.error(f"Batch extraction error: {e}")
            raise
