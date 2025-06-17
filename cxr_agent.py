"""
CXR Agent - Main Integration Module
Integrates lung tools with RAG pipeline to create a comprehensive CXR analysis system
"""

import sys
import os
from pathlib import Path
import numpy as np
import logging
from typing import Dict, List, Union, Optional, Any
import json
import asyncio
from datetime import datetime

# Add paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "rag-pipeline"))

from lung_tools import (
    CXRImageProcessor, 
    CXRClassifier, 
    LungSegmenter, 
    CXRFeatureExtractor, 
    PathologyDetector
)

# Import RAG components
try:
    from rag_pipeline.qwen_agent import QwenAgent, AgenticRAG
    from rag_pipeline.document_processor import DocumentProcessor, VectorStore
    from rag_pipeline.config import DEFAULT_CONFIG
except ImportError:
    # Fallback imports if structure is different
    from qwen_agent import QwenAgent, AgenticRAG
    from document_processor import DocumentProcessor, VectorStore
    from config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)

class CXRAgent:
    """
    Comprehensive CXR Agent that combines:
    - Image analysis (classification, segmentation, pathology detection)
    - Agentic RAG for medical knowledge
    - Clinical report generation
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or DEFAULT_CONFIG
        
        # Initialize components
        self.image_processor = CXRImageProcessor()
        self.classifier = CXRClassifier()
        self.segmenter = LungSegmenter()
        self.feature_extractor = CXRFeatureExtractor()
        self.pathology_detector = PathologyDetector()
        
        # Initialize RAG system
        self.rag_system = None
        self.vector_store = None
        self.qwen_agent = None
        
        self._initialize_rag_system()
        
        logger.info("CXR Agent initialized successfully")
    
    def _initialize_rag_system(self):
        """Initialize the RAG system for medical knowledge"""
        try:
            # Initialize document processor
            processor = DocumentProcessor(
                chunk_size=self.config.document.chunk_size,
                chunk_overlap=self.config.document.chunk_overlap
            )
            
            # Initialize vector store
            self.vector_store = VectorStore(
                collection_name=self.config.vector_store.collection_name,
                embedding_model=self.config.vector_store.embedding_model
            )
            
            # Initialize QWEN agent
            self.qwen_agent = QwenAgent(
                model_name=self.config.model.model_name,
                load_in_4bit=self.config.model.load_in_4bit,
                max_new_tokens=self.config.model.max_new_tokens
            )
            
            # Create agentic RAG system
            self.rag_system = AgenticRAG(
                qwen_agent=self.qwen_agent,
                vector_store=self.vector_store,
                config=self.config
            )
            
            logger.info("RAG system initialized successfully")
        
        except Exception as e:
            logger.error(f"Error initializing RAG system: {str(e)}")
            raise
    
    async def analyze_cxr(self, image_path: Union[str, Path], 
                         include_rag: bool = True,
                         generate_report: bool = True) -> Dict[str, Any]:
        """
        Comprehensive CXR analysis
        
        Args:
            image_path: Path to CXR image
            include_rag: Whether to include RAG-based analysis
            generate_report: Whether to generate clinical report
        
        Returns:
            Dictionary containing all analysis results
        """
        try:
            logger.info(f"Starting CXR analysis for: {image_path}")
            
            # Load and preprocess image
            image = self.image_processor.load_image(image_path)
            
            # Step 1: Image Analysis
            analysis_results = {}
            
            # Classification
            logger.info("Performing classification...")
            classification_results = self.classifier.classify_image(image)
            analysis_results['classification'] = classification_results
            
            # Segmentation
            logger.info("Performing segmentation...")
            segmentation_results = self.segmenter.segment_lungs(image)
            analysis_results['segmentation'] = {
                'masks': segmentation_results,
                'features': self.segmenter.extract_lung_features(image, segmentation_results)
            }
            
            # Feature extraction
            logger.info("Extracting features...")
            features = self.feature_extractor.extract_all_features(image, segmentation_results)
            analysis_results['features'] = features
            
            # Pathology detection
            logger.info("Detecting pathologies...")
            pathology_results = self.pathology_detector.detect_all_pathologies(image)
            analysis_results['pathology_detection'] = pathology_results
            
            # Step 2: RAG-based Analysis (if requested)
            if include_rag and self.rag_system:
                logger.info("Performing RAG-based analysis...")
                rag_analysis = await self._perform_rag_analysis(analysis_results)
                analysis_results['rag_analysis'] = rag_analysis
            
            # Step 3: Generate Clinical Report (if requested)
            if generate_report:
                logger.info("Generating clinical report...")
                clinical_report = self._generate_clinical_report(analysis_results)
                analysis_results['clinical_report'] = clinical_report
            
            # Add metadata
            analysis_results['metadata'] = {
                'image_path': str(image_path),
                'timestamp': datetime.now().isoformat(),
                'image_stats': self.image_processor.get_image_stats(image),
                'version': '1.0.0'
            }
            
            logger.info("CXR analysis completed successfully")
            return analysis_results
        
        except Exception as e:
            logger.error(f"Error in CXR analysis: {str(e)}")
            raise
    
    async def _perform_rag_analysis(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform RAG-based analysis and interpretation"""
        rag_results = {}
        
        try:
            # Extract key findings for RAG queries
            findings = self._extract_key_findings(analysis_results)
            
            # Generate RAG queries based on findings
            rag_queries = self._generate_rag_queries(findings)
            
            # Query RAG system for each finding
            rag_responses = {}
            for query_type, query in rag_queries.items():
                try:
                    response = await self.rag_system.agentic_query(query, max_context_chunks=5)
                    rag_responses[query_type] = response
                except Exception as e:
                    logger.error(f"Error in RAG query for {query_type}: {str(e)}")
                    rag_responses[query_type] = {"error": str(e)}
            
            rag_results['responses'] = rag_responses
            
            # Generate interpretation
            interpretation = self._interpret_findings_with_rag(findings, rag_responses)
            rag_results['interpretation'] = interpretation
            
            # Generate recommendations
            recommendations = self._generate_recommendations(findings, rag_responses)
            rag_results['recommendations'] = recommendations
            
        except Exception as e:
            logger.error(f"Error in RAG analysis: {str(e)}")
            rag_results['error'] = str(e)
        
        return rag_results
    
    def _extract_key_findings(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key findings from analysis results"""
        findings = {}
        
        # Extract high-confidence classifications
        classifications = analysis_results.get('classification', {})
        high_confidence_findings = {k: v for k, v in classifications.items() if v > 0.5}
        findings['high_confidence_pathologies'] = high_confidence_findings
        
        # Extract detected pathologies
        pathology_detection = analysis_results.get('pathology_detection', {})
        detected_pathologies = []
        
        rule_based = pathology_detection.get('rule_based_detection', {})
        for pathology, result in rule_based.items():
            if result.get('detected', False):
                detected_pathologies.append({
                    'pathology': pathology,
                    'confidence': result.get('confidence', 0.0),
                    'indicators': result.get('indicators', {})
                })
        
        findings['detected_pathologies'] = detected_pathologies
        
        # Extract abnormal features
        features = analysis_results.get('features', {})
        abnormal_features = self._identify_abnormal_features(features)
        findings['abnormal_features'] = abnormal_features
        
        # Extract segmentation findings
        segmentation = analysis_results.get('segmentation', {})
        seg_features = segmentation.get('features', {})
        findings['lung_morphology'] = seg_features
        
        return findings
    
    def _identify_abnormal_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Identify abnormal features based on normal ranges"""
        abnormal = {}
        
        # Basic feature thresholds (these would be refined based on clinical data)
        basic_features = features.get('basic', {})
        
        # Check for unusual intensity patterns
        if basic_features.get('mean_intensity', 0) < 50:  # Very low intensity
            abnormal['low_lung_density'] = basic_features.get('mean_intensity')
        
        if basic_features.get('std_intensity', 0) > 80:  # High variability
            abnormal['high_heterogeneity'] = basic_features.get('std_intensity')
        
        # Check radiological features
        radiological = features.get('radiological', {})
        
        if radiological.get('lung_field_symmetry', 1.0) < 0.8:
            abnormal['asymmetric_lung_fields'] = radiological.get('lung_field_symmetry')
        
        if radiological.get('costophrenic_angle_clarity', 100) < 20:
            abnormal['blunted_costophrenic_angles'] = radiological.get('costophrenic_angle_clarity')
        
        return abnormal
    
    def _generate_rag_queries(self, findings: Dict[str, Any]) -> Dict[str, str]:
        """Generate RAG queries based on findings"""
        queries = {}
        
        # Query for high-confidence pathologies
        high_conf_pathologies = findings.get('high_confidence_pathologies', {})
        if high_conf_pathologies:
            pathology_list = list(high_conf_pathologies.keys())
            queries['pathology_explanation'] = f"Explain the clinical significance and management of {', '.join(pathology_list)} on chest X-ray."
        
        # Query for detected pathologies
        detected_pathologies = findings.get('detected_pathologies', [])
        if detected_pathologies:
            for pathology_info in detected_pathologies:
                pathology = pathology_info['pathology']
                queries[f'{pathology}_management'] = f"What are the treatment options and clinical management for {pathology}? Include differential diagnosis."
        
        # Query for abnormal features
        abnormal_features = findings.get('abnormal_features', {})
        if 'asymmetric_lung_fields' in abnormal_features:
            queries['asymmetry_causes'] = "What are the causes of asymmetric lung fields on chest X-ray? Include pathological conditions that cause unilateral changes."
        
        if 'blunted_costophrenic_angles' in abnormal_features:
            queries['costophrenic_blunting'] = "What causes blunting of costophrenic angles on chest X-ray? Discuss pleural effusion and other differential diagnoses."
        
        # General interpretation query
        if high_conf_pathologies or detected_pathologies:
            queries['general_interpretation'] = "Provide a comprehensive interpretation of chest X-ray findings including clinical correlations and next steps."
        
        return queries
    
    def _interpret_findings_with_rag(self, findings: Dict[str, Any], rag_responses: Dict[str, Any]) -> str:
        """Generate interpretation combining findings with RAG knowledge"""
        interpretation_parts = []
        
        # Start with detected pathologies
        detected_pathologies = findings.get('detected_pathologies', [])
        high_conf_pathologies = findings.get('high_confidence_pathologies', {})
        
        if detected_pathologies or high_conf_pathologies:
            interpretation_parts.append("DETECTED ABNORMALITIES:")
            
            # Add detected pathologies
            for pathology_info in detected_pathologies:
                pathology = pathology_info['pathology']
                confidence = pathology_info['confidence']
                
                interpretation_parts.append(f"- {pathology.replace('_', ' ').title()}: Confidence {confidence:.2f}")
                
                # Add RAG-based explanation if available
                rag_key = f'{pathology}_management'
                if rag_key in rag_responses and 'response' in rag_responses[rag_key]:
                    rag_text = rag_responses[rag_key]['response'][:200] + "..."
                    interpretation_parts.append(f"  Clinical Context: {rag_text}")
            
            # Add high-confidence classifications
            for pathology, confidence in high_conf_pathologies.items():
                interpretation_parts.append(f"- {pathology}: Classification confidence {confidence:.2f}")
        
        # Add morphological findings
        lung_morphology = findings.get('lung_morphology', {})
        if lung_morphology:
            interpretation_parts.append("\nLUNG MORPHOLOGY:")
            
            for lung, features in lung_morphology.items():
                if lung != 'background':
                    area = features.get('area', 0)
                    mean_intensity = features.get('mean_intensity', 0)
                    interpretation_parts.append(f"- {lung.replace('_', ' ').title()}: Area {area:.0f} pixels, Mean intensity {mean_intensity:.1f}")
        
        # Add abnormal features
        abnormal_features = findings.get('abnormal_features', {})
        if abnormal_features:
            interpretation_parts.append("\nABNORMAL FEATURES:")
            for feature, value in abnormal_features.items():
                interpretation_parts.append(f"- {feature.replace('_', ' ').title()}: {value:.2f}")
        
        # Add general RAG interpretation if available
        if 'general_interpretation' in rag_responses and 'response' in rag_responses['general_interpretation']:
            interpretation_parts.append("\nCLINICAL INTERPRETATION:")
            interpretation_parts.append(rag_responses['general_interpretation']['response'][:500] + "...")
        
        return "\n".join(interpretation_parts)
    
    def _generate_recommendations(self, findings: Dict[str, Any], rag_responses: Dict[str, Any]) -> List[str]:
        """Generate clinical recommendations"""
        recommendations = []
        
        # Based on detected pathologies
        detected_pathologies = findings.get('detected_pathologies', [])
        
        for pathology_info in detected_pathologies:
            pathology = pathology_info['pathology']
            confidence = pathology_info['confidence']
            
            if confidence > 0.7:
                recommendations.append(f"High suspicion for {pathology.replace('_', ' ')}: Recommend immediate clinical correlation and appropriate management.")
            elif confidence > 0.5:
                recommendations.append(f"Moderate suspicion for {pathology.replace('_', ' ')}: Consider clinical context and additional imaging if indicated.")
        
        # Based on abnormal features
        abnormal_features = findings.get('abnormal_features', {})
        
        if 'asymmetric_lung_fields' in abnormal_features:
            recommendations.append("Asymmetric lung fields detected: Consider CT chest for detailed evaluation.")
        
        if 'blunted_costophrenic_angles' in abnormal_features:
            recommendations.append("Blunted costophrenic angles: Evaluate for pleural effusion, consider lateral view or ultrasound.")
        
        # General recommendations
        if detected_pathologies:
            recommendations.append("All AI-generated findings require validation by a qualified radiologist.")
            recommendations.append("Clinical correlation with patient history and physical examination is essential.")
        
        if not detected_pathologies and not abnormal_features:
            recommendations.append("No significant abnormalities detected by AI analysis. Clinical correlation recommended.")
        
        return recommendations
    
    def _generate_clinical_report(self, analysis_results: Dict[str, Any]) -> Dict[str, str]:
        """Generate structured clinical report"""
        report = {}
        
        # Report header
        metadata = analysis_results.get('metadata', {})
        report['header'] = f"""
CHEST X-RAY AI ANALYSIS REPORT
Generated: {metadata.get('timestamp', 'Unknown')}
Image: {Path(metadata.get('image_path', 'Unknown')).name}
Analysis Version: {metadata.get('version', 'Unknown')}
"""
        
        # Clinical findings
        pathology_detection = analysis_results.get('pathology_detection', {})
        final_diagnosis = pathology_detection.get('final_diagnosis', {})
        
        findings = []
        for pathology, diagnosis in final_diagnosis.items():
            if 'High probability' in diagnosis:
                findings.append(f"- {pathology.replace('_', ' ').title()}: {diagnosis}")
            elif 'Moderate probability' in diagnosis:
                findings.append(f"- {pathology.replace('_', ' ').title()}: {diagnosis}")
        
        report['findings'] = "FINDINGS:\n" + ("\n".join(findings) if findings else "No significant abnormalities detected.")
        
        # RAG-based interpretation
        rag_analysis = analysis_results.get('rag_analysis', {})
        if 'interpretation' in rag_analysis:
            report['interpretation'] = "INTERPRETATION:\n" + rag_analysis['interpretation']
        
        # Recommendations
        if 'recommendations' in rag_analysis:
            recommendations = rag_analysis['recommendations']
            report['recommendations'] = "RECOMMENDATIONS:\n" + "\n".join([f"- {rec}" for rec in recommendations])
        
        # Technical details
        image_stats = metadata.get('image_stats', {})
        report['technical'] = f"""
TECHNICAL DETAILS:
- Image dimensions: {image_stats.get('shape', 'Unknown')}
- Image type: {image_stats.get('dtype', 'Unknown')}
- File size: {image_stats.get('size_mb', 0):.2f} MB
"""
        
        # Disclaimer
        report['disclaimer'] = """
DISCLAIMER:
This report is generated by AI and is intended for educational and research purposes only.
All findings must be validated by a qualified radiologist.
This AI analysis should not be used as the sole basis for clinical decision-making.
"""
        
        return report
    
    def save_analysis_results(self, results: Dict[str, Any], output_dir: Union[str, Path]) -> None:
        """Save analysis results to files"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save complete results as JSON
        results_path = output_dir / "cxr_analysis_results.json"
        
        # Create serializable version of results
        serializable_results = self._make_serializable(results)
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Save clinical report as text
        if 'clinical_report' in results:
            report_path = output_dir / "clinical_report.txt"
            with open(report_path, 'w') as f:
                for section, content in results['clinical_report'].items():
                    f.write(content + "\n\n")
        
        # Save segmentation masks
        if 'segmentation' in results and 'masks' in results['segmentation']:
            masks_dir = output_dir / "segmentation_masks"
            self.segmenter.save_masks(results['segmentation']['masks'], masks_dir)
        
        logger.info(f"Analysis results saved to {output_dir}")
    
    def _make_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj
    
    async def query_medical_knowledge(self, question: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Query medical knowledge using RAG system"""
        if not self.rag_system:
            raise ValueError("RAG system not initialized")
        
        try:
            # Add context if provided
            if context:
                context_str = f"Given the following CXR analysis context: {json.dumps(context, indent=2)}\n\nQuestion: {question}"
                query = context_str
            else:
                query = question
            
            response = await self.rag_system.agentic_query(query, max_context_chunks=5)
            return response
        
        except Exception as e:
            logger.error(f"Error querying medical knowledge: {str(e)}")
            raise
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and health check"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'components': {
                'image_processor': 'operational',
                'classifier': 'operational',
                'segmenter': 'operational',
                'feature_extractor': 'operational',
                'pathology_detector': 'operational',
                'rag_system': 'operational' if self.rag_system else 'not_initialized',
                'vector_store': 'operational' if self.vector_store else 'not_initialized'
            },
            'config': {
                'model_name': self.config.model.model_name,
                'chunk_size': self.config.document.chunk_size,
                'embedding_model': self.config.vector_store.embedding_model
            }
        }
        
        # Check vector store stats if available
        if self.vector_store:
            try:
                stats = self.vector_store.get_collection_stats()
                status['vector_store_stats'] = stats
            except Exception as e:
                status['vector_store_stats'] = {'error': str(e)}
        
        return status
