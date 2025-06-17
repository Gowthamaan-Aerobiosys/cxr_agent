"""
MCP Server for CXR Agent
Provides scalable access to CXR analysis capabilities via Model Context Protocol
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import base64
import io
from datetime import datetime

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
from PIL import Image

from cxr_agent import CXRAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for request/response validation
class CXRAnalysisRequest(BaseModel):
    image_path: Optional[str] = None
    image_base64: Optional[str] = None
    include_rag: bool = True
    generate_report: bool = True
    analysis_options: Dict[str, Any] = Field(default_factory=dict)

class CXRAnalysisResponse(BaseModel):
    success: bool
    analysis_id: str
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: str

class MedicalQueryRequest(BaseModel):
    question: str
    context: Optional[Dict[str, Any]] = None
    max_context_chunks: int = 5

class MedicalQueryResponse(BaseModel):
    success: bool
    query_id: str
    response: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: str

class SystemStatusResponse(BaseModel):
    success: bool
    status: Dict[str, Any]
    timestamp: str

class BatchAnalysisRequest(BaseModel):
    image_paths: List[str]
    include_rag: bool = True
    generate_report: bool = True
    batch_id: Optional[str] = None

class BatchAnalysisResponse(BaseModel):
    success: bool
    batch_id: str
    status: str  # 'queued', 'processing', 'completed', 'failed'
    results: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
    timestamp: str

# MCP Server Implementation
class CXRAgentMCPServer:
    """MCP Server for CXR Agent"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.app = FastAPI(
            title="CXR Agent MCP Server",
            description="Model Context Protocol Server for CXR Analysis",
            version="1.0.0"
        )
        
        # Configure CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Initialize CXR Agent
        self.cxr_agent = CXRAgent(config)
        
        # Track ongoing analyses
        self.active_analyses = {}
        self.batch_analyses = {}
        
        # Setup routes
        self._setup_routes()
        
        logger.info("CXR Agent MCP Server initialized")
    
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/", response_model=Dict[str, str])
        async def root():
            return {
                "service": "CXR Agent MCP Server",
                "version": "1.0.0",
                "status": "operational",
                "endpoints": [
                    "/analyze",
                    "/batch_analyze", 
                    "/query",
                    "/status",
                    "/health"
                ]
            }
        
        @self.app.post("/analyze", response_model=CXRAnalysisResponse)
        async def analyze_cxr(request: CXRAnalysisRequest):
            """Analyze a single CXR image"""
            analysis_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            try:
                # Handle image input
                image_path = None
                if request.image_base64:
                    # Decode base64 image
                    image_path = await self._save_base64_image(request.image_base64, analysis_id)
                elif request.image_path:
                    image_path = request.image_path
                else:
                    raise HTTPException(status_code=400, detail="Either image_path or image_base64 must be provided")
                
                # Perform analysis
                self.active_analyses[analysis_id] = "processing"
                
                results = await self.cxr_agent.analyze_cxr(
                    image_path=image_path,
                    include_rag=request.include_rag,
                    generate_report=request.generate_report
                )
                
                self.active_analyses[analysis_id] = "completed"
                
                return CXRAnalysisResponse(
                    success=True,
                    analysis_id=analysis_id,
                    results=results,
                    timestamp=datetime.now().isoformat()
                )
                
            except Exception as e:
                self.active_analyses[analysis_id] = "failed"
                logger.error(f"Analysis failed for {analysis_id}: {str(e)}")
                
                return CXRAnalysisResponse(
                    success=False,
                    analysis_id=analysis_id,
                    error=str(e),
                    timestamp=datetime.now().isoformat()
                )
        
        @self.app.post("/upload_analyze")
        async def upload_and_analyze(
            file: UploadFile = File(...),
            include_rag: bool = True,
            generate_report: bool = True
        ):
            """Upload and analyze CXR image"""
            analysis_id = f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            try:
                # Save uploaded file
                upload_dir = Path("uploads")
                upload_dir.mkdir(exist_ok=True)
                
                file_path = upload_dir / f"{analysis_id}_{file.filename}"
                
                with open(file_path, "wb") as buffer:
                    content = await file.read()
                    buffer.write(content)
                
                # Perform analysis
                self.active_analyses[analysis_id] = "processing"
                
                results = await self.cxr_agent.analyze_cxr(
                    image_path=str(file_path),
                    include_rag=include_rag,
                    generate_report=generate_report
                )
                
                self.active_analyses[analysis_id] = "completed"
                
                return CXRAnalysisResponse(
                    success=True,
                    analysis_id=analysis_id,
                    results=results,
                    timestamp=datetime.now().isoformat()
                )
                
            except Exception as e:
                self.active_analyses[analysis_id] = "failed"
                logger.error(f"Upload analysis failed for {analysis_id}: {str(e)}")
                
                return CXRAnalysisResponse(
                    success=False,
                    analysis_id=analysis_id,
                    error=str(e),
                    timestamp=datetime.now().isoformat()
                )
        
        @self.app.post("/batch_analyze", response_model=BatchAnalysisResponse)
        async def batch_analyze_cxr(request: BatchAnalysisRequest, background_tasks: BackgroundTasks):
            """Analyze multiple CXR images in batch"""
            batch_id = request.batch_id or f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            try:
                # Initialize batch tracking
                self.batch_analyses[batch_id] = {
                    "status": "queued",
                    "total_images": len(request.image_paths),
                    "completed": 0,
                    "failed": 0,
                    "results": []
                }
                
                # Start batch processing in background
                background_tasks.add_task(
                    self._process_batch,
                    batch_id,
                    request.image_paths,
                    request.include_rag,
                    request.generate_report
                )
                
                return BatchAnalysisResponse(
                    success=True,
                    batch_id=batch_id,
                    status="queued",
                    timestamp=datetime.now().isoformat()
                )
                
            except Exception as e:
                logger.error(f"Batch analysis failed for {batch_id}: {str(e)}")
                
                return BatchAnalysisResponse(
                    success=False,
                    batch_id=batch_id,
                    status="failed",
                    error=str(e),
                    timestamp=datetime.now().isoformat()
                )
        
        @self.app.get("/batch_status/{batch_id}", response_model=BatchAnalysisResponse)
        async def get_batch_status(batch_id: str):
            """Get status of batch analysis"""
            if batch_id not in self.batch_analyses:
                raise HTTPException(status_code=404, detail="Batch not found")
            
            batch_info = self.batch_analyses[batch_id]
            
            return BatchAnalysisResponse(
                success=True,
                batch_id=batch_id,
                status=batch_info["status"],
                results=batch_info.get("results"),
                timestamp=datetime.now().isoformat()
            )
        
        @self.app.post("/query", response_model=MedicalQueryResponse)
        async def query_medical_knowledge(request: MedicalQueryRequest):
            """Query medical knowledge using RAG"""
            query_id = f"query_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            try:
                response = await self.cxr_agent.query_medical_knowledge(
                    question=request.question,
                    context=request.context
                )
                
                return MedicalQueryResponse(
                    success=True,
                    query_id=query_id,
                    response=response,
                    timestamp=datetime.now().isoformat()
                )
                
            except Exception as e:
                logger.error(f"Medical query failed for {query_id}: {str(e)}")
                
                return MedicalQueryResponse(
                    success=False,
                    query_id=query_id,
                    error=str(e),
                    timestamp=datetime.now().isoformat()
                )
        
        @self.app.get("/status", response_model=SystemStatusResponse)
        async def get_system_status():
            """Get system status"""
            try:
                status = self.cxr_agent.get_system_status()
                
                # Add server-specific status
                status.update({
                    "server": {
                        "active_analyses": len(self.active_analyses),
                        "batch_analyses": len(self.batch_analyses),
                        "uptime": "operational"  # Could add actual uptime tracking
                    }
                })
                
                return SystemStatusResponse(
                    success=True,
                    status=status,
                    timestamp=datetime.now().isoformat()
                )
                
            except Exception as e:
                logger.error(f"Status check failed: {str(e)}")
                
                return SystemStatusResponse(
                    success=False,
                    status={"error": str(e)},
                    timestamp=datetime.now().isoformat()
                )
        
        @self.app.get("/health")
        async def health_check():
            """Simple health check endpoint"""
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
        
        @self.app.get("/analysis_status/{analysis_id}")
        async def get_analysis_status(analysis_id: str):
            """Get status of individual analysis"""
            if analysis_id not in self.active_analyses:
                raise HTTPException(status_code=404, detail="Analysis not found")
            
            return {
                "analysis_id": analysis_id,
                "status": self.active_analyses[analysis_id],
                "timestamp": datetime.now().isoformat()
            }
    
    async def _save_base64_image(self, base64_string: str, analysis_id: str) -> str:
        """Save base64 encoded image to disk"""
        try:
            # Remove data URL prefix if present
            if base64_string.startswith('data:image'):
                base64_string = base64_string.split(',')[1]
            
            # Decode base64
            image_data = base64.b64decode(base64_string)
            
            # Create uploads directory
            upload_dir = Path("uploads")
            upload_dir.mkdir(exist_ok=True)
            
            # Save image
            image_path = upload_dir / f"{analysis_id}.png"
            
            # Convert to PIL Image and save
            image = Image.open(io.BytesIO(image_data))
            image.save(image_path)
            
            return str(image_path)
            
        except Exception as e:
            logger.error(f"Error saving base64 image: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")
    
    async def _process_batch(self, batch_id: str, image_paths: List[str], 
                           include_rag: bool, generate_report: bool):
        """Process batch analysis in background"""
        try:
            self.batch_analyses[batch_id]["status"] = "processing"
            results = []
            
            for i, image_path in enumerate(image_paths):
                try:
                    # Analyze individual image
                    result = await self.cxr_agent.analyze_cxr(
                        image_path=image_path,
                        include_rag=include_rag,
                        generate_report=generate_report
                    )
                    
                    results.append({
                        "image_path": image_path,
                        "success": True,
                        "result": result
                    })
                    
                    self.batch_analyses[batch_id]["completed"] += 1
                    
                except Exception as e:
                    logger.error(f"Failed to analyze {image_path}: {str(e)}")
                    
                    results.append({
                        "image_path": image_path,
                        "success": False,
                        "error": str(e)
                    })
                    
                    self.batch_analyses[batch_id]["failed"] += 1
                
                # Update progress
                progress = (i + 1) / len(image_paths) * 100
                logger.info(f"Batch {batch_id} progress: {progress:.1f}%")
            
            # Update batch status
            self.batch_analyses[batch_id]["status"] = "completed"
            self.batch_analyses[batch_id]["results"] = results
            
            logger.info(f"Batch {batch_id} completed: {len(results)} images processed")
            
        except Exception as e:
            logger.error(f"Batch processing failed for {batch_id}: {str(e)}")
            self.batch_analyses[batch_id]["status"] = "failed"
            self.batch_analyses[batch_id]["error"] = str(e)
    
    def run(self, host: str = "0.0.0.0", port: int = 8000, **kwargs):
        """Run the MCP server"""
        uvicorn.run(self.app, host=host, port=port, **kwargs)

# CLI for running the server
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="CXR Agent MCP Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    
    args = parser.parse_args()
    
    # Initialize and run server
    server = CXRAgentMCPServer()
    
    server.run(
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers
    )

if __name__ == "__main__":
    main()
