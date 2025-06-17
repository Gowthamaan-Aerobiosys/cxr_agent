import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import re
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document"""
    text: str
    metadata: Dict[str, Any]
    chunk_id: str
    source: str
    page_number: int
    
class DocumentProcessor:
    """Handles PDF document processing and text extraction"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text from PDF and return pages with metadata"""
        pages = []
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(doc.page_count):
                page = doc[page_num]
                text = page.get_text()
                if text.strip():  # Only include pages with text
                    pages.append({
                        'text': text,
                        'page_number': page_num + 1,
                        'source': os.path.basename(pdf_path)
                    })
            doc.close()
            logger.info(f"Extracted {len(pages)} pages from {pdf_path}")
            return pages
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {str(e)}")
            return []
    
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Split text into overlapping chunks"""
        chunks = []
        text = self.clean_text(text)
        
        # Split by sentences first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = ""
        current_length = 0
        chunk_id = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Create chunk
                chunk = DocumentChunk(
                    text=current_chunk.strip(),
                    metadata=metadata,
                    chunk_id=f"{metadata['source']}_chunk_{chunk_id}",
                    source=metadata['source'],
                    page_number=metadata['page_number']
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                current_chunk = overlap_text + " " + sentence
                current_length = len(current_chunk)
                chunk_id += 1
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                current_length += sentence_length
        
        # Add final chunk
        if current_chunk.strip():
            chunk = DocumentChunk(
                text=current_chunk.strip(),
                metadata=metadata,
                chunk_id=f"{metadata['source']}_chunk_{chunk_id}",
                source=metadata['source'],
                page_number=metadata['page_number']
            )
            chunks.append(chunk)
        
        return chunks
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep medical terms
        text = re.sub(r'[^\w\s\-\.\,\;\:\!\?\(\)\[\]\/\%\&]', '', text)
        return text.strip()
    
    def process_documents(self, pdf_directory: str) -> List[DocumentChunk]:
        """Process all PDFs in directory and return chunks"""
        all_chunks = []
        pdf_files = list(Path(pdf_directory).glob("*.pdf"))
        
        logger.info(f"Processing {len(pdf_files)} PDF files")
        
        for pdf_path in pdf_files:
            logger.info(f"Processing: {pdf_path.name}")
            pages = self.extract_text_from_pdf(str(pdf_path))
            
            for page_data in pages:
                chunks = self.chunk_text(page_data['text'], page_data)
                all_chunks.extend(chunks)
        
        logger.info(f"Generated {len(all_chunks)} text chunks total")
        return all_chunks

class VectorStore:
    """Manages vector embeddings and similarity search"""
    
    def __init__(self, collection_name: str = "respiratory_care_docs", 
                 embedding_model: str = "all-MiniLM-L6-v2"):
        self.collection_name = collection_name
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Loaded existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(name=collection_name)
            logger.info(f"Created new collection: {collection_name}")
    
    def add_documents(self, chunks: List[DocumentChunk]):
        """Add document chunks to vector store"""
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Prepare metadata
        metadatas = []
        ids = []
        
        for chunk in chunks:
            metadata = {
                'source': chunk.source,
                'page_number': chunk.page_number,
                'chunk_id': chunk.chunk_id
            }
            metadatas.append(metadata)
            ids.append(chunk.chunk_id)
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"Added {len(chunks)} documents to vector store")
    
    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        query_embedding = self.embedding_model.encode([query])
        
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i],
                'id': results['ids'][0][i]
            })
        
        return formatted_results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        count = self.collection.count()
        return {
            'total_documents': count,
            'collection_name': self.collection_name
        }

if __name__ == "__main__":
    # Example usage
    processor = DocumentProcessor()
    vector_store = VectorStore()
    
    # Process documents
    dataset_path = "../dataset/books"
    chunks = processor.process_documents(dataset_path)
    
    # Add to vector store
    if chunks:
        vector_store.add_documents(chunks)
        print(f"Successfully processed and stored {len(chunks)} document chunks")
        print(f"Collection stats: {vector_store.get_collection_stats()}")
