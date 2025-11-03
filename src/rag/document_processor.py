import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any
import fitz 
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import json
import torch
import re
from datetime import datetime
from .utils import get_device

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
        self.global_chunk_counter = 0

    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text from PDF and return pages with metadata"""
        pages = []
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(doc.page_count):
                page = doc[page_num]
                text = page.get_text()
                if text.strip():  # Only include pages with text
                    pages.append(
                        {
                            "text": text,
                            "page_number": page_num + 1,
                            "source": os.path.basename(pdf_path),
                        }
                    )
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
        sentences = re.split(r"(?<=[.!?])\s+", text)

        current_chunk = ""
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Create chunk with globally unique ID
                chunk = DocumentChunk(
                    text=current_chunk.strip(),
                    metadata=metadata,
                    chunk_id=f"{metadata['source']}_page_{metadata['page_number']}_chunk_{self.global_chunk_counter}",
                    source=metadata["source"],
                    page_number=metadata["page_number"],
                )
                chunks.append(chunk)
                self.global_chunk_counter += 1

                # Start new chunk with overlap
                overlap_text = (
                    current_chunk[-self.chunk_overlap :]
                    if len(current_chunk) > self.chunk_overlap
                    else current_chunk
                )
                current_chunk = overlap_text + " " + sentence
                current_length = len(current_chunk)
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                current_length += sentence_length

        # Add final chunk
        if current_chunk.strip():
            chunk = DocumentChunk(
                text=current_chunk.strip(),
                metadata=metadata,
                chunk_id=f"{metadata['source']}_page_{metadata['page_number']}_chunk_{self.global_chunk_counter}",
                source=metadata["source"],
                page_number=metadata["page_number"],
            )
            chunks.append(chunk)
            self.global_chunk_counter += 1

        return chunks

    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove special characters but keep medical terms
        text = re.sub(r"[^\w\s\-\.\,\;\:\!\?\(\)\[\]\/\%\&]", "", text)
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
                chunks = self.chunk_text(page_data["text"], page_data)
                all_chunks.extend(chunks)

        logger.info(f"Generated {len(all_chunks)} text chunks total")
        return all_chunks


class VectorStore:
    """Manages vector embeddings and similarity search"""

    def __init__(
        self,
        collection_name: str = "respiratory_care_docs",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.collection_name = collection_name

        # Initialize embedding model with proper device handling
        device = get_device()
        logger.info(f"Initializing embedding model on device: {device}")

        # Try different embedding models in order of preference
        models_to_try = [
            embedding_model,
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
        ]

        self.embedding_model = None
        for model_name in models_to_try:
            try:
                # Force CPU for embedding model to avoid reshape issues
                logger.info(f"Attempting to load {model_name}")
                self.embedding_model = SentenceTransformer(
                    model_name, device=get_device()
                )
                logger.info(
                    f"Successfully loaded embedding model: {model_name} on {get_device()}"
                )
                break
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {str(e)}")
                continue

        if self.embedding_model is None:
            raise RuntimeError("Failed to load any embedding model")

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path="./chroma_db", settings=Settings(anonymized_telemetry=False)
        )

        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Loaded existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(name=collection_name)
            logger.info(f"Created new collection: {collection_name}")

    def clear_collection(self):
        """Clear all documents from the collection"""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(name=self.collection_name)
            logger.info(f"Cleared and recreated collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}")

    def add_documents(
        self,
        chunks: List[DocumentChunk],
        clear_existing: bool = False,
        batch_size: int = 100,
    ):
        """Add document chunks to vector store in batches"""
        if clear_existing:
            self.clear_collection()

        # Check for existing documents to avoid duplicates
        existing_count = self.collection.count()
        if existing_count > 0:
            logger.info(f"Collection already contains {existing_count} documents")

        total_chunks = len(chunks)
        logger.info(
            f"Adding {total_chunks} chunks to vector store in batches of {batch_size}..."
        )

        # Process chunks in batches
        for i in range(0, total_chunks, batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_chunks + batch_size - 1) // batch_size

            logger.info(
                f"Processing batch {batch_num}/{total_batches} ({len(batch_chunks)} chunks)"
            )

            texts = [chunk.text for chunk in batch_chunks]

            # Truncate texts that are too long for the embedding model
            max_length = 256  # More conservative length to avoid issues
            truncated_texts = []
            for text in texts:
                # Count tokens more accurately
                words = text.split()
                if len(words) > max_length:
                    truncated_text = " ".join(words[:max_length])
                    truncated_texts.append(truncated_text)
                else:
                    truncated_texts.append(text)

            try:
                # Clear any cached computations to free memory
                if hasattr(torch, "cuda") and torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Encode with smaller batch size and error handling
                embeddings = self.embedding_model.encode(
                    truncated_texts,
                    show_progress_bar=True,
                    batch_size=8,  # Much smaller batch size to avoid memory issues
                    convert_to_tensor=False,  # Return numpy arrays instead of tensors
                    normalize_embeddings=True,  # Normalize embeddings for better similarity
                )

                # Ensure embeddings are numpy arrays
                if not isinstance(embeddings, np.ndarray):
                    embeddings = np.array(embeddings)

            except Exception as e:
                logger.error(f"Error encoding batch: {str(e)}")
                logger.info("Trying single-text encoding as fallback...")

                # Try with single text encoding as fallback
                embeddings = []
                embedding_dim = None

                for idx, text in enumerate(truncated_texts):
                    try:
                        # Clear cache before each encoding
                        if hasattr(torch, "cuda") and torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        emb = self.embedding_model.encode(
                            [text], convert_to_tensor=False, normalize_embeddings=True
                        )

                        if isinstance(emb, list):
                            emb = np.array(emb)
                        if emb.ndim > 1:
                            emb = emb[0]  # Get first embedding if batch

                        embeddings.append(emb)

                        if embedding_dim is None:
                            embedding_dim = len(emb)

                    except Exception as single_error:
                        logger.error(f"Error encoding text {idx}: {str(single_error)}")

                        # Create zero embedding as fallback
                        if embedding_dim is None:
                            # Try to get model dimension
                            try:
                                embedding_dim = (
                                    self.embedding_model.get_sentence_embedding_dimension()
                                )
                            except:
                                embedding_dim = (
                                    384  # Default dimension for all-MiniLM-L6-v2
                                )

                        fallback_embedding = np.zeros(embedding_dim, dtype=np.float32)
                        embeddings.append(fallback_embedding)

                embeddings = np.array(embeddings)
                logger.info(
                    f"Completed fallback encoding with shape: {embeddings.shape}"
                )

            # Prepare metadata
            metadatas = []
            ids = []

            for chunk in batch_chunks:
                metadata = {
                    "source": chunk.source,
                    "page_number": chunk.page_number,
                    "chunk_id": chunk.chunk_id,
                }
                metadatas.append(metadata)
                ids.append(chunk.chunk_id)

            # Add batch to collection
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas,
                ids=ids,
            )

            logger.info(
                f"Added batch {batch_num}/{total_batches} ({len(batch_chunks)} documents) to vector store"
            )

        logger.info(f"Successfully added all {total_chunks} documents to vector store")

    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        try:
            # Clear cache before encoding
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()

            query_embedding = self.embedding_model.encode(
                [query], convert_to_tensor=False, normalize_embeddings=True
            )

            # Ensure proper format
            if isinstance(query_embedding, np.ndarray):
                if query_embedding.ndim > 1:
                    query_embedding = query_embedding[0]
                query_embedding = query_embedding.tolist()
            elif isinstance(query_embedding, list) and len(query_embedding) > 0:
                if isinstance(query_embedding[0], np.ndarray):
                    query_embedding = query_embedding[0].tolist()

            results = self.collection.query(
                query_embeddings=[query_embedding], n_results=n_results
            )

            # Format results
            formatted_results = []
            for i in range(len(results["documents"][0])):
                formatted_results.append(
                    {
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i],
                        "id": results["ids"][0][i],
                    }
                )

            return formatted_results

        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            # Return empty results if search fails
            return []

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        count = self.collection.count()
        return {"total_documents": count, "collection_name": self.collection_name}


if __name__ == "__main__":
    # Example usage
    processor = DocumentProcessor()
    vector_store = VectorStore()

    # Process documents
    dataset_path = "../dataset/books"
    chunks = processor.process_documents(dataset_path)

    # Add to vector store (clear existing to avoid duplicates)
    if chunks:
        vector_store.add_documents(chunks, clear_existing=True)
        print(f"Successfully processed and stored {len(chunks)} document chunks")
        print(f"Collection stats: {vector_store.get_collection_stats()}")
