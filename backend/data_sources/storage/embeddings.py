"""
Generic pgvector embedding operations using LlamaIndex and OpenAI
Low-level embedding utilities that can be reused across different data sources
"""

from typing import List, Dict, Any, Optional, Tuple
import os
from pathlib import Path
import asyncio
import logging
from datetime import datetime
import re
import json
import tiktoken
import numpy as np

from dotenv import load_dotenv
load_dotenv()

# LlamaIndex imports
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import TextNode, BaseNode
from llama_index.core.vector_stores.types import VectorStoreQuery, VectorStoreQueryResult

# Database connection
import psycopg2
from sqlalchemy import create_engine, text

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_processed_text_files(
    chunks_file: str = "literacy_analysis_chunks.json",
    metadata_file: str = "literacy_analysis_metadata.json", 
    ids_file: str = "literacy_analysis_ids.json"
) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
    """
    Load processed text files from metadata.py output
    
    Args:
        chunks_file: Path to chunks JSON file
        metadata_file: Path to metadata JSON file  
        ids_file: Path to IDs JSON file
        
    Returns:
        Tuple of (chunks, metadatas, ids)
    """
    print(f"[DEBUG] Loading processed text files...")
    
    try:
        # Load chunks
        with open(chunks_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        print(f"[DEBUG] Loaded {len(chunks)} chunks from {chunks_file}")
        
        # Load metadata
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadatas = json.load(f)
        print(f"[DEBUG] Loaded {len(metadatas)} metadata entries from {metadata_file}")
        
        # Load IDs
        with open(ids_file, "r", encoding="utf-8") as f:
            ids = json.load(f)
        print(f"[DEBUG] Loaded {len(ids)} IDs from {ids_file}")
        
        # Verify lengths match
        if len(chunks) != len(metadatas) != len(ids):
            raise ValueError(f"Mismatched lengths: chunks={len(chunks)}, metadata={len(metadatas)}, ids={len(ids)}")
        
        print(f"[DEBUG] Top 5 chunks: {[chunk[:50] + '...' for chunk in chunks[:5]]}")
        print(f"[DEBUG] Top 5 metadata: {metadatas[:5]}")
        print(f"[DEBUG] Top 5 IDs: {ids[:5]}")
        
        return chunks, metadatas, ids
        
    except Exception as e:
        logger.error(f"Failed to load processed text files: {e}")
        raise

def load_combined_file(combined_file: str = "literacy_analysis_combined.json") -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
    """
    Load processed text from combined JSON file
    
    Args:
        combined_file: Path to combined JSON file
        
    Returns:
        Tuple of (chunks, metadatas, ids)
    """
    print(f"[DEBUG] Loading combined file: {combined_file}")
    
    try:
        with open(combined_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        chunks = data["chunks"]
        metadatas = data["metadatas"]
        ids = data["ids"]
        
        print(f"[DEBUG] Loaded from combined file:")
        print(f"[DEBUG] - {len(chunks)} chunks")
        print(f"[DEBUG] - {len(metadatas)} metadata entries")
        print(f"[DEBUG] - {len(ids)} IDs")
        
        print(f"[DEBUG] Top 5 chunks: {[chunk[:50] + '...' for chunk in chunks[:5]]}")
        print(f"[DEBUG] Top 5 metadata: {metadatas[:5]}")
        print(f"[DEBUG] Top 5 IDs: {ids[:5]}")
        
        return chunks, metadatas, ids
        
    except Exception as e:
        logger.error(f"Failed to load combined file: {e}")
        raise

class PGVectorEmbeddingService:
    """
    Generic embedding service using pgvector with LlamaIndex
    Handles low-level embedding operations that can be reused across data sources
    """
    
    def __init__(
        self, 
        table_name: str = "embeddings_table",
        embed_dim: int = 3072,  # OpenAI text-embedding-3-large dimension
        connection_string: Optional[str] = None
    ):
        """
        Initialize pgvector embedding service
        
        Args:
            table_name: Name of the pgvector table
            embed_dim: Embedding dimension for OpenAI large model
            connection_string: PostgreSQL connection string (uses env if None)
        """
        print(f"[DEBUG] Initializing PGVectorEmbeddingService with table: {table_name}")
        
        self.table_name = table_name
        self.embed_dim = embed_dim
        
        # Initialize OpenAI embedding model (LlamaIndex wrapper)
        self.embedding_model = OpenAIEmbedding(
            model="text-embedding-3-large",
            api_key=os.getenv("OPENAI_API_KEY"),
            embed_batch_size=100  # Batch size for efficiency
        )
        
        # Database connection setup
        self.connection_string = connection_string or self._build_connection_string()
        self.engine = create_engine(self.connection_string)
        
        # Initialize pgvector store
        self.vector_store = PGVectorStore.from_params(
            database=os.getenv("POSTGRES_DB", "lightbulb_db"),
            host=os.getenv("POSTGRES_HOST", "localhost"),
            password=os.getenv("POSTGRES_PASSWORD", "your_password"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            user=os.getenv("POSTGRES_USER", "postgres"),
            table_name=self.table_name,
            embed_dim=self.embed_dim,
            hnsw_kwargs={"hnsw_m": 16, "hnsw_ef_construction": 64},  # Performance tuning
            text_col="content",
            metadata_col="metadata",
            node_id_col="node_id",
            embedding_col="embedding"
        )
        
        # Storage context for LlamaIndex
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        
        print(f"[DEBUG] Embedding service initialized with dimension: {self.embed_dim}")
    
    def _build_connection_string(self) -> str:
        """Build PostgreSQL connection string from environment variables"""
        return (
            f"postgresql://{os.getenv('POSTGRES_USER', 'postgres')}:"
            f"{os.getenv('POSTGRES_PASSWORD', 'your_password')}@"
            f"{os.getenv('POSTGRES_HOST', 'localhost')}:"
            f"{os.getenv('POSTGRES_PORT', '5432')}/"
            f"{os.getenv('POSTGRES_DB', 'lightbulb_db')}"
        )
    
    def setup_database(self) -> None:
        """
        Setup pgvector extension and tables
        Run this once during initial setup
        """
        print("[DEBUG] Setting up pgvector database...")
        
        try:
            with self.engine.connect() as connection:
                # Enable pgvector extension
                connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                connection.commit()
                print("[DEBUG] pgvector extension enabled")
                
                # Create table will be handled by LlamaIndex automatically
                print(f"[DEBUG] Table '{self.table_name}' setup completed")
                
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            raise
    
    def create_embedding(self, text: str) -> List[float]:
        """
        Create embedding for a single text
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        print(f"[DEBUG] Creating embedding for text: {text[:50]}...")
        
        try:
            # Use LlamaIndex embedding model
            embedding = self.embedding_model.get_text_embedding(text)
            print(f"[DEBUG] Embedding created with dimension: {len(embedding)}")
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding creation failed: {e}")
            raise
    
    def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for multiple texts efficiently
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors
        """
        print(f"[DEBUG] Creating embeddings for {len(texts)} texts...")
        
        try:
            # Use LlamaIndex batch embedding
            embeddings = self.embedding_model.get_text_embedding_batch(texts)
            print(f"[DEBUG] Batch embeddings created: {len(embeddings)} vectors")
            return embeddings
            
        except Exception as e:
            logger.error(f"Batch embedding creation failed: {e}")
            raise
    
    def store_embeddings(
        self, 
        texts: List[str], 
        metadata_list: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Store texts and their embeddings in pgvector
        
        Args:
            texts: List of texts to store
            metadata_list: Optional metadata for each text
            ids: Optional custom IDs for each text
            
        Returns:
            List of node IDs that were stored
        """
        print(f"[DEBUG] Storing {len(texts)} texts with embeddings...")
        
        try:
            # Create TextNode objects for LlamaIndex
            nodes = []
            for i, text in enumerate(texts):
                node_id = ids[i] if ids else f"node_{datetime.now().timestamp()}_{i}"
                metadata = metadata_list[i] if metadata_list else {}
                
                node = TextNode(
                    text=text,
                    id_=node_id,
                    metadata=metadata
                )
                nodes.append(node)
            
            # Create index and store embeddings
            index = VectorStoreIndex(nodes, storage_context=self.storage_context)
            
            node_ids = [node.id_ for node in nodes]
            print(f"[DEBUG] Stored {len(node_ids)} embeddings successfully")
            print(f"[DEBUG] Top 5 stored IDs: {node_ids[:5]}")
            
            return node_ids
            
        except Exception as e:
            logger.error(f"Embedding storage failed: {e}")
            raise
    
    def embed_from_processed_files(
        self,
        chunks_file: str = "literacy_analysis_chunks.json",
        metadata_file: str = "literacy_analysis_metadata.json", 
        ids_file: str = "literacy_analysis_ids.json"
    ) -> List[str]:
        """
        Load processed files from metadata.py and embed them
        
        Args:
            chunks_file: Path to chunks JSON file
            metadata_file: Path to metadata JSON file  
            ids_file: Path to IDs JSON file
            
        Returns:
            List of node IDs that were stored
        """
        print(f"[DEBUG] Embedding from processed files...")
        
        # Load the processed data
        chunks, metadatas, ids = load_processed_text_files(chunks_file, metadata_file, ids_file)
        
        # Store embeddings using existing function
        node_ids = self.store_embeddings(chunks, metadatas, ids)
        
        print(f"[DEBUG] Successfully embedded {len(node_ids)} chunks from processed files")
        return node_ids
    
    def embed_from_combined_file(self, combined_file: str = "literacy_analysis_combined.json") -> List[str]:
        """
        Load combined file from metadata.py and embed it
        
        Args:
            combined_file: Path to combined JSON file
            
        Returns:
            List of node IDs that were stored
        """
        print(f"[DEBUG] Embedding from combined file...")
        
        # Load the combined data
        chunks, metadatas, ids = load_combined_file(combined_file)
        
        # Store embeddings using existing function
        node_ids = self.store_embeddings(chunks, metadatas, ids)
        
        print(f"[DEBUG] Successfully embedded {len(node_ids)} chunks from combined file")
        return node_ids
    
    def delete_embeddings(self, node_ids: List[str]) -> bool:
        """
        Delete embeddings by node IDs
        
        Args:
            node_ids: List of node IDs to delete
            
        Returns:
            True if successful
        """
        print(f"[DEBUG] Deleting {len(node_ids)} embeddings...")
        
        try:
            self.vector_store.delete_nodes(node_ids)
            print(f"[DEBUG] Successfully deleted embeddings")
            return True
            
        except Exception as e:
            logger.error(f"Embedding deletion failed: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the embedding store
        
        Returns:
            Dictionary with store statistics
        """
        try:
            with self.engine.connect() as connection:
                # Get table count
                count_query = text(f"SELECT COUNT(*) FROM {self.table_name}")
                result = connection.execute(count_query)
                total_embeddings = result.scalar()
                
                stats = {
                    "total_embeddings": total_embeddings,
                    "table_name": self.table_name,
                    "embedding_dimension": self.embed_dim,
                    "model": "text-embedding-3-large"
                }
                
                print(f"[DEBUG] Embedding store stats: {stats}")
                return stats
                
        except Exception as e:
            logger.error(f"Stats retrieval failed: {e}")
            return {"error": str(e)}


# Factory function for easy instantiation
def create_embedding_service(
    table_name: str = "default_embeddings",
    connection_string: Optional[str] = None
) -> PGVectorEmbeddingService:
    """
    Factory function to create embedding service with common defaults
    
    Args:
        table_name: Name for the pgvector table
        connection_string: Optional custom connection string
        
    Returns:
        Configured PGVectorEmbeddingService instance
    """
    print(f"[DEBUG] Creating embedding service for table: {table_name}")
    
    service = PGVectorEmbeddingService(
        table_name=table_name,
        connection_string=connection_string
    )
    
    return service


# Example usage placeholder
if __name__ == "__main__":
    # This is for testing - remove in production
    print("[DEBUG] Testing embedding service...")
    
    # Initialize service
    service = create_embedding_service("test_embeddings")
    
    # Setup database (run once)
    # service.setup_database()
    
    # Example operations
    # embeddings = service.create_embeddings_batch(["Hello world", "How are you?"])
    # results = service.similarity_search("greeting", limit=3)
    
    print("[DEBUG] Embedding service test completed")
