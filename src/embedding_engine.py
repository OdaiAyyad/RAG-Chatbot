"""
Embedding & Indexing Module

This module handles:
1. Converting text to embeddings (vectors)
2. Building FAISS index for fast similarity search
3. Saving/loading indexes and metadata

Key Concepts:
- Embedding: Neural network that maps text → high-dimensional vector
- FAISS: Facebook AI Similarity Search - optimized for billion-scale search
- Cosine Similarity: Measures angle between vectors (0 = orthogonal, 1 = identical)
"""

import numpy as np
import faiss
import json
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
from config import (
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    CHUNK_METADATA_PATH
)


class EmbeddingEngine:
    """
    Handles text-to-vector conversion using sentence-transformers.
    
    Why sentence-transformers?
    - Pre-trained on semantic similarity tasks
    - Better than raw BERT for this use case
    - Fast inference (no GPU needed for small-scale)
    """
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """
        Load the embedding model.
        
        First run will download ~80MB model from HuggingFace.
        Subsequent runs load from cache (~/.cache/torch/sentence_transformers/)
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def embed_texts(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Convert list of texts to embeddings.
        
        Args:
            texts: List of strings to embed
            show_progress: Show progress bar (useful for large batches)
        
        Returns:
            numpy array of shape (num_texts, embedding_dim)
        
        Technical Details:
        - Uses mean pooling over token embeddings
        - Normalizes vectors to unit length (for cosine similarity)
        - Batches internally for efficiency
        """
        print(f"Embedding {len(texts)} texts...")
        
        # encode() handles batching automatically
        # convert_to_numpy=True returns numpy array (not torch tensor)
        # show_progress_bar for user feedback
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
            batch_size=32  # Process 32 texts at once (tune based on RAM)
        )
        
        print(f"Embedding complete. Shape: {embeddings.shape}")
        return embeddings


class FAISSIndex:
    """
    Manages FAISS index for fast similarity search.
    
    FAISS Index Types (we use IndexFlatIP):
    - IndexFlatL2: Euclidean distance (not ideal for text)
    - IndexFlatIP: Inner Product = Cosine similarity if vectors are normalized
    - IndexIVFFlat: For 100k+ vectors (quantization for speed)
    - IndexHNSW: Graph-based, great for <1M vectors
    
    For this demo, IndexFlatIP is perfect (exact search, fast enough for 10k chunks).
    """
    
    def __init__(self, embedding_dim: int):
        """
        Initialize FAISS index.
        
        Args:
            embedding_dim: Dimensionality of embeddings (e.g., 384)
        """
        self.embedding_dim = embedding_dim
        
        # IndexFlatIP: Flat (exact search) using Inner Product
        # We'll normalize embeddings, so IP = cosine similarity
        self.index = faiss.IndexFlatIP(embedding_dim)
        
        # Metadata storage (FAISS only stores vectors, not text)
        self.chunk_metadata = []
        
        print(f"Initialized FAISS index (dim={embedding_dim})")
    
    def add_embeddings(self, embeddings: np.ndarray, metadata: List[Dict]):
        """
        Add embeddings to the index.
        
        Args:
            embeddings: numpy array of shape (N, embedding_dim)
            metadata: List of N dicts with chunk info (text, doc_id, etc.)
        
        Important: 
        - Normalize embeddings BEFORE adding (for cosine similarity)
        - FAISS index and metadata list must stay in sync
        """
        if len(embeddings) != len(metadata):
            raise ValueError(f"Mismatch: {len(embeddings)} embeddings but {len(metadata)} metadata entries")
        
        # Normalize embeddings to unit length
        # After normalization, inner product = cosine similarity
        # ||a|| = ||b|| = 1 → a·b = cos(θ)
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        
        # Store metadata
        self.chunk_metadata.extend(metadata)
        
        print(f"Added {len(embeddings)} embeddings. Total in index: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for most similar embeddings.
        
        Args:
            query_embedding: Single embedding vector (1, embedding_dim)
            top_k: Number of results to return
        
        Returns:
            scores: Array of similarity scores (higher = more similar)
            indices: Array of chunk indices in metadata list
        
        Important:
        - Normalize query embedding before search!
        - Scores are cosine similarities (range: -1 to 1, typically 0 to 1)
        """
        # Ensure query is 2D: (1, embedding_dim)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize query
        faiss.normalize_L2(query_embedding)
        
        # Search
        # Returns: (distances, indices) both shape (1, top_k)
        scores, indices = self.index.search(query_embedding, top_k)
        
        return scores[0], indices[0]  # Return 1D arrays
    
    def save(self, index_path: str = FAISS_INDEX_PATH, metadata_path: str = CHUNK_METADATA_PATH):
        """
        Save FAISS index and metadata to disk.
        
        Why separate files?
        - FAISS index: Binary format (efficient)
        - Metadata: JSON (human-readable, easy to inspect)
        """
        print(f"Saving FAISS index to: {index_path}")
        faiss.write_index(self.index, index_path)
        
        print(f"Saving metadata to: {metadata_path}")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunk_metadata, f, indent=2, ensure_ascii=False)
        
        print("Save complete!")
    
    @classmethod
    def load(cls, index_path: str = FAISS_INDEX_PATH, metadata_path: str = CHUNK_METADATA_PATH):
        """
        Load FAISS index and metadata from disk.
        
        Returns:
            FAISSIndex instance with loaded data
        """
        print(f"Loading FAISS index from: {index_path}")
        index = faiss.read_index(index_path)
        
        print(f"Loading metadata from: {metadata_path}")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Create instance
        embedding_dim = index.d  # FAISS index stores dimensionality
        faiss_index = cls(embedding_dim)
        faiss_index.index = index
        faiss_index.chunk_metadata = metadata
        
        print(f"Loaded index with {index.ntotal} embeddings")
        return faiss_index


def build_index_from_chunks(chunks: List[Dict]) -> FAISSIndex:
    """
    Complete pipeline: chunks → embeddings → FAISS index.
    
    This is the main function you'll call during setup.
    
    Args:
        chunks: List of chunk dicts from data_processor.py
    
    Returns:
        FAISSIndex ready for searching
    """
    # Step 1: Initialize embedding model
    embedding_engine = EmbeddingEngine()
    
    # Step 2: Extract texts and prepare metadata
    texts = [chunk['text'] for chunk in chunks]
    
    # Metadata we want to retrieve later
    metadata = [
        {
            'text': chunk['text'],
            'doc_id': chunk['doc_id'],
            'chunk_id': chunk['chunk_id'],
            'token_count': chunk['token_count']
        }
        for chunk in chunks
    ]
    
    # Step 3: Generate embeddings
    embeddings = embedding_engine.embed_texts(texts)
    
    # Step 4: Build FAISS index
    faiss_index = FAISSIndex(embedding_engine.embedding_dim)
    faiss_index.add_embeddings(embeddings, metadata)
    
    return faiss_index


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Test with sample data
    sample_chunks = [
        {
            'text': 'To reset your password, go to Settings and click Forgot Password.',
            'doc_id': 'doc_1',
            'chunk_id': 0,
            'token_count': 15
        },
        {
            'text': 'Our premium plan costs $29 per month and includes unlimited storage.',
            'doc_id': 'doc_2',
            'chunk_id': 0,
            'token_count': 12
        }
    ]
    
    # Build index
    index = build_index_from_chunks(sample_chunks)
    
    # Test search
    embedding_engine = EmbeddingEngine()
    query = "How do I change my password?"
    query_embedding = embedding_engine.embed_texts([query])
    
    scores, indices = index.search(query_embedding, top_k=2)
    
    print("\nSearch Results:")
    for i, (score, idx) in enumerate(zip(scores, indices)):
        print(f"\nRank {i+1}:")
        print(f"  Score: {score:.4f}")
        print(f"  Text: {index.chunk_metadata[idx]['text']}")