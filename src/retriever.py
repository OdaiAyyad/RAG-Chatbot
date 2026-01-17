"""
Retrieval Module

This module handles:
1. Query processing
2. Similarity search via FAISS
3. Confidence-based filtering
4. Result ranking and formatting

Key Concept: Confidence Threshold
- FAISS always returns K results, even if they're garbage
- We filter results below a similarity threshold
- If no results pass, we escalate to human
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from embedding_engine import EmbeddingEngine, FAISSIndex
from config import (
    TOP_K_RETRIEVAL,
    CONFIDENCE_THRESHOLD,
    FAISS_INDEX_PATH,
    CHUNK_METADATA_PATH
)


class RetrievalResult:
    """
    Data class for search results.
    
    Why a class instead of a dict?
    - Type safety
    - Auto-completion in IDEs
    - Clear interface
    """
    
    def __init__(
        self, 
        text: str, 
        score: float, 
        doc_id: str, 
        chunk_id: int,
        rank: int
    ):
        self.text = text
        self.score = score  # Cosine similarity (0.0 to 1.0)
        self.doc_id = doc_id
        self.chunk_id = chunk_id
        self.rank = rank  # 1-indexed rank
    
    def __repr__(self):
        return f"RetrievalResult(rank={self.rank}, score={self.score:.3f}, doc={self.doc_id})"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'text': self.text,
            'score': self.score,
            'doc_id': self.doc_id,
            'chunk_id': self.chunk_id,
            'rank': self.rank
        }


class Retriever:
    """
    Main retrieval engine.
    
    Responsibilities:
    1. Load FAISS index and embedding model
    2. Process queries
    3. Apply confidence filtering
    4. Return ranked results
    """
    
    def __init__(
        self,
        faiss_index: Optional[FAISSIndex] = None,
        embedding_engine: Optional[EmbeddingEngine] = None,
        top_k: int = TOP_K_RETRIEVAL,
        confidence_threshold: float = CONFIDENCE_THRESHOLD
    ):
        """
        Initialize retriever.
        
        Args:
            faiss_index: Pre-loaded FAISS index (or None to load from disk)
            embedding_engine: Pre-loaded embedding model (or None to load)
            top_k: Number of candidates to retrieve
            confidence_threshold: Minimum similarity score to accept
        """
        # Load index if not provided
        if faiss_index is None:
            print("Loading FAISS index from disk...")
            self.faiss_index = FAISSIndex.load(FAISS_INDEX_PATH, CHUNK_METADATA_PATH)
        else:
            self.faiss_index = faiss_index
        
        # Load embedding model if not provided
        if embedding_engine is None:
            print("Loading embedding model...")
            self.embedding_engine = EmbeddingEngine()
        else:
            self.embedding_engine = embedding_engine
        
        self.top_k = top_k
        self.confidence_threshold = confidence_threshold
        
        print(f"Retriever initialized:")
        print(f"  - Index size: {self.faiss_index.index.ntotal} chunks")
        print(f"  - Top K: {self.top_k}")
        print(f"  - Confidence threshold: {self.confidence_threshold}")
    
    def retrieve(self, query: str) -> Tuple[List[RetrievalResult], bool]:
        """
        Main retrieval function.
        
        Args:
            query: User's question
        
        Returns:
            (results, has_confident_match)
            - results: List of RetrievalResult objects (may be empty)
            - has_confident_match: True if at least one result passed threshold
        
        Flow:
        1. Embed query
        2. Search FAISS
        3. Filter by confidence
        4. Format results
        """
        # Step 1: Embed the query
        query_embedding = self.embedding_engine.embed_texts([query], show_progress=False)
        
        # Step 2: Search FAISS index
        scores, indices = self.faiss_index.search(query_embedding, self.top_k)
        
        # Step 3: Filter and format results
        results = []
        for rank, (score, idx) in enumerate(zip(scores, indices), start=1):
            # Only include results above confidence threshold
            if score < self.confidence_threshold:
                # Log this for debugging/monitoring
                print(f"  Rank {rank}: score={score:.3f} below threshold, skipping")
                continue
            
            # Retrieve metadata
            metadata = self.faiss_index.chunk_metadata[idx]
            
            # Create result object
            result = RetrievalResult(
                text=metadata['text'],
                score=float(score),  # Convert numpy float to Python float
                doc_id=metadata['doc_id'],
                chunk_id=metadata['chunk_id'],
                rank=rank
            )
            results.append(result)
        
        # Step 4: Determine if we have confident matches
        has_confident_match = len(results) > 0
        
        # Log retrieval stats
        print(f"\nRetrieval for query: '{query[:50]}...'")
        print(f"  Found {len(results)} results above threshold")
        if results:
            print(f"  Top score: {results[0].score:.3f}")
        else:
            print(f"  Best score: {scores[0]:.3f} (below threshold)")
        
        return results, has_confident_match
    
    def format_context_for_llm(self, results: List[RetrievalResult]) -> str:
        """
        Format retrieved chunks into context string for LLM.
        
        Format:
        ---
        Document 1 (doc_id, score):
        <text>
        
        Document 2 (doc_id, score):
        <text>
        ---
        
        Why this format?
        - Clear document boundaries
        - Includes score for LLM to gauge relevance
        - Numbered for easy citation
        """
        if not results:
            return "No relevant documents found."
        
        context_parts = []
        for result in results:
            context_parts.append(
                f"Document {result.rank} (ID: {result.doc_id}, Similarity: {result.score:.2f}):\n"
                f"{result.text}"
            )
        
        return "\n\n---\n\n".join(context_parts)
    
    def retrieve_and_format(self, query: str) -> Tuple[str, List[RetrievalResult], bool]:
        """
        Convenience method: retrieve + format in one call.
        
        Returns:
            (context_string, results, has_confident_match)
        """
        results, has_confident_match = self.retrieve(query)
        context = self.format_context_for_llm(results)
        return context, results, has_confident_match


# ============================================================================
# DEBUGGING & ANALYSIS UTILITIES
# ============================================================================

def analyze_retrieval_quality(retriever: Retriever, test_queries: List[str]):
    """
    Helper function to test retrieval quality on a set of queries.
    
    Use this during development to tune your confidence threshold.
    
    Args:
        retriever: Retriever instance
        test_queries: List of example questions
    """
    print("\n" + "="*80)
    print("RETRIEVAL QUALITY ANALYSIS")
    print("="*80)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[Query {i}] {query}")
        print("-" * 80)
        
        results, has_match = retriever.retrieve(query)
        
        if not has_match:
            print("❌ NO CONFIDENT MATCH - Would escalate to human")
        else:
            print(f"✓ Found {len(results)} confident matches")
            for result in results[:3]:  # Show top 3
                print(f"\n  Rank {result.rank} | Score: {result.score:.3f}")
                print(f"  Doc: {result.doc_id}")
                print(f"  Text: {result.text[:150]}...")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Initialize retriever (loads from disk)
    retriever = Retriever()
    
    # Test queries
    test_queries = [
        "How do I reset my password?",
        "What does the premium plan cost?",
        "Can you tell me a joke?",  # Should fail (no relevant docs)
        "What are the supported payment methods?"
    ]
    
    # Analyze retrieval
    analyze_retrieval_quality(retriever, test_queries)
    
    # Example: Get formatted context for LLM
    query = "How do I reset my password?"
    context, results, has_match = retriever.retrieve_and_format(query)
    
    print("\n" + "="*80)
    print("FORMATTED CONTEXT FOR LLM:")
    print("="*80)
    print(context)