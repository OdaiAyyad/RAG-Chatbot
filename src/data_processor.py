"""
Data Processing Module

This module handles:
1. Loading raw corpus data
2. Cleaning and normalizing text
3. Chunking documents into optimal sizes
4. Preserving metadata (doc_id, source, etc.)

Key Concepts:
- Chunking Strategy: We use overlapping windows to preserve context
- Token Counting: Approximate via word count (1 token ≈ 0.75 words in English)
"""

import pandas as pd
import re
import json
from typing import List, Dict
from config import (
    CHUNK_SIZE, 
    CHUNK_OVERLAP, 
    RAW_DATA_PATH, 
    PROCESSED_DATA_PATH
)


class DocumentChunker:
    """
    Handles intelligent document chunking with overlap.
    
    Why overlap?
    - Prevents splitting mid-sentence or mid-concept
    - Ensures queries matching chunk boundaries find relevant info
    - Example: Query about "user authentication" might match better 
      with overlap if that phrase spans two chunks
    """
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
        """
        Args:
            chunk_size: Maximum tokens per chunk (roughly words * 1.3)
            overlap: Number of tokens to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def estimate_tokens(self, text: str) -> int:
        """
        Rough token count estimation.
        
        Why not exact? 
        - Tokenization varies by model (GPT uses BPE, others use WordPiece)
        - For chunking, approximate is fine (we're not hitting limits)
        - Exact counting would require loading the tokenizer (slow)
        
        Heuristic: 1 token ≈ 4 characters for English
        This slightly overestimates, which is safer for chunk size limits
        """
        return len(text) // 4
    
    def clean_text(self, text: str) -> str:
        """
        Normalize text for consistent embedding quality.
        
        Steps:
        1. Remove excessive whitespace
        2. Fix common encoding issues
        3. Remove special characters that don't add meaning
        
        What NOT to remove:
        - Punctuation (helps LLM understand structure)
        - Numbers (often critical in QA)
        - Standard capitalization (preserves named entities)
        """
        # Fix common encoding issues (e.g., "don't" became "donâ€™t")
        text = text.encode('utf-8', 'ignore').decode('utf-8')
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove control characters but keep newlines as periods
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        text = text.replace('\n', '. ')
        
        # Remove excessive punctuation (e.g., "!!!" → "!")
        text = re.sub(r'([!?.])\1+', r'\1', text)
        
        return text.strip()
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using regex.
        
        Why sentence-level splitting?
        - More semantic than arbitrary character splits
        - Prevents mid-sentence breaks
        - Works better with overlap strategy
        
        Regex explanation:
        - (?<=[.!?]) → After punctuation
        - \\s+ → Followed by whitespace
        - (?=[A-Z]) → Before capital letter (next sentence)
        """
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def chunk_document(self, text: str, doc_id: str) -> List[Dict]:
        """
        Split document into overlapping chunks.
        
        Algorithm:
        1. Clean the text
        2. Split into sentences
        3. Combine sentences into chunks up to chunk_size
        4. Add overlap by including last N tokens from previous chunk
        
        Returns:
            List of dicts with structure:
            {
                'text': str,           # Chunk content
                'doc_id': str,         # Original document ID
                'chunk_id': int,       # Chunk number within document
                'token_count': int     # Estimated tokens
            }
        """
        text = self.clean_text(text)
        sentences = self.split_into_sentences(text)
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        overlap_sentences = []
        
        for sentence in sentences:
            sentence_tokens = self.estimate_tokens(sentence)
            
            # If adding this sentence exceeds chunk_size, finalize current chunk
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'doc_id': doc_id,
                    'chunk_id': len(chunks),
                    'token_count': current_tokens
                })
                
                # Prepare overlap for next chunk
                # Keep last few sentences that fit in overlap window
                overlap_tokens = 0
                overlap_sentences = []
                for sent in reversed(current_chunk):
                    sent_tokens = self.estimate_tokens(sent)
                    if overlap_tokens + sent_tokens <= self.overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_tokens += sent_tokens
                    else:
                        break
                
                # Start new chunk with overlap
                current_chunk = overlap_sentences.copy()
                current_tokens = overlap_tokens
            
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'doc_id': doc_id,
                'chunk_id': len(chunks),
                'token_count': current_tokens
            })
        
        return chunks


def load_and_process_corpus(input_path: str = RAW_DATA_PATH) -> List[Dict]:
    """
    Load corpus and process into chunks.
    
    Expected CSV format:
        doc_id,text,metadata (optional)
    
    Why CSV?
    - Simple and universal
    - Easy to edit/inspect
    - Can load in Excel/Google Sheets
    
    Args:
        input_path: Path to raw corpus CSV
    
    Returns:
        List of processed chunks with metadata
    """
    print(f"Loading corpus from: {input_path}")
    
    # Load CSV
    # Note: We use 'utf-8' encoding and handle errors gracefully
    try:
        df = pd.read_csv(input_path, encoding='utf-8')
    except UnicodeDecodeError:
        # Fallback for files with mixed encoding
        df = pd.read_csv(input_path, encoding='latin-1')
    
    print(f"Loaded {len(df)} documents")
    
    # Validate required columns
    if 'text' not in df.columns:
        raise ValueError("CSV must have a 'text' column")
    
    # If no doc_id, generate one
    if 'doc_id' not in df.columns:
        df['doc_id'] = [f"doc_{i}" for i in range(len(df))]
    
    # Remove duplicates (based on text content)
    original_count = len(df)
    df = df.drop_duplicates(subset=['text'], keep='first')
    if len(df) < original_count:
        print(f"Removed {original_count - len(df)} duplicate documents")
    
    # Remove empty documents
    df = df[df['text'].notna() & (df['text'].str.strip() != '')]
    print(f"Processing {len(df)} non-empty documents")
    
    # Initialize chunker
    chunker = DocumentChunker()
    
    # Process each document
    all_chunks = []
    for idx, row in df.iterrows():
        doc_id = row['doc_id']
        text = row['text']
        
        # Chunk the document
        chunks = chunker.chunk_document(text, doc_id)
        all_chunks.extend(chunks)
        
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(df)} documents...")
    
    print(f"\nProcessing complete!")
    print(f"Total documents: {len(df)}")
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Avg chunks per document: {len(all_chunks) / len(df):.2f}")
    
    return all_chunks


def save_processed_chunks(chunks: List[Dict], output_path: str = PROCESSED_DATA_PATH):
    """
    Save processed chunks to JSON.
    
    Why JSON instead of CSV?
    - Preserves nested structure better
    - Easier to load programmatically
    - Human-readable for debugging
    """
    print(f"Saving {len(chunks)} chunks to: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    
    print("Save complete!")


def load_processed_chunks(input_path: str = PROCESSED_DATA_PATH) -> List[Dict]:
    """Load previously processed chunks from JSON."""
    print(f"Loading processed chunks from: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f"Loaded {len(chunks)} chunks")
    return chunks


# ============================================================================
# EXAMPLE USAGE (for testing this module standalone)
# ============================================================================

if __name__ == "__main__":
    # Example: Create sample corpus
    sample_data = {
        'doc_id': ['doc_1', 'doc_2', 'doc_3'],
        'text': [
            "To reset your password, go to Settings > Security > Password. Click 'Forgot Password' and follow the email instructions. You'll receive a reset link within 5 minutes.",
            "Our premium plan costs $29/month and includes unlimited storage, priority support, and advanced analytics. The basic plan is $9/month with 10GB storage.",
            "If you experience login issues, first clear your browser cache. Then try disabling browser extensions. If the problem persists, contact support@company.com with your username."
        ]
    }
    
    # Save sample CSV
    import os
    os.makedirs('data', exist_ok=True)
    df = pd.DataFrame(sample_data)
    df.to_csv('data/raw_corpus.csv', index=False)
    
    # Process it
    chunks = load_and_process_corpus('data/raw_corpus.csv')
    save_processed_chunks(chunks, 'data/processed_chunks.json')
    
    # Inspect first chunk
    print("\nFirst chunk:")
    print(json.dumps(chunks[0], indent=2))