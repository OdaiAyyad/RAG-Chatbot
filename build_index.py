"""
Index Building Script

This script should be run ONCE to:
1. Load and preprocess your corpus
2. Generate embeddings
3. Build FAISS index
4. Save everything to disk

Run this before starting the Streamlit app.

Usage:
    python build_index.py

Or with custom corpus path:
    python build_index.py --corpus data/my_corpus.csv
"""

import argparse
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_processor import load_and_process_corpus, save_processed_chunks
from embedding_engine import build_index_from_chunks
from config import RAW_DATA_PATH, PROCESSED_DATA_PATH


def build_index(corpus_path: str = RAW_DATA_PATH):
    """
    Complete index building pipeline.
    
    Steps:
    1. Load raw corpus (CSV)
    2. Clean and chunk documents
    3. Generate embeddings
    4. Build FAISS index
    5. Save to disk
    
    Args:
        corpus_path: Path to raw corpus CSV file
    """
    print("\n" + "="*80)
    print("RAG INDEX BUILDER")
    print("="*80)
    
    # Step 1: Load and process corpus
    print("\n[Step 1/4] Loading and processing corpus...")
    try:
        chunks = load_and_process_corpus(corpus_path)
    except FileNotFoundError:
        print(f"\n❌ ERROR: Corpus file not found: {corpus_path}")
        print("\nPlease ensure your corpus CSV exists with columns:")
        print("  - text (required): Document text")
        print("  - doc_id (optional): Document identifier")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: Failed to process corpus: {e}")
        sys.exit(1)
    
    if not chunks:
        print("\n❌ ERROR: No chunks generated. Check your corpus file.")
        sys.exit(1)
    
    print(f"✓ Generated {len(chunks)} chunks")
    
    # Step 2: Save processed chunks (for future reference)
    print("\n[Step 2/4] Saving processed chunks...")
    save_processed_chunks(chunks, PROCESSED_DATA_PATH)
    print(f"✓ Saved to {PROCESSED_DATA_PATH}")
    
    # Step 3: Build embeddings and FAISS index
    print("\n[Step 3/4] Building embeddings and FAISS index...")
    print("This may take a few minutes depending on corpus size...")
    
    try:
        faiss_index = build_index_from_chunks(chunks)
    except Exception as e:
        print(f"\n❌ ERROR: Failed to build index: {e}")
        print("\nPossible causes:")
        print("  - Out of memory (try reducing corpus size)")
        print("  - Missing sentence-transformers library")
        print("  - GPU/CPU compatibility issues")
        sys.exit(1)
    
    print(f"✓ Index built with {faiss_index.index.ntotal} embeddings")
    
    # Step 4: Save FAISS index
    print("\n[Step 4/4] Saving FAISS index to disk...")
    try:
        faiss_index.save()
    except Exception as e:
        print(f"\n❌ ERROR: Failed to save index: {e}")
        sys.exit(1)
    
    print("✓ Index saved successfully")
    
    # Summary
    print("\n" + "="*80)
    print("✅ INDEX BUILD COMPLETE")
    print("="*80)
    print(f"\nStatistics:")
    print(f"  - Total documents: {len(set(c['doc_id'] for c in chunks))}")
    print(f"  - Total chunks: {len(chunks)}")
    print(f"  - Average chunk size: {sum(c['token_count'] for c in chunks) / len(chunks):.0f} tokens")
    print(f"  - Embedding dimension: {faiss_index.embedding_dim}")
    print(f"\nFiles created:")
    print(f"  - {PROCESSED_DATA_PATH}")
    print(f"  - {faiss_index.index}")
    print(f"\nYou can now run the Streamlit app:")
    print(f"  streamlit run app.py")
    print("="*80 + "\n")


def verify_setup():
    """
    Verify that all required files and dependencies exist.
    """
    print("\n[Verification] Checking setup...")
    
    # Check directories
    dirs_to_check = ['data', 'embeddings', 'src']
    for dir_name in dirs_to_check:
        if not os.path.exists(dir_name):
            print(f"⚠️  Directory missing: {dir_name} (will be created)")
            os.makedirs(dir_name, exist_ok=True)
    
    # Check Python dependencies
    try:
        import pandas
        import faiss
        import sentence_transformers
        import google.generativeai
        import streamlit
        print("✓ All Python dependencies installed")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("\nPlease run: pip install -r requirements.txt")
        sys.exit(1)
    
    # Check for API key
    if not os.getenv('GEMINI_API_KEY'):
        print("\n⚠️  WARNING: GEMINI_API_KEY environment variable not set")
        print("The app will not work without this. Set it using:")
        print("  export GEMINI_API_KEY='your_key_here'")
        print("\nContinuing anyway (you can set it later)...\n")
    else:
        print("✓ GEMINI_API_KEY is set")
    
    print("✓ Setup verification complete\n")


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Build FAISS index for RAG chatbot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python build_index.py
  python build_index.py --corpus data/my_corpus.csv
  python build_index.py --verify-only

For more information, see README.md
        """
    )
    
    parser.add_argument(
        '--corpus',
        type=str,
        default=RAW_DATA_PATH,
        help=f'Path to corpus CSV file (default: {RAW_DATA_PATH})'
    )
    
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify setup without building index'
    )
    
    args = parser.parse_args()
    
    # Verify setup
    verify_setup()
    
    if args.verify_only:
        print("✓ Verification complete. Exiting.")
        return
    
    # Build index
    build_index(args.corpus)


if __name__ == "__main__":
    main()