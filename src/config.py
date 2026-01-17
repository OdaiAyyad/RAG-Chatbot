"""
Configuration file for RAG chatbot system.

All hyperparameters and constants are defined here for easy tuning.
"""

# ============================================================================
# EMBEDDING SETTINGS
# ============================================================================

# Embedding model: We use sentence-transformers for high-quality embeddings
# This model creates 384-dimensional vectors (balance of quality vs speed)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Why this model?
# - Fast inference (important for real-time retrieval)
# - Good performance on semantic similarity tasks
# - Open-source (no API costs)
# - Alternative: "sentence-transformers/all-mpnet-base-v2" (better quality, slower)

# ============================================================================
# CHUNKING SETTINGS
# ============================================================================

# Maximum tokens per chunk
# Why 512? LLMs handle this well, and it captures complete thoughts
# Too small (e.g., 128) → Context is fragmented
# Too large (e.g., 2048) → Retrieval is less precise
CHUNK_SIZE = 512

# Overlap between consecutive chunks (in tokens)
# Why overlap? Ensures sentences split across chunks appear in both
# Example: "...end of chunk 1. Start of chunk 2..." 
# With overlap, "Start of chunk 2" appears in chunk 1 too
CHUNK_OVERLAP = 50

# ============================================================================
# RETRIEVAL SETTINGS
# ============================================================================

# Number of documents to retrieve from FAISS
# We retrieve more than we might use to have fallback options
TOP_K_RETRIEVAL = 5

# Minimum cosine similarity score to consider a match valid
# Scale: 0.0 (unrelated) to 1.0 (identical)
# 0.65 is conservative: "somewhat related" is our threshold
# Tune this based on your data:
#   - Medical/Legal: Use 0.75+ (strict)
#   - General FAQ: Use 0.60-0.70
#   - Creative content: Use 0.50-0.60
CONFIDENCE_THRESHOLD = 0.65

# ============================================================================
# LLM SETTINGS (Gemini)
# ============================================================================

# Gemini model version
# Options:
# - "gemini-pro": Fast, good for most tasks
# - "gemini-pro-vision": If you need image understanding
GEMINI_MODEL = "gemini-pro"

# Temperature: Controls randomness
# 0.0 = Deterministic (same input → same output)
# 1.0 = Creative/random
# For QA, we want LOW temperature to avoid hallucination
GEMINI_TEMPERATURE = 0.0

# Max output tokens
# Keep this reasonable to avoid rambling answers
GEMINI_MAX_TOKENS = 256

# ============================================================================
# SYSTEM PROMPT (The Most Important Part)
# ============================================================================

# This prompt is sent to Gemini with EVERY query
# It defines the chatbot's behavior and constraints
SYSTEM_PROMPT = """You are a helpful customer support assistant for a company.

STRICT RULES:
1. Answer ONLY using information from the provided context documents below.
2. If the context does not contain enough information to answer the question, respond EXACTLY with: "I cannot find an answer in the knowledge base. Your query will be escalated to a human agent."
3. Do NOT use external knowledge or make assumptions.
4. Always cite which document(s) you used (e.g., "According to Document 2...").
5. Keep answers concise and professional.
6. If multiple documents provide conflicting information, acknowledge this.

Context Documents:
{context}

Question: {question}

Answer:"""

# Why this prompt structure?
# - "STRICT RULES" → Grabs LLM's attention; these are hard constraints
# - "Answer ONLY using..." → Reduces hallucination
# - Exact fallback phrase → Ensures consistency when escalating
# - "Cite which document" → Enables verification
# - {context} and {question} → Placeholders we'll fill at runtime

# ============================================================================
# UI SETTINGS
# ============================================================================

# Streamlit page configuration
PAGE_TITLE = "RAG Customer Support Chatbot"
PAGE_ICON = "🤖"

# Message to show when confidence is too low
ESCALATION_MESSAGE = "I cannot find an answer in the knowledge base. Your query will be escalated to a human agent."

# ============================================================================
# FILE PATHS
# ============================================================================

import os

# Base directory (where this config.py file lives)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data paths
RAW_DATA_PATH = os.path.join(BASE_DIR, "..", "data", "raw_corpus.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "..", "data", "processed_chunks.json")

# Embedding paths
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "..", "embeddings", "faiss_index.bin")
CHUNK_METADATA_PATH = os.path.join(BASE_DIR, "..", "embeddings", "chunk_metadata.json")

# Create directories if they don't exist
os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)