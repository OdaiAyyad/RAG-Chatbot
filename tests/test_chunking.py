from src.data_processor import DocumentChunker

def test_chunking():
    chunker = DocumentChunker(chunk_size=100, overlap=20)

    # Test case 1: Short document (should be 1 chunk)
    text = "This is a short document with only one sentence."
    chunks = chunker.chunk_document(text, "test_1")
    assert len(chunks) == 1, f"Expected 1 chunk, got {len(chunks)}"

    # Test case 2: Long document (should be multiple chunks)
    long_text = ". ".join([f"Sentence {i}" for i in range(100)])
    chunks = chunker.chunk_document(long_text, "test_2")
    print(f"Number of chunks: {len(chunks)}")
    print(f"First chunk text: {chunks[0]['text'][:50]}...")
    print(f"Second chunk text: {chunks[1]['text'][:50]}...")
    assert len(chunks) > 1, "Long document should create multiple chunks"

    # Test case 3: Overlap exists
    print(f"Checking overlap: {chunks[1]['text'][:20]} in first chunk?")
    assert chunks[1]['text'][:20] in chunks[0]['text'], "Chunks should overlap"

    print("✓ All chunking tests passed")

if __name__ == "__main__":
    test_chunking()