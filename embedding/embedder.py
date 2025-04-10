from sentence_transformers import SentenceTransformer
from utils.logger import log_tokens_per_second

# Load once globally if needed
model = SentenceTransformer('all-MiniLM-L6-v2')


def generate_embeddings(chunks):
    # Extract text from the chunks
    texts = [chunk['text'] for chunk in chunks]

    # Extract metadata with fallback for missing keys
    metadata = [{
        "file": chunk.get("file", "Unknown"),  # Use .get to avoid KeyError
        "page": chunk.get("page", "Unknown"),  # Default to "Unknown" if 'page' is missing
        "chunk": chunk.get("chunk", "Unknown")  # Default to "Unknown" if 'chunk' is missing
    } for chunk in chunks]

    try:
        # Encode the texts into embeddings
        with log_tokens_per_second("🔢 Embedding"):
            vectors = model.encode(texts, show_progress_bar=True)
    except Exception as e:
        # Return error message in case of failure
        return f"❌ Error during embedding: {e}"

    # Package the embeddings with their metadata and texts
    embedded_chunks = []
    for vector, meta, text in zip(vectors, metadata, texts):
        embedded_chunks.append({
            "embedding": vector,
            "metadata": meta,
            "text": text
        })

    return embedded_chunks
