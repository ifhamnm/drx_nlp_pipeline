import faiss
import numpy as np
import pickle
import os

DB_PATH = "vector_db/drx_index.faiss"
META_PATH = "vector_db/metadata.pkl"

# Caching in memory for faster access
faiss_index = None
metadata_store = None

def build_vector_db(embedded_chunks):
    """
    Build the FAISS vector DB with embedded chunks and save to disk.
    """
    embeddings = np.array([item['embedding'] for item in embedded_chunks]).astype("float32")
    metadata = [item['metadata'] | {"text": item['text']} for item in embedded_chunks]

    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(embeddings)

    os.makedirs("vector_db", exist_ok=True)
    faiss.write_index(index, DB_PATH)

    with open(META_PATH, "wb") as f:
        pickle.dump(metadata, f)

    # Cache the index and metadata
    global faiss_index, metadata_store
    faiss_index = index
    metadata_store = metadata


def load_vector_db():
    """
    Load FAISS index and metadata into memory if not already loaded.
    """
    global faiss_index, metadata_store

    if faiss_index is None or metadata_store is None:
        if not os.path.exists(DB_PATH) or not os.path.exists(META_PATH):
            raise FileNotFoundError("FAISS index or metadata not found. Run `build_vector_db()` first.")

        print("📥 Loading FAISS index and metadata into memory...")
        faiss_index = faiss.read_index(DB_PATH)
        with open(META_PATH, "rb") as f:
            metadata_store = pickle.load(f)
        print("✅ FAISS index and metadata loaded.")


def search_vector_db(query_embedding, top_k=5):
    """
    Perform similarity search using FAISS.
    """
    load_vector_db()  # Load once if not already loaded

    if faiss_index is None or metadata_store is None:
        raise RuntimeError("FAISS index or metadata is not loaded properly.")

    query = np.array([query_embedding]).astype("float32")
    D, I = faiss_index.search(query, top_k)
    results = [metadata_store[i] for i in I[0]]
    return results
