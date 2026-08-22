# src/config.py
# Single source of truth for shared constants used across the pipeline.
# Embedding model must stay identical between the vector-DB build step and the
# query-time store; diverging them silently breaks VectorSearchTool.
#
# Embeddings run through the local Ollama server (deterministic, offline) rather
# than downloading sentence-transformers weights from HuggingFace, which is
# unreachable in this network environment.
EMBEDDING_PROVIDER = "ollama"  # "ollama" | "huggingface"
EMBEDDING_MODEL = "nomic-embed-text"  # Ollama embedding model name
OLLAMA_URL = "http://localhost:11434"

CHROMA_DIR = "./vector_db"
CHROMA_COLLECTION = "product_catalog"

DATASET_DIR = "./dataset"


def build_embeddings():
    """Return the vector store embedding function (shared by index & query)."""
    if EMBEDDING_PROVIDER == "ollama":
        from langchain_ollama import OllamaEmbeddings

        # Omit base_url: langchain_ollama's Client(host=...) misroutes explicit
        # hosts; None uses the default localhost:11434 which is reliable here.
        return OllamaEmbeddings(model=EMBEDDING_MODEL)

    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )