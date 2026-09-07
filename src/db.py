"""Shared database client and embedding model.

Environment flags are set before importing chromadb and torch, since both
read them at import time.
"""
import os

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "False")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from config import DB_PATH, COLLECTION_NAME, EMBED_MODEL

_model: SentenceTransformer | None = None
_client = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=str(DB_PATH),
            settings=Settings(anonymized_telemetry=False),
        )
    return _client


def get_collection():
    return get_client().get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def embed(texts: list[str], batch_size: int = 64, progress: bool = False):
    return get_model().encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=progress,
        normalize_embeddings=True,
    ).tolist()
