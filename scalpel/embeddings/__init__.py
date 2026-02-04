"""Embedding providers for Scalpel."""

from scalpel.embeddings.base import EmbeddingProvider
from scalpel.embeddings.sentence_transformer import SentenceTransformerProvider
from scalpel.embeddings.similarity import compute_adjacent_similarities

__all__ = [
    "EmbeddingProvider",
    "SentenceTransformerProvider",
    "compute_adjacent_similarities",
]
