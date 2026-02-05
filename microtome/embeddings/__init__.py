"""Embedding providers for Microtome."""

from microtome.embeddings.base import EmbeddingProvider
from microtome.embeddings.sentence_transformer import SentenceTransformerProvider
from microtome.embeddings.similarity import compute_adjacent_similarities

__all__ = [
    "EmbeddingProvider",
    "SentenceTransformerProvider",
    "compute_adjacent_similarities",
]
