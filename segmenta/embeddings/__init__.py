"""Embedding providers for Segmenta."""

from segmenta.embeddings.base import EmbeddingProvider
from segmenta.embeddings.sentence_transformer import SentenceTransformerProvider
from segmenta.embeddings.similarity import compute_adjacent_similarities

__all__ = [
    "EmbeddingProvider",
    "SentenceTransformerProvider",
    "compute_adjacent_similarities",
]
