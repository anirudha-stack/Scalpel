"""Abstract base class for embedding providers."""

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        pass

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the dimensionality of embeddings."""
        pass

    def embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.

        Args:
            text: Text string to embed

        Returns:
            numpy array of shape (embedding_dim,)
        """
        result = self.embed([text])
        return result[0]
