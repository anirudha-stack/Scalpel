"""Sentence Transformer embedding provider."""

from typing import List, Optional

import numpy as np

from microtome.embeddings.base import EmbeddingProvider
from microtome.exceptions import MicrotomeEmbeddingError


class SentenceTransformerProvider(EmbeddingProvider):
    """Embedding provider using Sentence Transformers."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        batch_size: int = 32,
    ) -> None:
        """Initialize the Sentence Transformer provider.

        Args:
            model_name: Name of the Sentence Transformer model
            device: Device to run model on ('cuda', 'cpu', or None for auto)
            batch_size: Batch size for encoding
        """
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._model = None  # Lazy loading
        self._embedding_dim: Optional[int] = None

    @property
    def model(self):
        """Lazy load the model on first use."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(
                    self._model_name, device=self._device
                )
                self._embedding_dim = self._model.get_sentence_embedding_dimension()
            except ImportError:
                raise MicrotomeEmbeddingError(
                    "sentence-transformers is not installed. "
                    "Install it with: pip install sentence-transformers",
                    model_name=self._model_name,
                )
            except Exception as e:
                raise MicrotomeEmbeddingError(
                    f"Failed to load embedding model '{self._model_name}': {e}",
                    model_name=self._model_name,
                )
        return self._model

    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([])

        try:
            embeddings = self.model.encode(
                texts,
                batch_size=self._batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return embeddings
        except Exception as e:
            raise MicrotomeEmbeddingError(
                f"Failed to generate embeddings: {e}",
                model_name=self._model_name,
            )

    @property
    def embedding_dim(self) -> int:
        """Return the dimensionality of embeddings."""
        if self._embedding_dim is None:
            # Force model loading to get dimension
            _ = self.model
        return self._embedding_dim or 384  # Default for MiniLM

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name
