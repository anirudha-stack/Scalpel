"""
Microtome: Semantic Document Chunking Library

Transform unstructured documents into semantically coherent, metadata-enriched chunks.
"""

from microtome.config import MicrotomeConfig
from microtome.microtome import Microtome, MicrotomeBuilder
from microtome.models import (
    Document,
    Section,
    Paragraph,
    Chunk,
    ChunkMetadata,
    MicrotomeResult,
    Intent,
    ElementType,
    BoundaryVerdict,
)
from microtome.exceptions import (
    MicrotomeError,
    MicrotomeConfigError,
    MicrotomeParseError,
    MicrotomeLLMError,
    MicrotomeEmbeddingError,
    UnsupportedFileTypeError,
)

__version__ = "1.0.0"
__all__ = [
    # Main classes
    "Microtome",
    "MicrotomeBuilder",
    "MicrotomeConfig",
    # Models
    "Document",
    "Section",
    "Paragraph",
    "Chunk",
    "ChunkMetadata",
    "MicrotomeResult",
    # Enums
    "Intent",
    "ElementType",
    "BoundaryVerdict",
    # Exceptions
    "MicrotomeError",
    "MicrotomeConfigError",
    "MicrotomeParseError",
    "MicrotomeLLMError",
    "MicrotomeEmbeddingError",
    "UnsupportedFileTypeError",
]
