"""
Scalpel: Semantic Document Chunking Library

Transform unstructured documents into semantically coherent, metadata-enriched chunks.
"""

from scalpel.config import ScalpelConfig
from scalpel.scalpel import Scalpel, ScalpelBuilder
from scalpel.models import (
    Document,
    Section,
    Paragraph,
    Chunk,
    ChunkMetadata,
    ScalpelResult,
    Intent,
    ElementType,
    BoundaryVerdict,
)
from scalpel.exceptions import (
    ScalpelError,
    ScalpelConfigError,
    ScalpelParseError,
    ScalpelLLMError,
    ScalpelEmbeddingError,
    UnsupportedFileTypeError,
)

__version__ = "1.0.0"
__all__ = [
    # Main classes
    "Scalpel",
    "ScalpelBuilder",
    "ScalpelConfig",
    # Models
    "Document",
    "Section",
    "Paragraph",
    "Chunk",
    "ChunkMetadata",
    "ScalpelResult",
    # Enums
    "Intent",
    "ElementType",
    "BoundaryVerdict",
    # Exceptions
    "ScalpelError",
    "ScalpelConfigError",
    "ScalpelParseError",
    "ScalpelLLMError",
    "ScalpelEmbeddingError",
    "UnsupportedFileTypeError",
]
