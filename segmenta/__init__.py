"""
Segmenta: Semantic Document Chunking Library

Transform unstructured documents into semantically coherent, metadata-enriched chunks.
"""

from segmenta.config import SegmentaConfig
from segmenta.segmenta import Segmenta, SegmentaBuilder
from segmenta.models import (
    Document,
    Section,
    Paragraph,
    Chunk,
    ChunkMetadata,
    SegmentaResult,
    Intent,
    ElementType,
    BoundaryVerdict,
)
from segmenta.exceptions import (
    SegmentaError,
    SegmentaConfigError,
    SegmentaParseError,
    SegmentaLLMError,
    SegmentaEmbeddingError,
    UnsupportedFileTypeError,
)

__version__ = "1.0.0"
__all__ = [
    # Main classes
    "Segmenta",
    "SegmentaBuilder",
    "SegmentaConfig",
    # Models
    "Document",
    "Section",
    "Paragraph",
    "Chunk",
    "ChunkMetadata",
    "SegmentaResult",
    # Enums
    "Intent",
    "ElementType",
    "BoundaryVerdict",
    # Exceptions
    "SegmentaError",
    "SegmentaConfigError",
    "SegmentaParseError",
    "SegmentaLLMError",
    "SegmentaEmbeddingError",
    "UnsupportedFileTypeError",
]
