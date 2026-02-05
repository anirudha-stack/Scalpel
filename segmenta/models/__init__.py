"""Data models for Segmenta."""

from segmenta.models.enums import ElementType, Intent, BoundaryVerdict
from segmenta.models.document import Document, Section, Paragraph
from segmenta.models.chunk import Chunk, ChunkMetadata
from segmenta.models.boundary import BoundaryProposal, BoundaryDecision
from segmenta.models.result import SegmentaResult

__all__ = [
    "ElementType",
    "Intent",
    "BoundaryVerdict",
    "Document",
    "Section",
    "Paragraph",
    "Chunk",
    "ChunkMetadata",
    "BoundaryProposal",
    "BoundaryDecision",
    "SegmentaResult",
]
