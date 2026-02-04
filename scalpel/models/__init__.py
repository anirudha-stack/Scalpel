"""Data models for Scalpel."""

from scalpel.models.enums import ElementType, Intent, BoundaryVerdict
from scalpel.models.document import Document, Section, Paragraph
from scalpel.models.chunk import Chunk, ChunkMetadata
from scalpel.models.boundary import BoundaryProposal, BoundaryDecision
from scalpel.models.result import ScalpelResult

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
    "ScalpelResult",
]
