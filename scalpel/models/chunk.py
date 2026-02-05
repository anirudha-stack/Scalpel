"""Chunk models for Scalpel."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from scalpel.models.enums import Intent


@dataclass
class ChunkMetadata:
    """Metadata for a document chunk."""

    chunk_id: str
    title: str
    summary: str
    intent: Intent
    keywords: List[str]
    parent_section: str
    token_count: int
    questions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "chunk_id": self.chunk_id,
            "title": self.title,
            "summary": self.summary,
            "intent": self.intent.value,
            "questions": self.questions,
            "keywords": self.keywords,
            "parent_section": self.parent_section,
            "token_count": self.token_count,
        }

    def to_yaml_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        return self.to_dict()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkMetadata":
        """Create ChunkMetadata from dictionary."""
        return cls(
            chunk_id=data.get("chunk_id", ""),
            title=data.get("title", ""),
            summary=data.get("summary", ""),
            intent=Intent.from_string(data.get("intent", "unknown")),
            questions=data.get("questions", []),
            keywords=data.get("keywords", []),
            parent_section=data.get("parent_section", ""),
            token_count=data.get("token_count", 0),
        )


@dataclass
class Chunk:
    """A semantically coherent chunk of document content."""

    content: str
    metadata: Optional[ChunkMetadata] = None
    source_paragraphs: List[int] = field(default_factory=list)

    @property
    def is_enriched(self) -> bool:
        """Check if this chunk has metadata."""
        return self.metadata is not None

    @property
    def token_count(self) -> int:
        """Get token count from metadata or 0."""
        if self.metadata:
            return self.metadata.token_count
        return 0

    def __len__(self) -> int:
        """Return the length of the content."""
        return len(self.content)
