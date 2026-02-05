"""Result models for Segmenta."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from segmenta.models.chunk import Chunk


@dataclass
class SegmentaResult:
    """Result of a chunking operation."""

    success: bool
    output_path: Optional[str]
    chunks: List[Chunk]
    metrics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def chunk_count(self) -> int:
        """Get the number of chunks."""
        return len(self.chunks)

    @property
    def total_tokens(self) -> int:
        """Get total token count across all chunks."""
        return sum(c.metadata.token_count for c in self.chunks if c.metadata)

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0

    @property
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return len(self.errors) > 0

    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)

    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "output_path": self.output_path,
            "chunk_count": self.chunk_count,
            "total_tokens": self.total_tokens,
            "metrics": self.metrics,
            "warnings": self.warnings,
            "errors": self.errors,
        }
