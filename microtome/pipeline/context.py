"""Pipeline context for carrying state through stages."""

from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict

from microtome.models import (
    Document,
    Paragraph,
    Chunk,
    BoundaryProposal,
    BoundaryDecision,
)
from microtome.config import MicrotomeConfig


@dataclass
class PipelineContext:
    """Carries state through the pipeline stages."""

    # Input
    input_path: str
    output_dir: str
    config: MicrotomeConfig

    # Stage outputs (populated as pipeline progresses)
    document: Optional[Document] = None
    paragraphs: List[Paragraph] = field(default_factory=list)
    boundary_proposals: List[BoundaryProposal] = field(default_factory=list)
    boundary_decisions: List[BoundaryDecision] = field(default_factory=list)
    chunks: List[Chunk] = field(default_factory=list)
    output_path: Optional[str] = None

    # Control flow
    should_stop: bool = False
    skip_boundary_detection: bool = False  # For short documents
    dry_run: bool = False  # Skip LLM calls

    # Adaptive settings (populated by planning stages)
    atomize_sentences_per_paragraph_override: Optional[int] = None
    atomize_min_sentences_override: Optional[int] = None

    # Metrics
    metrics: Dict[str, Any] = field(default_factory=dict)

    # Error tracking
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)

    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)

    def add_metric(self, key: str, value: Any) -> None:
        """Add or update a metric."""
        self.metrics[key] = value

    def increment_metric(self, key: str, amount: int = 1) -> None:
        """Increment a numeric metric."""
        current = self.metrics.get(key, 0)
        self.metrics[key] = current + amount

    @property
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0

    @property
    def paragraph_count(self) -> int:
        """Get the number of paragraphs."""
        return len(self.paragraphs)

    @property
    def chunk_count(self) -> int:
        """Get the number of chunks."""
        return len(self.chunks)
