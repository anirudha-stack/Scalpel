"""Boundary models for Scalpel."""

from dataclasses import dataclass
from typing import Optional

from scalpel.models.document import Paragraph
from scalpel.models.enums import BoundaryVerdict


@dataclass
class BoundaryProposal:
    """A proposed chunk boundary from Stage 1 (embedding similarity)."""

    position: int  # Index in paragraph list where boundary should occur
    similarity_score: float  # Cosine similarity between adjacent paragraphs
    paragraph_before: Paragraph
    paragraph_after: Paragraph

    @property
    def context_window(self) -> str:
        """Get text context around the boundary for LLM validation."""
        return (
            f"END OF CHUNK:\n{self.paragraph_before.text}\n\n"
            f"START OF NEW CHUNK:\n{self.paragraph_after.text}"
        )

    def get_text_before(self, max_chars: int = 500) -> str:
        """Get text before the boundary, truncated if needed."""
        text = self.paragraph_before.text
        if len(text) > max_chars:
            return "..." + text[-max_chars:]
        return text

    def get_text_after(self, max_chars: int = 500) -> str:
        """Get text after the boundary, truncated if needed."""
        text = self.paragraph_after.text
        if len(text) > max_chars:
            return text[:max_chars] + "..."
        return text


@dataclass
class BoundaryDecision:
    """LLM decision on a boundary proposal from Stage 2."""

    proposal: BoundaryProposal
    verdict: BoundaryVerdict
    reason: str
    adjusted_position: Optional[int] = None  # Only if verdict is ADJUST
    confidence: float = 1.0

    @property
    def final_position(self) -> Optional[int]:
        """Get the final boundary position after decision."""
        if self.verdict == BoundaryVerdict.MERGE:
            return None
        if self.verdict == BoundaryVerdict.ADJUST and self.adjusted_position is not None:
            return self.adjusted_position
        return self.proposal.position

    @property
    def should_split(self) -> bool:
        """Check if this decision results in a split."""
        return self.verdict != BoundaryVerdict.MERGE
