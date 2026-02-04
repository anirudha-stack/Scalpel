"""Base classes for pipeline components."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scalpel.pipeline.context import PipelineContext


class PipelineStage(ABC):
    """Base class for all pipeline stages."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stage name for logging and metrics."""
        pass

    @abstractmethod
    def process(self, context: "PipelineContext") -> "PipelineContext":
        """Process the context and return updated context.

        Args:
            context: Pipeline context with current state

        Returns:
            Updated pipeline context
        """
        pass

    def should_skip(self, context: "PipelineContext") -> bool:
        """Override to conditionally skip this stage.

        Args:
            context: Pipeline context

        Returns:
            True if this stage should be skipped
        """
        return False
