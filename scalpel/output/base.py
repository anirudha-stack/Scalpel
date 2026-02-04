"""Abstract base class for output formatters."""

from abc import ABC, abstractmethod
from typing import List

from scalpel.models import Chunk


class OutputFormatter(ABC):
    """Abstract base class for output formatters."""

    @abstractmethod
    def format(self, chunks: List[Chunk], output_path: str) -> str:
        """Format chunks and write to output file.

        Args:
            chunks: List of chunks to format
            output_path: Path to write output to

        Returns:
            Path to the written output file
        """
        pass

    @abstractmethod
    def format_chunk(self, chunk: Chunk) -> str:
        """Format a single chunk.

        Args:
            chunk: Chunk to format

        Returns:
            Formatted string representation of the chunk
        """
        pass
