"""Markdown output formatter."""

from typing import List
from pathlib import Path

import yaml

from scalpel.output.base import OutputFormatter
from scalpel.models import Chunk


class MarkdownFormatter(OutputFormatter):
    """Output formatter producing Markdown files with YAML frontmatter."""

    def __init__(self, include_separator: bool = True) -> None:
        """Initialize the Markdown formatter.

        Args:
            include_separator: Whether to include separators between chunks
        """
        self._include_separator = include_separator

    def format(self, chunks: List[Chunk], output_path: str) -> str:
        """Format chunks and write to output file.

        Args:
            chunks: List of chunks to format
            output_path: Path to write output to

        Returns:
            Path to the written output file
        """
        output = []

        for i, chunk in enumerate(chunks):
            formatted = self.format_chunk(chunk)
            output.append(formatted)

            # Add separator between chunks (not after the last one)
            if self._include_separator and i < len(chunks) - 1:
                output.append("")  # Empty line before next chunk

        content = "\n".join(output)

        # Write to file
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

        return str(path)

    def format_chunk(self, chunk: Chunk) -> str:
        """Format a single chunk with YAML frontmatter.

        Args:
            chunk: Chunk to format

        Returns:
            Formatted string representation of the chunk
        """
        parts = []

        # Add YAML frontmatter if metadata exists
        if chunk.metadata:
            metadata_dict = chunk.metadata.to_yaml_dict()
            yaml_content = yaml.dump(
                metadata_dict,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )
            parts.append("---")
            parts.append(yaml_content.rstrip())
            parts.append("---")
            parts.append("")  # Empty line after frontmatter

        # Add content
        parts.append(chunk.content)

        return "\n".join(parts)

    def format_to_string(self, chunks: List[Chunk]) -> str:
        """Format chunks to a string without writing to file.

        Args:
            chunks: List of chunks to format

        Returns:
            Formatted string
        """
        output = []

        for i, chunk in enumerate(chunks):
            formatted = self.format_chunk(chunk)
            output.append(formatted)

            if self._include_separator and i < len(chunks) - 1:
                output.append("")

        return "\n".join(output)
