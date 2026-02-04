"""Chunk formation stage - forms chunks from paragraphs and boundary decisions."""

import time
from typing import List, Set

from scalpel.pipeline.base import PipelineStage
from scalpel.pipeline.context import PipelineContext
from scalpel.models import Chunk, Paragraph, BoundaryVerdict


class ChunkFormStage(PipelineStage):
    """Stage that forms chunks based on boundary decisions."""

    @property
    def name(self) -> str:
        return "chunk_form"

    def process(self, context: PipelineContext) -> PipelineContext:
        """Form chunks from paragraphs using boundary decisions.

        Args:
            context: Pipeline context

        Returns:
            Updated context with chunks
        """
        start_time = time.time()

        paragraphs = context.paragraphs
        if not paragraphs:
            context.add_metric("chunk_form_time", time.time() - start_time)
            return context

        # Get final boundary positions
        boundary_positions: Set[int] = set()

        # If we skipped boundary detection, no internal boundaries
        if not context.skip_boundary_detection:
            for decision in context.boundary_decisions:
                if decision.should_split:
                    pos = decision.final_position
                    if pos is not None:
                        boundary_positions.add(pos)

        # Form chunks by grouping paragraphs
        chunks = self._form_chunks(paragraphs, boundary_positions)
        context.chunks = chunks

        # Record metrics
        context.add_metric("chunk_form_time", time.time() - start_time)
        context.add_metric("chunks_formed", len(chunks))

        return context

    def _form_chunks(
        self,
        paragraphs: List[Paragraph],
        boundary_positions: Set[int],
    ) -> List[Chunk]:
        """Form chunks by grouping paragraphs at boundary positions.

        Args:
            paragraphs: List of all paragraphs
            boundary_positions: Set of paragraph indices where chunks should split

        Returns:
            List of formed chunks
        """
        if not paragraphs:
            return []

        chunks: List[Chunk] = []
        current_paragraphs: List[Paragraph] = []

        for paragraph in paragraphs:
            # Check if we should start a new chunk
            if paragraph.index in boundary_positions and current_paragraphs:
                # Create chunk from current paragraphs
                chunk = self._create_chunk(current_paragraphs)
                chunks.append(chunk)
                current_paragraphs = []

            current_paragraphs.append(paragraph)

        # Don't forget the last chunk
        if current_paragraphs:
            chunk = self._create_chunk(current_paragraphs)
            chunks.append(chunk)

        return chunks

    def _create_chunk(self, paragraphs: List[Paragraph]) -> Chunk:
        """Create a chunk from a list of paragraphs.

        Args:
            paragraphs: Paragraphs to include in the chunk

        Returns:
            Created chunk
        """
        # Combine paragraph texts
        content_parts = []
        source_indices = []

        for para in paragraphs:
            content_parts.append(para.text)
            source_indices.append(para.index)

        # Join with appropriate separators
        content = "\n\n".join(content_parts)

        return Chunk(
            content=content,
            metadata=None,  # Will be filled by enrichment stage
            source_paragraphs=source_indices,
        )
