"""Boundary detection stage - uses embeddings to propose chunk boundaries."""

import time
from typing import List

from segmenta.pipeline.base import PipelineStage
from segmenta.pipeline.context import PipelineContext
from segmenta.embeddings.base import EmbeddingProvider
from segmenta.embeddings.similarity import (
    compute_adjacent_similarities,
    find_boundary_candidates,
    compute_similarity_statistics,
)
from segmenta.models import BoundaryProposal, Paragraph


class BoundaryDetectStage(PipelineStage):
    """Stage that detects chunk boundaries using embedding similarity."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        threshold: float = 0.5,
    ) -> None:
        """Initialize the boundary detection stage.

        Args:
            embedding_provider: Provider for generating embeddings
            threshold: Similarity threshold for boundary detection
        """
        self._embedding_provider = embedding_provider
        self._threshold = threshold

    @property
    def name(self) -> str:
        return "boundary_detect"

    def should_skip(self, context: PipelineContext) -> bool:
        """Skip for very short documents."""
        return context.skip_boundary_detection

    def process(self, context: PipelineContext) -> PipelineContext:
        """Detect boundaries using embedding similarity.

        Args:
            context: Pipeline context

        Returns:
            Updated context with boundary proposals
        """
        start_time = time.time()

        paragraphs = context.paragraphs
        if len(paragraphs) < 2:
            context.add_metric("boundary_detect_time", time.time() - start_time)
            return context

        # Get texts from splittable paragraphs
        splittable_paragraphs: List[Paragraph] = [
            p for p in paragraphs if p.is_splittable
        ]

        if len(splittable_paragraphs) < 2:
            # Not enough splittable paragraphs
            context.add_metric("boundary_detect_time", time.time() - start_time)
            return context

        texts = [p.text for p in splittable_paragraphs]

        # Compute embeddings
        embeddings = self._embedding_provider.embed(texts)

        # Compute similarities between adjacent paragraphs
        similarities = compute_adjacent_similarities(embeddings)

        # Record similarity statistics
        stats = compute_similarity_statistics(similarities)
        context.add_metric("similarity_stats", stats)

        # Use configured threshold
        threshold = self._threshold

        # Find boundary candidates
        candidates = find_boundary_candidates(similarities, threshold)

        # Create boundary proposals
        for position, similarity_score in candidates:
            # Map position back to original paragraph indices
            para_before = splittable_paragraphs[position - 1]
            para_after = splittable_paragraphs[position]

            proposal = BoundaryProposal(
                position=para_after.index,  # Use original paragraph index
                similarity_score=similarity_score,
                paragraph_before=para_before,
                paragraph_after=para_after,
            )
            context.boundary_proposals.append(proposal)

        # Record metrics
        context.add_metric("boundary_detect_time", time.time() - start_time)
        context.add_metric("boundary_proposals_count", len(context.boundary_proposals))
        context.add_metric("similarity_threshold", threshold)

        return context
