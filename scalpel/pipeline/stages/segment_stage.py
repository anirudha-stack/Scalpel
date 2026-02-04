"""Segment stage - extracts paragraphs from document."""

import time

from scalpel.pipeline.base import PipelineStage
from scalpel.pipeline.context import PipelineContext


class SegmentStage(PipelineStage):
    """Stage that extracts and segments paragraphs from the document."""

    @property
    def name(self) -> str:
        return "segment"

    def process(self, context: PipelineContext) -> PipelineContext:
        """Extract paragraphs from the document.

        Args:
            context: Pipeline context

        Returns:
            Updated context with paragraphs
        """
        start_time = time.time()

        if context.document is None:
            context.add_error("No document to segment")
            context.should_stop = True
            return context

        # Extract all paragraphs from document
        paragraphs = context.document.all_paragraphs
        context.paragraphs = paragraphs

        # Check for short document
        paragraph_count = len(paragraphs)
        if paragraph_count < context.config.short_document_threshold:
            context.skip_boundary_detection = True
            context.add_warning(
                f"Short document ({paragraph_count} paragraphs). "
                "Returning as single chunk."
            )

        # Check for empty document
        if paragraph_count == 0:
            context.add_warning("Document is empty, no paragraphs found.")
            context.skip_boundary_detection = True

        # Record metrics
        context.add_metric("segment_time", time.time() - start_time)
        context.add_metric("paragraph_count", paragraph_count)
        context.add_metric(
            "atomic_element_count",
            sum(1 for p in paragraphs if p.is_atomic),
        )

        return context
