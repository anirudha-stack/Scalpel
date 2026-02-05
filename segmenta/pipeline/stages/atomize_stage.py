"""Atomize stage - splits long paragraphs into smaller sentence groups.

This improves boundary detection on documents where paragraphs contain multiple
subtopics (common in PDFs).
"""

from __future__ import annotations

import logging
import re
import time
from typing import List

from segmenta.pipeline.base import PipelineStage
from segmenta.pipeline.context import PipelineContext
from segmenta.models import Paragraph, ElementType

logger = logging.getLogger(__name__)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


class AtomizeStage(PipelineStage):
    """Stage that atomizes paragraphs into smaller sentence groups."""

    @property
    def name(self) -> str:
        return "atomize"

    def should_skip(self, context: PipelineContext) -> bool:
        planned = context.atomize_sentences_per_paragraph_override
        if planned is not None:
            return planned <= 0
        return context.config.atomize_sentences_per_paragraph <= 0

    def process(self, context: PipelineContext) -> PipelineContext:
        start_time = time.time()

        paragraphs = context.paragraphs
        if not paragraphs:
            context.add_metric("atomize_time", time.time() - start_time)
            return context

        group_size = (
            context.atomize_sentences_per_paragraph_override
            if context.atomize_sentences_per_paragraph_override is not None
            else context.config.atomize_sentences_per_paragraph
        )
        min_sentences = (
            context.atomize_min_sentences_override
            if context.atomize_min_sentences_override is not None
            else context.config.atomize_min_sentences
        )

        if group_size <= 0:
            # Planning stage may disable atomization; still record metrics.
            context.add_metric("atomize_time", time.time() - start_time)
            context.add_metric("paragraphs_before_atomize", len(paragraphs))
            context.add_metric("paragraphs_after_atomize", len(paragraphs))
            context.add_metric("paragraphs_atomized", 0)
            context.add_metric("atoms_total", len(paragraphs))
            context.add_metric("atomize_sentences_per_paragraph", group_size)
            context.add_metric("atomize_min_sentences", min_sentences)
            return context

        before_count = len(paragraphs)
        after: List[Paragraph] = []

        paragraphs_atomized = 0
        atoms_total = 0
        next_index = 0

        for para in paragraphs:
            # Preserve atomic / non-paragraph elements as-is (but re-index).
            if para.is_atomic or para.element_type != ElementType.PARAGRAPH:
                after.append(
                    Paragraph(
                        text=para.text,
                        index=next_index,
                        element_type=para.element_type,
                        is_atomic=para.is_atomic,
                        language=para.language,
                        source_index=para.source_index if para.source_index is not None else para.index,
                    )
                )
                next_index += 1
                atoms_total += 1
                continue

            text = para.text.strip()
            if not text:
                continue

            sentences = [
                s.strip()
                for s in _SENTENCE_SPLIT_RE.split(text)
                if s.strip()
            ]

            if len(sentences) < max(min_sentences, group_size + 1):
                after.append(
                    Paragraph(
                        text=text,
                        index=next_index,
                        element_type=para.element_type,
                        is_atomic=False,
                        language=para.language,
                        source_index=para.source_index if para.source_index is not None else para.index,
                    )
                )
                next_index += 1
                atoms_total += 1
                continue

            paragraphs_atomized += 1

            groups: List[List[str]] = []
            current: List[str] = []
            for sent in sentences:
                current.append(sent)
                if len(current) >= group_size:
                    groups.append(current)
                    current = []

            if current:
                # Avoid creating tiny tail fragments (e.g., a single short sentence).
                if groups and len(current) < group_size:
                    groups[-1].extend(current)
                else:
                    groups.append(current)

            for group in groups:
                group_text = " ".join(group).strip()
                if not group_text:
                    continue
                after.append(
                    Paragraph(
                        text=group_text,
                        index=next_index,
                        element_type=ElementType.PARAGRAPH,
                        is_atomic=False,
                        language=para.language,
                        source_index=para.source_index if para.source_index is not None else para.index,
                    )
                )
                next_index += 1
                atoms_total += 1

        context.paragraphs = after

        # If the document was previously classified as "short", atomization might
        # increase the paragraph count enough to make boundary detection worthwhile.
        if (
            context.skip_boundary_detection
            and len(context.paragraphs) >= context.config.short_document_threshold
        ):
            context.skip_boundary_detection = False
            context.add_warning(
                "Atomization increased paragraph count; enabling boundary detection."
            )

        context.add_metric("atomize_time", time.time() - start_time)
        context.add_metric("paragraphs_before_atomize", before_count)
        context.add_metric("paragraphs_after_atomize", len(context.paragraphs))
        context.add_metric("paragraphs_atomized", paragraphs_atomized)
        context.add_metric("atoms_total", atoms_total)
        context.add_metric("atomize_sentences_per_paragraph", group_size)
        context.add_metric("atomize_min_sentences", min_sentences)

        if context.config.verbose:
            logger.info(
                "Atomized paragraphs: %s -> %s (atomized=%s)",
                before_count,
                len(context.paragraphs),
                paragraphs_atomized,
            )

        return context
