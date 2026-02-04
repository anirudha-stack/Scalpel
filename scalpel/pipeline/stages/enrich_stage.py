"""Enrichment stage - adds metadata to chunks using LLM."""

import time
import logging
from typing import Optional, List
import uuid

from scalpel.pipeline.base import PipelineStage
from scalpel.pipeline.context import PipelineContext
from scalpel.llm.base import LLMProvider
from scalpel.llm.prompts.enrichment import (
    ENRICHMENT_SYSTEM,
    format_enrichment_prompt,
)
from scalpel.models import Chunk, ChunkMetadata, Intent
from scalpel.utils.retry import RetryHandler
from scalpel.utils.token_counter import TokenCounter
from scalpel.exceptions import ScalpelLLMError

logger = logging.getLogger(__name__)


class EnrichStage(PipelineStage):
    """Stage that enriches chunks with LLM-generated metadata."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        token_counter: TokenCounter,
        retry_handler: Optional[RetryHandler] = None,
        fallback_enabled: bool = True,
    ) -> None:
        """Initialize the enrichment stage.

        Args:
            llm_provider: LLM provider for enrichment
            token_counter: Token counter for measuring chunk size
            retry_handler: Optional retry handler for LLM calls
            fallback_enabled: Whether to use fallback metadata on failure
        """
        self._llm = llm_provider
        self._token_counter = token_counter
        self._retry = retry_handler
        self._fallback_enabled = fallback_enabled
        self._chunk_counter = 0

    @property
    def name(self) -> str:
        return "enrich"

    def should_skip(self, context: PipelineContext) -> bool:
        """Skip if in dry run mode."""
        return context.dry_run

    def process(self, context: PipelineContext) -> PipelineContext:
        """Enrich chunks with metadata.

        Args:
            context: Pipeline context

        Returns:
            Updated context with enriched chunks
        """
        start_time = time.time()
        self._chunk_counter = 0

        enriched_chunks: List[Chunk] = []

        for chunk in context.chunks:
            # Find parent section for this chunk
            parent_section = self._find_parent_section(chunk, context)

            try:
                metadata = self._enrich_chunk(chunk, parent_section)
                chunk.metadata = metadata
            except ScalpelLLMError as e:
                logger.warning(f"LLM enrichment failed: {e}")
                if self._fallback_enabled:
                    chunk.metadata = self._fallback_metadata(chunk, parent_section)
                    context.add_warning(f"Using fallback metadata: {e}")
                else:
                    context.add_error(f"Enrichment failed: {e}")

            enriched_chunks.append(chunk)

        context.chunks = enriched_chunks

        # Record metrics
        context.add_metric("enrich_time", time.time() - start_time)
        context.add_metric(
            "chunks_enriched",
            sum(1 for c in context.chunks if c.is_enriched),
        )

        return context

    def _find_parent_section(self, chunk: Chunk, context: PipelineContext) -> str:
        """Find the parent section title for a chunk.

        Args:
            chunk: Chunk to find parent for
            context: Pipeline context

        Returns:
            Parent section title
        """
        if not chunk.source_paragraphs or context.document is None:
            return ""

        # Get the first paragraph's section
        first_para_index = chunk.source_paragraphs[0]
        section = context.document.get_section_for_paragraph(first_para_index)

        if section:
            return section.title

        return ""

    def _enrich_chunk(self, chunk: Chunk, parent_section: str) -> ChunkMetadata:
        """Enrich a chunk with LLM-generated metadata.

        Args:
            chunk: Chunk to enrich
            parent_section: Parent section title

        Returns:
            ChunkMetadata for the chunk
        """
        prompt = format_enrichment_prompt(chunk.content, parent_section)

        if self._retry:
            result = self._retry.execute(
                self._llm.complete_json,
                prompt,
                ENRICHMENT_SYSTEM,
            )
        else:
            result = self._llm.complete_json(prompt, ENRICHMENT_SYSTEM)

        # Generate chunk ID
        self._chunk_counter += 1
        chunk_id = f"chunk_{self._chunk_counter:03d}"

        # Count tokens
        token_count = self._token_counter.count(chunk.content)

        return ChunkMetadata(
            chunk_id=chunk_id,
            title=result.get("title", "Untitled"),
            summary=result.get("summary", ""),
            intent=Intent.from_string(result.get("intent", "unknown")),
            keywords=result.get("keywords", []),
            parent_section=parent_section,
            token_count=token_count,
        )

    def _fallback_metadata(self, chunk: Chunk, parent_section: str) -> ChunkMetadata:
        """Generate fallback metadata when LLM fails.

        Args:
            chunk: Chunk to generate metadata for
            parent_section: Parent section title

        Returns:
            Fallback ChunkMetadata
        """
        self._chunk_counter += 1
        chunk_id = f"chunk_{self._chunk_counter:03d}"

        # Extract first sentence as title
        sentences = chunk.content.split(". ")
        title = sentences[0][:50] + "..." if len(sentences[0]) > 50 else sentences[0]

        # Count tokens
        token_count = self._token_counter.count(chunk.content)

        return ChunkMetadata(
            chunk_id=chunk_id,
            title=title,
            summary="Content summary unavailable",
            intent=Intent.UNKNOWN,
            keywords=self._extract_simple_keywords(chunk.content),
            parent_section=parent_section,
            token_count=token_count,
        )

    def _extract_simple_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """Simple keyword extraction as fallback.

        Args:
            text: Text to extract keywords from
            max_keywords: Maximum number of keywords

        Returns:
            List of keywords
        """
        # Common stop words
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "can", "this", "that", "these",
            "those", "i", "you", "he", "she", "it", "we", "they", "what", "which",
            "who", "whom", "when", "where", "why", "how", "all", "each", "every",
            "both", "few", "more", "most", "other", "some", "such", "no", "nor",
            "not", "only", "own", "same", "so", "than", "too", "very", "just",
            "and", "but", "if", "or", "because", "as", "until", "while", "of",
            "at", "by", "for", "with", "about", "against", "between", "into",
            "through", "during", "before", "after", "above", "below", "to",
            "from", "up", "down", "in", "out", "on", "off", "over", "under",
        }

        words = text.lower().split()
        word_freq: dict = {}

        for word in words:
            # Clean the word
            word = "".join(c for c in word if c.isalnum())
            if word and word not in stop_words and len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1

        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:max_keywords]]
