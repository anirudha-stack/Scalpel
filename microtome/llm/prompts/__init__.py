"""Prompt templates for LLM operations."""

from microtome.llm.prompts.validation import (
    BOUNDARY_VALIDATION_SYSTEM,
    BOUNDARY_VALIDATION_TEMPLATE,
)
from microtome.llm.prompts.enrichment import (
    ENRICHMENT_SYSTEM,
    ENRICHMENT_TEMPLATE,
)
from microtome.llm.prompts.granularity import (
    GRANULARITY_SYSTEM,
    GRANULARITY_TEMPLATE,
)

__all__ = [
    "BOUNDARY_VALIDATION_SYSTEM",
    "BOUNDARY_VALIDATION_TEMPLATE",
    "ENRICHMENT_SYSTEM",
    "ENRICHMENT_TEMPLATE",
    "GRANULARITY_SYSTEM",
    "GRANULARITY_TEMPLATE",
]
