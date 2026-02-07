"""Prompt templates for boundary validation."""

from __future__ import annotations

from typing import Optional

BOUNDARY_VALIDATION_SYSTEM = """You are an expert at semantic chunking for retrieval (RAG).
Your task is to decide whether a proposed boundary between two adjacent text segments should be a chunk split.

Priorities (most important first):
1) Topic purity: Prefer separating distinct subtopics into different chunks, even if they share a broader theme.
2) Coherence: Do NOT split a continuous thought where the second segment is clearly a continuation, example,
   definition, or immediate elaboration of the first.
3) Retrievability: Avoid creating tiny fragments that only make sense with the previous segment (e.g., segments
   starting with anaphora like "this/these/both/together", or purely transitional/summary sentences).

Guidance:
- KEEP if the second segment introduces a new operational subtopic (policy area, process step, responsibility,
  risk, exception, measurement, etc.).
- MERGE only if the split would reduce clarity or break a single unified point.
- When unsure, lean KEEP.

You must respond with valid JSON only, no additional text."""

BOUNDARY_VALIDATION_TEMPLATE = """Below is a proposed chunk boundary in a document.
Analyze whether these segments should be separate chunks for retrieval.

{context_block}END OF CURRENT CHUNK:
\"\"\"
{text_before}
\"\"\"

START OF PROPOSED NEW CHUNK:
\"\"\"
{text_after}
\"\"\"

Respond with a JSON object:
{{
    "verdict": "KEEP" | "MERGE" | "ADJUST",
    "reason": "Brief explanation of your decision",
    "confidence": 0.0-1.0
}}

Rules:
- KEEP: The boundary is correct. These are distinct logical units.
- MERGE: These belong together. The split breaks a continuous thought.
- ADJUST: The split point is wrong but a boundary nearby makes sense.

Respond with JSON only."""


BOUNDARY_CRITIQUE_SYSTEM = """You are a strict boundary critic for retrieval chunking.

You must output exactly one token: YES or NO (uppercase), and nothing else.

Interpret the question as: "Is there a real discourse break here such that a boundary should exist?"
- YES: keep the boundary (split here)
- NO: veto the boundary (do not split here)

No explanations. No rewriting. No restructuring."""

BOUNDARY_CRITIQUE_TEMPLATE = """{context_block}END OF CURRENT CHUNK:
\"\"\"
{text_before}
\"\"\"

START OF PROPOSED NEW CHUNK:
\"\"\"
{text_after}
\"\"\"

Question (answer YES or NO only):
Does inserting a boundary here break narrative continuity, reference resolution, or discourse intent?"""


def format_validation_prompt(
    text_before: str,
    text_after: str,
    *,
    context_block: Optional[str] = None,
) -> str:
    """Format the validation prompt with the given text segments.

    Args:
        text_before: Text at the end of the current chunk
        text_after: Text at the start of the proposed new chunk
        context_block: Optional context block (should include trailing blank line)

    Returns:
        Formatted prompt string
    """
    return BOUNDARY_VALIDATION_TEMPLATE.format(
        context_block=context_block or "",
        text_before=text_before,
        text_after=text_after,
    )


def format_boundary_critique_prompt(
    text_before: str,
    text_after: str,
    *,
    context_block: Optional[str] = None,
) -> str:
    """Format the boundary critique prompt with the given text segments."""
    return BOUNDARY_CRITIQUE_TEMPLATE.format(
        context_block=context_block or "",
        text_before=text_before,
        text_after=text_after,
    )
