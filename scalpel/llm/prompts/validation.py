"""Prompt templates for boundary validation."""

BOUNDARY_VALIDATION_SYSTEM = """You are an expert at analyzing document structure and logical coherence.
Your task is to determine if a proposed chunk boundary between two text segments is appropriate.
A good boundary separates distinct logical units or topics.
A bad boundary splits a continuous thought or related concepts.

You must respond with valid JSON only, no additional text."""

BOUNDARY_VALIDATION_TEMPLATE = """Below is a proposed chunk boundary in a document.
Analyze whether these segments should be separate chunks.

END OF CURRENT CHUNK:
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


def format_validation_prompt(text_before: str, text_after: str) -> str:
    """Format the validation prompt with the given text segments.

    Args:
        text_before: Text at the end of the current chunk
        text_after: Text at the start of the proposed new chunk

    Returns:
        Formatted prompt string
    """
    return BOUNDARY_VALIDATION_TEMPLATE.format(
        text_before=text_before,
        text_after=text_after,
    )
