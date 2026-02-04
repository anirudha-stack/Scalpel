"""Prompt templates for chunk enrichment."""

ENRICHMENT_SYSTEM = """You are an expert at analyzing text and extracting structured metadata.
Your task is to extract key information that summarizes and categorizes a chunk of text.

You must respond with valid JSON only, no additional text."""

ENRICHMENT_TEMPLATE = """Extract metadata for this text chunk:

\"\"\"
{chunk_content}
\"\"\"

Parent Section: {parent_section}

Respond with a JSON object:
{{
    "title": "Concise, descriptive title (5-10 words)",
    "summary": "1-2 sentence summary of the main point",
    "intent": "explains" | "lists" | "warns" | "defines" | "instructs" | "compares",
    "keywords": ["keyword1", "keyword2", ...] // 3-7 relevant terms
}}

Guidelines:
- Title: Capture the essence without being too generic
- Summary: What would someone learn from this chunk?
- Intent: What is this chunk trying to do for the reader?
  - explains: Provides explanation or description
  - lists: Enumerates items or options
  - warns: Highlights cautions or warnings
  - defines: Provides definitions
  - instructs: Gives step-by-step instructions
  - compares: Compares or contrasts items
- Keywords: Terms someone might search for to find this content

Respond with JSON only."""


def format_enrichment_prompt(chunk_content: str, parent_section: str) -> str:
    """Format the enrichment prompt with the given chunk content.

    Args:
        chunk_content: The text content of the chunk
        parent_section: The parent section title

    Returns:
        Formatted prompt string
    """
    return ENRICHMENT_TEMPLATE.format(
        chunk_content=chunk_content,
        parent_section=parent_section or "Document Root",
    )
