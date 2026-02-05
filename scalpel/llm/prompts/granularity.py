"""Prompt templates for granularity planning.

This is used as an optional first-pass LLM call to estimate:
- high-level topics covered by the document
- expected number of chunks for retrieval
- recommended atomization settings (sentence-group size)
"""

GRANULARITY_SYSTEM = """You are an expert at semantic chunking for retrieval (RAG).

Your goal is to choose a chunking granularity that:
- maximizes topic purity (separate distinct subtopics into different chunks)
- avoids tiny fragments that only make sense with adjacent text
- balances cost (too-fine atomization can create too many boundaries to validate)

You must respond with valid JSON only, no additional text."""

GRANULARITY_TEMPLATE = """Plan chunking granularity for this document.

Context:
- The pipeline may atomize long paragraphs into smaller sentence groups *before* boundary detection.
- Atomization is only a pre-processing step to expose semantic boundaries that PDFs often hide.

Document stats:
- file_type: {file_type}
- paragraph_count: {paragraph_count}
- section_count: {section_count}
- section_titles: {section_titles}

Current constraints:
- min_chunk_tokens: {min_chunk_tokens}
- max_chunk_tokens: {max_chunk_tokens}
- similarity_threshold: {similarity_threshold}

Sampled paragraphs (each line is a paragraph with basic stats):
{paragraph_lines}

Task:
1) List the brief topics covered by the document (6-20 items).
2) Estimate the expected number of semantic chunks for retrieval.
3) Recommend atomization settings:
   - atomize_sentences_per_paragraph: 0 (disable) OR an integer 2-6
   - atomize_min_sentences: integer 4-12
4) Provide a short rationale.

Respond with this JSON schema:
{{
  "topics": ["topic1", "topic2", "..."],
  "expected_chunk_count": 12,
  "expected_chunk_count_range": [10, 14],
  "atomize_sentences_per_paragraph": 2,
  "atomize_min_sentences": 6,
  "rationale": "brief explanation",
  "confidence": 0.0
}}

Rules:
- Return JSON only.
- expected_chunk_count and the range must be positive integers.
- confidence must be between 0.0 and 1.0."""


def format_granularity_prompt(
    *,
    file_type: str,
    paragraph_count: int,
    section_count: int,
    section_titles: str,
    min_chunk_tokens: int,
    max_chunk_tokens: int,
    similarity_threshold: float,
    paragraph_lines: str,
) -> str:
    """Format the granularity prompt."""
    return GRANULARITY_TEMPLATE.format(
        file_type=file_type,
        paragraph_count=paragraph_count,
        section_count=section_count,
        section_titles=section_titles,
        min_chunk_tokens=min_chunk_tokens,
        max_chunk_tokens=max_chunk_tokens,
        similarity_threshold=similarity_threshold,
        paragraph_lines=paragraph_lines,
    )
