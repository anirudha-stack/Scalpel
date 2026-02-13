# Stop Splitting Documents by Character Count: Semantic Chunking for RAG That Actually Works

*Why your retrieval pipeline deserves better than `text[:500]`, and how a two-stage boundary detection approach produces chunks that answer real questions.*

---

Most RAG pipelines start the same way. You have documents. You need chunks. So you reach for the obvious solution: split every 500 tokens, add some overlap, ship it.

It works -- until it doesn't.

A 500-token window cuts a definition in half. A paragraph about compliance regulation ends up in the same chunk as one about employee housing benefits. Your embedding model dutifully encodes these mixed-topic fragments, your vector database returns the "closest" match, and your LLM generates an answer that's technically grounded in the retrieved text but somehow misses the point.

The problem isn't your vector database. It isn't your embedding model. It isn't your reranker. It's that your chunks don't represent coherent ideas.

We built [Segmenta](https://github.com/segmenta/segmenta) to fix this. It's an open-source Python library that chunks documents along **semantic boundaries** instead of arbitrary character limits -- and enriches each chunk with structured metadata that makes retrieval measurably better.

<p align="center">
  <img src="docs/images/naive_vs_semantic_chunking.svg" width="900" alt="Side-by-side comparison: fixed-size chunking mixes topics while Segmenta produces topic-pure chunks" />
</p>

*Fixed-size chunking slices through topic boundaries, mixing 3+ unrelated subjects per chunk. Semantic chunking produces one coherent topic per chunk, each with structured metadata for retrieval.*

---

## Section 1: The Core Problem -- PDF Extraction Lies to You

**Image: `docs/images/chunking_visual_demo.png`**

<p align="center">
  <img src="docs/images/chunking_visual_demo.png" width="900" alt="Side-by-side: PDF text with color-coded chunk highlights (left) and the resulting 14 semantic chunks (right)" />
</p>

Before we even get to chunking strategy, there's a deeper issue that most tutorials gloss over entirely.

When you extract text from a PDF, you don't get the same paragraph structure the author wrote. PDF extraction tools -- even excellent ones like PyMuPDF -- collapse multiple distinct ideas into single "paragraphs." This happens because the PDF format stores text as positioned graphical blocks, not as semantic units. A paragraph break in the original Word document might just be a vertical offset in the PDF. A section header might be indistinguishable from body text without font-size analysis.

Here's what this actually looks like. We tested with a 12-topic document about workforce mobility and relocation policy -- a realistic enterprise document with governance, financial planning, vendor management, housing, compliance, and six other distinct topics:

```
PDF extracted paragraphs:  7
Actual semantic topics:    12+
```

Seven paragraphs. Twelve topics. That means multiple unrelated concepts are fused into single text blocks. If your chunking strategy relies on paragraph boundaries -- whether you're using LangChain's `RecursiveCharacterTextSplitter`, LlamaIndex's `SentenceSplitter`, or any other tool that respects structural boundaries -- you've already lost the ability to separate these topics. No amount of overlap or sliding windows will un-merge ideas that the PDF extractor has combined.

Segmenta addresses this head-on with an **atomization stage**. Before any boundary detection happens, long paragraphs are split into sentence groups (by default, pairs of sentences). This gives the boundary detector enough resolution to find the actual topic shifts:

```
PDF extracted paragraphs:      7
Atomized sentence groups:     21
Final semantic chunks:        14
```

From 7 blobs to 21 sentence groups to 14 topic-pure chunks. The atomization doesn't guess where topics change -- it just creates enough granularity for the next stages to find them.

---

## Section 2: Two-Stage Boundary Detection -- Embeddings + LLM

**Image: `docs/images/two_stage_boundary_detection.svg`**

<p align="center">
  <img src="docs/images/two_stage_boundary_detection.svg" width="900" alt="Two-stage boundary detection: embedding similarity proposes candidates, LLM validates with KEEP/MERGE/ADJUST verdicts" />
</p>

This is the core architectural decision in Segmenta, and it's worth understanding why *both* stages are necessary.

### Stage 1: Embedding Similarity (Fast, Approximate)

Segmenta computes embeddings for each text unit (paragraph or atomized sentence group) using a local Sentence Transformer model. The default is `all-MiniLM-L6-v2` -- a 384-dimensional model that runs entirely on CPU. No API calls. No token costs.

For each adjacent pair of text units, Segmenta computes cosine similarity. Where similarity drops below a configurable threshold (default: 0.5), it proposes a boundary candidate.

This is fast. For a 20-page document with 100 sentence groups, you get ~100 embedding calls (batched) and 99 similarity computations. The whole stage runs in seconds on a laptop.

It works well for obvious topic shifts -- when a document moves from "financial planning" to "housing assistance," the embedding vectors diverge noticeably. But embeddings have a fundamental limitation: they measure **surface-level semantic similarity**, not **logical coherence**.

Consider these two sentences:

> "The dog runs at 10 meters per second."
> "Running fast is physically demanding."

Both are about running. Both reference speed or physical effort. An embedding model will report high similarity. But they belong in completely different chunks: one is a factual measurement about a specific animal, the other is a general observation about physical exertion. Surface similarity is high; topical coherence is low.

This is why embeddings alone aren't enough.

### Stage 2: LLM Validation (Smart, Precise)

Each boundary candidate from Stage 1 gets reviewed by an LLM. The prompt includes the text on both sides of the proposed boundary plus surrounding context (typically 2-3 units in each direction). The LLM returns one of three verdicts:

- **KEEP** -- The split is correct. These text units discuss distinct topics and should be in separate chunks for retrieval.
- **MERGE** -- The embedding model was wrong. These units are topically related and should remain in the same chunk.
- **ADJUST** -- The boundary is in the right neighborhood, but should shift by one or two positions to avoid splitting a thought mid-sentence.

In practice, this two-stage approach is dramatically more efficient than validating every adjacent pair with an LLM. If your document has 100 sentence groups, you might get 15 boundary candidates from Stage 1. That's 15 LLM calls instead of 99. You get the speed of embeddings (filtering out the obvious non-boundaries) with the precision of LLM reasoning (catching the cases where surface similarity misleads).

The diagram above shows a concrete example: 4 boundary candidates enter Stage 2. The LLM keeps 2 (genuine topic shifts), merges 1 (a false positive where embedding similarity dipped but the topic continued), and adjusts 1 (correct area but off by one position). Result: 3 validated boundaries from 4 candidates.

---

## Section 3: What Comes Out -- Chunks That Answer Questions

**Image: `docs/images/chunk_metadata_anatomy.svg`**

<p align="center">
  <img src="docs/images/chunk_metadata_anatomy.svg" width="900" alt="Anatomy of a Segmenta chunk: YAML frontmatter with title, summary, intent, keywords, questions, and token count" />
</p>

Each chunk Segmenta produces isn't just a text passage -- it's a structured unit wrapped in YAML frontmatter metadata. Every chunk includes:

**Title** (5-10 words): A concise, descriptive heading generated by the LLM. Useful for human review, debug logs, and as an additional embedding surface.

**Summary** (1-2 sentences): A distilled description of the chunk's key point. When you're scanning 200 chunks to understand a document's structure, summaries are faster than reading full text.

**Intent classification**: One of six categories -- `explains`, `compares`, `warns`, `defines`, `instructs`, `lists`. This is a lightweight but powerful signal for retrieval. If a user asks "what's the difference between domestic and international relocation?", you can boost chunks with `compares` intent. If they ask "how do I set up tax equalization?", boost `instructs`.

**Keywords** (3-7): Domain-specific terms extracted by the LLM. These bridge the gap between semantic search (embeddings) and keyword search (BM25). If your retrieval system supports hybrid search, these keywords are ready to use.

**Questions** (3-6): This is the most impactful metadata field for RAG. These are natural-language queries that a user would plausibly ask to find this specific chunk. For example, a chunk about international compliance might have:

- *What are the differences between domestic and international relocation?*
- *How do compliance requirements change for cross-border moves?*
- *What tax implications apply to international employee transfers?*

If you embed these questions alongside the chunk content in your vector store, you're matching on **how users actually phrase their queries**, not just on the content itself. In practice, this dramatically improves recall for questions that use different vocabulary than the source document.

**Token count**: Counted using `tiktoken`. Useful for LLM context window budgeting when you're stitching chunks together for generation.

```yaml
---
chunk_id: chunk_006
title: Domestic vs International Mobility Programs
summary: Compares the regulatory and logistical differences between
  domestic and international employee relocation programs.
intent: compares
keywords:
  - domestic relocation
  - international mobility
  - compliance
  - tax treatment
questions:
  - What are the differences between domestic and international relocation?
  - How do compliance requirements change for cross-border moves?
  - What tax implications apply to international employee transfers?
parent_section: Compliance and International Programs
token_count: 36
---

Domestic relocations follow a streamlined approval process with
standard benefits. International assignments require additional
compliance review, tax equalization planning, and immigration support.
```

---

## Section 4: The Pipeline -- 9 Stages, Fully Inspectable

**Image: `docs/images/segmenta_pipeline_overview.png`**

<p align="center">
  <img src="docs/images/segmenta_pipeline_overview.png" width="900" alt="Segmenta pipeline: local stages (parse, segment, atomize, embeddings, form chunks) with LLM callouts (plan granularity, validate boundaries, enrich metadata)" />
</p>

Segmenta runs a deterministic 9-stage pipeline. Every stage produces observable intermediate output, and every LLM interaction is logged:

1. **Parse** -- Extracts document structure. PDF parsing uses PyMuPDF with font-size analysis for header detection. Markdown preserves headers, lists, code blocks, and tables. Plain text splits on blank lines.

2. **Segment** -- Converts the parsed document tree into an ordered list of `Paragraph` objects. Each paragraph carries its `element_type` (paragraph, code block, table, list, header, blockquote) and source position. Code blocks and tables are marked as *atomic* -- they're never split.

3. **Plan Granularity** (LLM) -- This is the adaptive intelligence stage. Before any chunking, Segmenta sends a compact sample of the document (up to 60 paragraphs, 280 chars each) to the LLM and asks: *How many distinct topics does this document cover? How many chunks should retrieval produce? What sentence-group size should atomization use?*

   This means a 2-page memo and a 50-page technical manual get different treatment automatically. The planner returns a JSON plan with `expected_chunk_count`, `atomize_sentences_per_paragraph`, and a confidence score.

4. **Atomize** -- Splits long paragraphs into sentence groups based on the granularity plan. Only triggers for paragraphs with 6+ sentences (configurable). Groups are typically 2 sentences each. This is critical for PDFs where extraction produces wall-of-text paragraphs.

5. **Boundary Detect** (Embeddings) -- Computes cosine similarity between adjacent text units and proposes boundaries where similarity drops below threshold. Returns candidates as `(position, similarity_score)` tuples.

6. **Boundary Validate** (LLM) -- Reviews each proposed boundary with context. Returns KEEP / MERGE / ADJUST verdicts with reasoning.

7. **Form Chunks** -- Groups text units into chunks using the validated boundaries. Enforces `min_chunk_tokens` and `max_chunk_tokens` constraints (soft limits -- atomic elements like code blocks may exceed the max).

8. **Enrich** (LLM) -- Extracts structured metadata for each chunk: title, summary, intent, keywords, and questions. If the LLM call fails and `fallback_enabled` is true, Segmenta generates basic metadata from the text itself (first sentence as summary, "unknown" intent, etc.).

9. **Output** -- Writes Markdown files with YAML frontmatter. Also writes a JSONL debug trace containing every LLM prompt and response, and a JSON granularity plan file.

**The trace log is the secret weapon for debugging.** When a chunk comes out wrong, you don't have to guess what happened. Open the JSONL file, find the boundary validation or enrichment call for that chunk, and see exactly what the LLM was asked and what it replied. Each record includes the prompt, system prompt, response, token count, timing, and success/error status:

```json
{
  "type": "llm_call",
  "run_id": "MyDoc_20260205_001122",
  "timestamp": "2026-02-05T00:11:22.000000+00:00",
  "prompt": "...",
  "system_prompt": "...",
  "response": "...",
  "tokens_used": 452,
  "success": true,
  "error": null
}
```

---

## Section 5: Getting Started -- 5 Lines of Python

**Image: `docs/images/segmenta_chunking_demo.gif`**

<p align="center">
  <img src="docs/images/segmenta_chunking_demo.gif" width="700" alt="Animated demo: how Segmenta chunks a long paragraph into sentence groups and then into semantic chunks" />
</p>

Install from PyPI:

```bash
pip install segmenta
```

The simplest possible usage -- three lines of Python to chunk a document:

```python
from segmenta import Segmenta

segmenta = Segmenta(openai_api_key="sk-...", model="gpt-4o")

result = segmenta.chunk("document.pdf", output_dir="./output")

print(f"Produced {result.chunk_count} chunks")
print(f"Output:  {result.output_path}")
print(f"Time:    {result.metrics['total_time']:.1f}s")
```

Or from the command line:

```bash
# Basic PDF chunking with verbose output
segmenta document.pdf -o ./output --verbose

# Markdown input with dry run (no LLM calls -- just parse + segment + atomize)
segmenta document.md -o ./output --dry-run

# Custom model and similarity threshold
segmenta document.pdf -m gpt-4o -t 0.6 --min-tokens 50 --max-tokens 500
```

The `dry_run` flag is particularly useful during development. It runs the first four pipeline stages (parse, segment, plan, atomize) without making any LLM calls. You can inspect the intermediate state -- how many paragraphs were extracted, how atomization split them, what the similarity scores look like -- before spending tokens on validation and enrichment.

Segmenta supports **PDF**, **Markdown**, and **plain text** out of the box. It works with any OpenAI-compatible API endpoint, so you can point it at Groq, Azure OpenAI, Together AI, local vLLM, or anything else that speaks the same protocol:

```python
from segmenta.llm import OpenAIProvider

llm = OpenAIProvider(
    api_key="your-groq-key",
    base_url="https://api.groq.com/openai/v1",
    model="llama-3.3-70b-versatile",
)

segmenta = Segmenta.builder().with_llm_provider(llm).build()
result = segmenta.chunk("document.pdf", output_dir="./output")
```

The builder pattern gives you full control over providers:

```python
from segmenta import Segmenta, SegmentaConfig
from segmenta.llm import OpenAIProvider
from segmenta.embeddings import SentenceTransformerProvider

segmenta = (
    Segmenta.builder()
    .with_config(SegmentaConfig(
        similarity_threshold=0.4,
        verbose=True,
        granularity_planning_enabled=True,
    ))
    .with_llm_provider(OpenAIProvider(api_key="sk-...", model="gpt-4o"))
    .with_embedding_provider(SentenceTransformerProvider(model="all-MiniLM-L6-v2"))
    .build()
)
```

---

## Section 6: Configuration Without Guesswork

**Image: `docs/images/configuration_knobs.svg`**

<p align="center">
  <img src="docs/images/configuration_knobs.svg" width="900" alt="Configuration overview: four categories -- Chunking Controls, LLM Settings, Behavior & Resilience, and Granularity Planning -- each with defaults" />
</p>

Segmenta's defaults work well for most documents, but everything is tunable. Configuration can be set via Python, YAML file, or CLI flags.

### YAML Configuration

```yaml
# segmenta.yaml
chunking:
  similarity_threshold: 0.5    # Lower = fewer, broader chunks. Higher = more, finer chunks.
  min_chunk_tokens: 50         # Minimum tokens per chunk (merge small chunks up)
  max_chunk_tokens: 500        # Target maximum (soft limit; atomic elements may exceed)
  atomize_sentences_per_paragraph: 2   # Sentence group size for atomization
  atomize_min_sentences: 6     # Only atomize paragraphs with >= this many sentences
  granularity_planning_enabled: true   # Let LLM estimate optimal settings

behavior:
  retry_attempts: 2            # Retry failed LLM calls
  retry_delay: 1.0             # Seconds between retries
  fallback_enabled: true       # Generate basic metadata if LLM fails
  continue_on_error: false     # Halt pipeline on errors (true = skip and continue)
  verbose: false               # Debug logging

llm:
  model: gpt-4o
  temperature: 0.1             # Low temperature for consistent, deterministic output
```

### Key Tuning Decisions

**`similarity_threshold`** is the most impactful parameter. At 0.3, you'll get fewer, broader chunks (only very obvious topic shifts). At 0.7, you'll get more, finer-grained chunks (even subtle shifts trigger boundaries). The default of 0.5 is a good middle ground for most multi-topic documents.

**`granularity_planning_enabled`** adds one LLM call at the start but adapts all downstream behavior. If you're processing many similar documents (e.g., all the same template), you might disable planning and use fixed settings. For varied document types, leave it on.

**`fallback_enabled`** is your safety net. If the enrichment LLM call fails for a chunk (rate limit, timeout, malformed response), Segmenta generates basic metadata: first sentence as summary, "unknown" as intent, no questions. This means the pipeline never silently drops chunks.

### CLI Flags

```bash
segmenta document.pdf -o ./output \
  -m gpt-4o \
  -t 0.6 \
  --min-tokens 80 \
  --max-tokens 400 \
  --verbose \
  --dry-run     # Parse only, no LLM calls
```

---

## Section 7: Extending Segmenta

**Image: `docs/images/extensibility_architecture.svg`**

<p align="center">
  <img src="docs/images/extensibility_architecture.svg" width="900" alt="Pluggable architecture: pipeline core with swappable Document Parsers, LLM Providers, Embedding Providers, and Output Formatters" />
</p>

Segmenta is built around **four extension points**, each defined by an abstract base class. The pipeline doesn't care about implementations -- it only cares about interface contracts.

### Custom Document Parser

Need DOCX support? HTML? EPUB? Register a parser:

```python
from segmenta.parsers.base import DocumentParser
from segmenta.parsers import ParserFactory

class DocxParser(DocumentParser):
    def supported_extensions(self):
        return [".docx"]

    def parse(self, file_path):
        # Return a Document with sections and paragraphs
        ...

ParserFactory.register(".docx", DocxParser)
```

After registration, `segmenta.chunk("report.docx", ...)` works automatically.

### Custom LLM Provider

If you're using an LLM that doesn't support the OpenAI-compatible API format, implement the `LLMProvider` interface:

```python
from segmenta.llm.base import LLMProvider, LLMResponse

class AnthropicProvider(LLMProvider):
    @property
    def model_name(self) -> str:
        return "claude-sonnet-4-5-20250929"

    def complete(self, prompt: str, system_prompt=None) -> LLMResponse:
        # Call your LLM, return LLMResponse(text=..., tokens_used=...)
        ...

    def complete_json(self, prompt: str, system_prompt=None) -> dict:
        # Return parsed JSON response
        ...
```

### Custom Embedding Provider

Swap in your own embedding model -- maybe you need multilingual support or want to use an API-based embedding service:

```python
from segmenta.embeddings.base import EmbeddingProvider
import numpy as np

class CohereEmbeddingProvider(EmbeddingProvider):
    def embed(self, texts: list[str]) -> np.ndarray:
        # Return (n, dim) numpy array
        ...
```

### Custom Output Formatter

By default, Segmenta writes Markdown with YAML frontmatter. But you can output JSON, CSV, or directly insert into a database:

```python
from segmenta.output.base import OutputFormatter

class JSONOutputFormatter(OutputFormatter):
    def format(self, chunks):
        # Return formatted string or write directly
        ...
```

---

## Section 8: What Segmenta Intentionally Doesn't Do

**Image: `docs/images/scope_boundaries.svg`**

<p align="center">
  <img src="docs/images/scope_boundaries.svg" width="900" alt="Scope boundaries: Segmenta handles parse, segment, boundary detection, enrichment, and output. You handle overlap, vector storage, retrieval, and generation." />
</p>

Design boundaries matter as much as features. Segmenta intentionally stays out of three areas:

**No chunk overlap.** Many RAG tutorials recommend overlapping chunks by 50-100 tokens to avoid losing context at boundaries. Segmenta doesn't do this because semantic boundaries are the *right* place to split -- there's no lost context to recover. If your downstream system still wants overlap, add it when loading chunks into your vector store. That's a retrieval concern, not a chunking concern.

**No vector database integration.** Segmenta produces Markdown files with structured metadata. It doesn't know about Pinecone, Weaviate, Chroma, Qdrant, or any other vector store. This is deliberate. Vector store APIs change. Schema requirements differ. By outputting a standard format (Markdown + YAML), Segmenta fits into any pipeline without coupling.

**No RAG embedding generation.** The embeddings Segmenta uses internally (for boundary detection) are a *means*, not an *end*. They use a small, fast model optimized for adjacent-text similarity. Your production RAG system likely uses a different (larger) embedding model optimized for query-document similarity. Conflating these two concerns would be a design mistake.

The result: Segmenta does **one thing well** -- transforms unstructured documents into semantically coherent, metadata-rich chunks. Everything downstream (indexing, retrieval, reranking, generation) stays in your control and your existing toolchain.

---

## Section 9: When to Use Segmenta

**Image: `docs/images/when_to_use_segmenta.svg`**

<p align="center">
  <img src="docs/images/when_to_use_segmenta.svg" width="900" alt="Decision guide: great fit scenarios vs probably overkill scenarios" />
</p>

### Great Fit

- **Multi-topic documents** -- Policy manuals, technical documentation, research papers, legal contracts. Anything where a single document covers multiple distinct subjects that a user might query independently.

- **PDF-heavy pipelines** -- If your document corpus is mostly PDFs, you're almost certainly dealing with the paragraph-collapse problem. Segmenta's atomization stage was built specifically for this.

- **Quality-sensitive RAG** -- When wrong answers have real consequences (legal, medical, financial, compliance), investing LLM tokens in chunk quality pays for itself through better retrieval precision.

- **Auditable pipelines** -- If you need to explain *why* a document was chunked a certain way (regulated industries, enterprise customers, debugging production issues), the JSONL trace log provides a complete audit trail.

- **Hybrid search architectures** -- If your retrieval uses both embeddings and keywords (BM25 + vector), the structured metadata (keywords, questions, intent) is immediately useful without additional processing.

### Probably Overkill

- **Short, single-topic documents** -- Blog posts, emails, chat messages. If the whole document is one topic, semantic boundary detection has nothing to find.

- **Keyword-only search** -- If you're using BM25 / Elasticsearch without vector retrieval, the boundary detection (which relies on embeddings) adds cost without proportional benefit.

- **Extreme token budget constraints** -- The LLM calls (granularity planning + boundary validation + enrichment) add cost. For a 20-page document, expect roughly 20-40 LLM calls. If tokens are extremely expensive in your setup, consider using `dry_run` mode with manual threshold tuning.

- **Already-structured data** -- JSON records, CSV rows, database exports. This data is already chunked by its structure; semantic boundary detection isn't needed.

- **Real-time / streaming ingestion** -- Segmenta processes one file at a time in batch mode. It's not designed for streaming document ingestion.

---

## Section 10: Try It

```bash
pip install segmenta
```

The library is open source under **Apache License 2.0**. The [GitHub repository](https://github.com/segmenta/segmenta) includes:

- A test PDF with 12 distributed topics that demonstrates the full pipeline
- Visual output showing how source PDF text maps to final semantic chunks
- YAML configuration examples
- Provider interop examples (Groq, Azure OpenAI)

If your RAG pipeline is splitting documents by character count, give semantic chunking a try. Run Segmenta on one of your real documents, compare the output to your current chunks, and see whether topic purity makes a difference in your retrieval quality.

The difference is usually hard to go back from.

---

*Segmenta is available on [PyPI](https://pypi.org/project/segmenta/) and [GitHub](https://github.com/segmenta/segmenta). Contributions, issues, and feedback are welcome.*

---

## Image Reference Guide (for Medium publishing)

When publishing this post on Medium, use the following image for each section:

| Section | Image File | Description |
|---------|-----------|-------------|
| Introduction (hero) | `docs/images/naive_vs_semantic_chunking.svg` | Side-by-side: fixed-size vs semantic chunking |
| 1. PDF Extraction | `docs/images/chunking_visual_demo.png` | PDF with color-coded chunk highlights + chunk list |
| 2. Boundary Detection | `docs/images/two_stage_boundary_detection.svg` | Two-stage flow: embeddings propose, LLM validates |
| 3. Chunk Metadata | `docs/images/chunk_metadata_anatomy.svg` | Annotated anatomy of a chunk with YAML frontmatter |
| 4. Pipeline | `docs/images/segmenta_pipeline_overview.png` | Full pipeline: local stages + LLM callouts |
| 5. Getting Started | `docs/images/segmenta_chunking_demo.gif` | Animated: sentence groups to semantic chunks |
| 6. Configuration | `docs/images/configuration_knobs.svg` | Four config categories with defaults |
| 7. Extensibility | `docs/images/extensibility_architecture.svg` | Pluggable providers: parsers, LLM, embeddings, output |
| 8. Scope Boundaries | `docs/images/scope_boundaries.svg` | What Segmenta does vs your responsibility |
| 9. When to Use | `docs/images/when_to_use_segmenta.svg` | Great fit vs probably overkill decision guide |
| 10. Try It | *(use GitHub repo screenshot or logo)* | Call-to-action |
