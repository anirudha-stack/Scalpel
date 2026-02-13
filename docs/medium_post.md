# Stop Splitting Documents by Character Count: Semantic Chunking for RAG That Actually Works

*Why your retrieval pipeline deserves better than `text[:500]`, and how a two-stage boundary detection approach produces chunks that answer real questions.*

---

Most RAG pipelines start the same way. You have documents. You need chunks. So you split on token count, maybe with some overlap, and hope for the best.

It works -- until it doesn't. A 500-token window cuts a definition in half. A paragraph about compliance lands in the same chunk as one about employee housing. Your retrieval returns fragments that almost answer the question but never quite get there.

The problem isn't your vector database or your embedding model. It's that your chunks don't represent coherent ideas.

We built [Segmenta](https://github.com/segmenta/segmenta) to fix this. It's an open-source Python library that chunks documents along semantic boundaries instead of arbitrary character limits -- and enriches each chunk with structured metadata that makes retrieval measurably better.

## The Core Problem: PDF Extraction Lies to You

Before we even get to chunking strategy, there's a deeper issue most tutorials ignore.

When you extract text from a PDF, you don't get the same paragraph structure the author wrote. PDF extraction tools (even good ones) collapse multiple distinct ideas into single "paragraphs" because the underlying format stores text as positioned blocks, not semantic units.

Here's what this looks like in practice. Take a 12-topic document about workforce mobility and relocation policy:

```
Extracted paragraphs:  7
Actual semantic topics: 12+
```

Seven paragraphs. Twelve topics. If you chunk by paragraph boundaries alone, you've already lost. Multiple unrelated concepts are fused together, and no amount of overlap or sliding windows will separate them.

Segmenta addresses this head-on with an **atomization stage** that splits long paragraphs into sentence groups *before* detecting boundaries. The result:

```
Extracted paragraphs:     7
Atomized sentence groups: 21
Final semantic chunks:    14
```

Now we have enough resolution to find the actual topic boundaries.

## Two-Stage Boundary Detection: Embeddings + LLM

This is the core architectural decision in Segmenta, and it's worth understanding why both stages matter.

### Stage 1: Embedding Similarity (Fast, Approximate)

Segmenta computes embeddings for each text unit (paragraph or atomized sentence group) using a local Sentence Transformer model (`all-MiniLM-L6-v2` by default -- 384 dimensions, runs on CPU). Then it calculates cosine similarity between adjacent units.

Where similarity drops below a threshold, Segmenta proposes a boundary.

This is fast and works well for obvious topic shifts. But embeddings have a fundamental limitation: they measure **surface-level semantic similarity**, not **logical coherence**.

Consider:

> "The dog runs at 10 meters per second."
> "Running fast is physically demanding."

These sentences have high embedding similarity -- both are about running and speed. But they belong in different chunks: one is a factual measurement, the other a general observation. An embedding model can't tell the difference.

### Stage 2: LLM Validation (Smart, Precise)

Each proposed boundary gets reviewed by an LLM with surrounding context. The LLM returns one of three verdicts:

- **KEEP**: The split is correct. These are distinct topics.
- **MERGE**: These should stay together. The embedding model was wrong.
- **ADJUST**: The boundary is in the right area but should shift by a position or two.

This two-stage approach gives you the speed of embeddings (no LLM calls for obvious non-boundaries) with the precision of LLM reasoning (catching the subtle cases where surface similarity misleads).

## What Comes Out: Chunks That Answer Questions

Each chunk Segmenta produces isn't just text -- it's a structured unit with YAML frontmatter metadata:

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

The `questions` field is particularly useful. These are LLM-generated retrieval queries that would naturally lead a user to this chunk. You can embed these alongside the chunk content to dramatically improve recall -- your vector search now matches on how users *ask* about the content, not just the content itself.

The `intent` field (explains, compares, warns, defines, instructs, lists) lets you filter or boost results based on what kind of answer the user needs.

## The Pipeline: 9 Stages, Fully Inspectable

Segmenta runs a deterministic pipeline where every stage produces observable intermediate output:

```
Parse -> Segment -> Plan Granularity (LLM) -> Atomize ->
Boundary Detect (Embeddings) -> Boundary Validate (LLM) ->
Form Chunks -> Enrich (LLM) -> Output
```

The **granularity planning** stage is worth highlighting. Before any chunking happens, Segmenta sends a compact sample of the document to the LLM and asks: how many topics are here? How granular should we go? This adaptive planning means a 2-page memo and a 50-page technical manual get different treatment automatically.

Every LLM call is logged to a JSONL trace file -- the exact prompt sent, the response received, token counts, and timing. When a chunk comes out wrong, you can trace exactly which stage made which decision.

## Getting Started: 5 Lines of Python

```python
from segmenta import Segmenta

segmenta = Segmenta(openai_api_key="sk-...", model="gpt-4o")

result = segmenta.chunk("document.pdf", output_dir="./output")

print(f"Produced {result.chunk_count} chunks")
print(f"Output: {result.output_path}")
```

Or from the command line:

```bash
pip install segmenta

segmenta document.pdf -o ./output --verbose
```

Segmenta supports PDF, Markdown, and plain text out of the box. It works with any OpenAI-compatible API endpoint, so you can point it at Groq, Azure OpenAI, local models, or anything else that speaks the same protocol:

```python
from segmenta.llm import OpenAIProvider

llm = OpenAIProvider(
    api_key="your-key",
    base_url="https://api.groq.com/openai/v1",
    model="llama-3.3-70b-versatile",
)

segmenta = Segmenta.builder().with_llm_provider(llm).build()
```

## Configuration Without Guesswork

The defaults work for most documents, but everything is tunable via YAML:

```yaml
chunking:
  similarity_threshold: 0.5    # Lower = fewer boundaries
  min_chunk_tokens: 50
  max_chunk_tokens: 500
  granularity_planning_enabled: true

behavior:
  retry_attempts: 2
  fallback_enabled: true       # Basic metadata if LLM fails
  verbose: false

llm:
  model: gpt-4o
  temperature: 0.1
```

The `dry_run` flag is useful during development -- it runs parsing, segmentation, and atomization without making any LLM calls, so you can inspect the intermediate state before spending tokens.

## Extending Segmenta

The library is designed around pluggable providers. Need to support DOCX? Register a parser:

```python
from segmenta.parsers.base import DocumentParser
from segmenta.parsers import ParserFactory

class DocxParser(DocumentParser):
    def supported_extensions(self):
        return [".docx"]
    def parse(self, file_path):
        ...

ParserFactory.register(".docx", DocxParser)
```

Custom embedding providers, LLM providers, and output formatters follow the same pattern. The pipeline doesn't care where embeddings come from or which LLM validates boundaries -- it only cares about the interface contract.

## What Segmenta Intentionally Doesn't Do

Design boundaries matter as much as features:

- **No chunk overlap** -- That's the consumer's responsibility when building the retrieval index.
- **No vector database integration** -- Segmenta produces chunks. What you do with them is up to you.
- **No embedding generation for RAG** -- The embeddings used internally for boundary detection are separate from what you'd store in a vector DB.

This keeps the library focused: transform documents into semantically coherent, metadata-rich chunks. Everything downstream -- indexing, retrieval, reranking -- stays in your control.

## When to Use Segmenta

Segmenta is a good fit when:

- Your documents have **multiple topics** that shouldn't be mixed in a single chunk
- You're working with **PDFs** where extraction collapses structure
- Your retrieval quality matters enough to spend **LLM tokens on chunk quality** (boundary validation + enrichment)
- You want **inspectable, auditable** chunking with trace logs
- You need **structured metadata** (intent, keywords, questions) beyond raw text

It's probably not necessary when:

- Your documents are short and single-topic
- You're doing simple keyword search, not semantic retrieval
- Token budget is extremely constrained (the LLM calls add cost)

## Try It

```bash
pip install segmenta
```

The library is open source under Apache 2.0. The [GitHub repository](https://github.com/segmenta/segmenta) includes a test PDF with 12 distributed topics that demonstrates the full pipeline, along with visual output showing how source text maps to final chunks.

If your RAG pipeline is splitting documents by character count, give semantic chunking a try. The difference in retrieval quality is hard to go back from.

---

*Segmenta is available on [PyPI](https://pypi.org/project/segmenta/) and [GitHub](https://github.com/segmenta/segmenta). Contributions and feedback are welcome.*
