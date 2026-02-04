# Scalpel

**Semantic Document Chunking Library**

Transform unstructured documents (PDF, Markdown, Text) into semantically coherent, metadata-enriched chunks.

## Core Value Proposition

Documents are split by **logical and semantic boundaries** — not arbitrary character/token limits — producing chunks that represent complete, self-contained ideas.

## Installation

```bash
pip install scalpel
```

Or install from source:

```bash
git clone https://github.com/your-org/scalpel.git
cd scalpel
pip install -e .
```

## Quick Start

### Simple Usage

```python
from scalpel import Scalpel

# Initialize with OpenAI API key
scalpel = Scalpel(
    openai_api_key="sk-...",
    model="gpt-4o"
)

# Process a document
result = scalpel.chunk(
    input_file="document.pdf",
    output_dir="./output"
)

print(f"Generated {result.chunk_count} chunks")
print(f"Output: {result.output_path}")
```

### Advanced Usage with Builder

```python
from scalpel import Scalpel, ScalpelConfig
from scalpel.llm import OpenAIProvider
from scalpel.embeddings import SentenceTransformerProvider

# Custom configuration
config = ScalpelConfig(
    similarity_threshold=0.5,
    min_chunk_tokens=50,
    max_chunk_tokens=500,
    retry_attempts=3,
    fallback_enabled=True,
)

# Build with custom components
scalpel = (
    Scalpel.builder()
    .with_config(config)
    .with_llm_provider(
        OpenAIProvider(
            api_key="sk-...",
            model="gpt-4o",
            temperature=0.1
        )
    )
    .with_embedding_provider(
        SentenceTransformerProvider(
            model_name="all-MiniLM-L6-v2",
            device="cuda"
        )
    )
    .build()
)

# Process with progress callback
def on_progress(stage: str, progress: float):
    print(f"[{progress:.0%}] {stage}")

result = scalpel.chunk(
    input_file="document.pdf",
    output_dir="./output",
    progress_callback=on_progress
)
```

## CLI Usage

```bash
# Basic usage
scalpel document.pdf

# With options
scalpel document.md -o ./output --model gpt-4o --verbose

# With config file
scalpel document.txt --config scalpel.yaml

# Dry run (no LLM calls)
scalpel document.pdf --dry-run
```

### CLI Options

```
Arguments:
  INPUT_FILE              Path to input document (PDF, MD, or TXT)

Options:
  -o, --output-dir PATH   Output directory [default: ./output]
  -m, --model TEXT        OpenAI model name [default: gpt-4o]
  -k, --api-key TEXT      OpenAI API key (or set OPENAI_API_KEY env var)
  -c, --config PATH       Path to configuration YAML file
  -t, --threshold FLOAT   Similarity threshold for boundaries [default: 0.5]
  --min-tokens INT        Minimum tokens per chunk [default: 50]
  --max-tokens INT        Maximum tokens per chunk [default: 500]
  -v, --verbose           Enable verbose output
  --dry-run               Parse document without LLM calls
  --version               Show version and exit
  --help                  Show this message and exit
```

## Output Format

Each chunk in the output file contains YAML frontmatter with metadata:

```markdown
---
chunk_id: chunk_001
title: PWM Fundamentals and Duty Cycle
summary: Explains PWM as a power control method using rapid on-off switching.
intent: explains
keywords: [PWM, pulse width modulation, duty cycle, power control]
parent_section: PWM Fundamentals
token_count: 58
---

PWM stands for Pulse Width Modulation. It controls power delivery by switching
voltage on and off rapidly. The duty cycle determines average power output.
```

## How It Works

### Two-Stage Chunking

1. **Stage 1 - Embedding-based Detection**: Uses Sentence Transformers to compute semantic similarity between adjacent paragraphs. Proposes boundaries where similarity drops below threshold.

2. **Stage 2 - LLM Validation**: Reviews each proposed boundary and decides:
   - **KEEP**: These are distinct logical units
   - **MERGE**: These belong together
   - **ADJUST**: Split point should be elsewhere

### Pipeline Stages

```
INPUT → PARSE → SEGMENT → BOUNDARY DETECT → VALIDATE → CHUNK → ENRICH → OUTPUT
```

## Configuration

### YAML Configuration File

```yaml
# scalpel.yaml
llm:
  provider: openai
  model: gpt-4o
  api_key: ${OPENAI_API_KEY}
  temperature: 0.1

embedding:
  model: all-MiniLM-L6-v2
  device: cuda

chunking:
  similarity_threshold: 0.5
  min_tokens: 50
  max_tokens: 500

behavior:
  retry_attempts: 3
  fallback_enabled: true
  continue_on_error: false
  verbose: false
```

## Extending Scalpel

### Custom Document Parser

```python
from scalpel.parsers.base import DocumentParser
from scalpel.parsers import ParserFactory

class DocxParser(DocumentParser):
    def supported_extensions(self):
        return ['.docx']

    def parse(self, file_path):
        # Implementation
        ...

# Register the parser
ParserFactory.register('.docx', DocxParser)
```

### Custom LLM Provider

```python
from scalpel.llm.base import LLMProvider, LLMResponse

class AnthropicProvider(LLMProvider):
    def complete(self, prompt, system_prompt=None):
        # Implementation
        ...

    def complete_json(self, prompt, system_prompt=None):
        # Implementation
        ...

# Use with Scalpel
scalpel = (
    Scalpel.builder()
    .with_llm_provider(AnthropicProvider(api_key="..."))
    .build()
)
```

## Dependencies

- `sentence-transformers` - Embedding generation
- `openai` - LLM integration
- `pymupdf` - PDF parsing
- `markdown-it-py` - Markdown parsing
- `tiktoken` - Token counting
- `click` / `rich` - CLI interface

## License

Apache License 2.0
