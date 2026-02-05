# Segmenta: Implementation Plan

## Technical Design Document

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Design Patterns](#2-design-patterns)
3. [Folder Structure](#3-folder-structure)
4. [Data Models](#4-data-models)
5. [Pipeline Architecture](#5-pipeline-architecture)
6. [Public API Design](#6-public-api-design)
7. [Component Specifications](#7-component-specifications)
8. [Error Handling Strategy](#8-error-handling-strategy)
9. [Edge Case Handling](#9-edge-case-handling)
10. [Dependencies](#10-dependencies)
11. [Testing Strategy](#11-testing-strategy)
12. [Implementation Phases](#12-implementation-phases)
13. [Extensibility Guide](#13-extensibility-guide)
14. [Performance Considerations](#14-performance-considerations)
15. [CLI Specification](#15-cli-specification)

---

## 1. Architecture Overview

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Segmenta LIBRARY                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   INPUT     │    │   PARSE     │    │  SEGMENT    │    │  BOUNDARY   │  │
│  │  (PDF/MD/   │───▶│  Document   │───▶│  Extract    │───▶│  DETECTION  │  │
│  │   TXT)      │    │  Parser     │    │  Paragraphs │    │  (Stage 1)  │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └──────┬──────┘  │
│                                                                   │         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌──────▼──────┐  │
│  │   OUTPUT    │    │   FORMAT    │    │  ENRICH     │    │  VALIDATE   │  │
│  │  (Markdown  │◀───│  Chunks to  │◀───│  with LLM   │◀───│  BOUNDARIES │  │
│  │   File)     │    │  Markdown   │    │  Metadata   │    │  (Stage 2)  │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Core Principles

| Principle | Implementation |
|-----------|----------------|
| **Single Responsibility** | Each module handles one concern |
| **Open/Closed** | Open for extension (new parsers, providers), closed for modification |
| **Dependency Inversion** | Depend on abstractions, not concrete implementations |
| **Interface Segregation** | Small, focused interfaces for each capability |
| **Testability** | All components injectable and mockable |

---

## 2. Design Patterns

### 2.1 Strategy Pattern

**Purpose**: Enable interchangeable algorithms for parsing, embedding, and LLM operations.

```python
# Abstract Strategy
class DocumentParser(ABC):
    @abstractmethod
    def parse(self, file_path: Path) -> Document:
        pass

    @abstractmethod
    def supports(self, file_extension: str) -> bool:
        pass

# Concrete Strategies
class PDFParser(DocumentParser):
    def parse(self, file_path: Path) -> Document: ...
    def supports(self, file_extension: str) -> bool:
        return file_extension.lower() == '.pdf'

class MarkdownParser(DocumentParser):
    def parse(self, file_path: Path) -> Document: ...
    def supports(self, file_extension: str) -> bool:
        return file_extension.lower() in ['.md', '.markdown']
```

**Applied to:**
- `DocumentParser` (PDF, Markdown, Text)
- `LLMProvider` (OpenAI, Azure OpenAI, Anthropic, Local)
- `EmbeddingProvider` (Sentence Transformers, OpenAI Embeddings)
- `OutputFormatter` (Markdown, JSON)

### 2.2 Factory Pattern

**Purpose**: Create appropriate instances without exposing creation logic.

```python
class ParserFactory:
    _parsers: Dict[str, Type[DocumentParser]] = {}
    
    @classmethod
    def register(cls, extension: str, parser_class: Type[DocumentParser]):
        cls._parsers[extension] = parser_class
    
    @classmethod
    def create(cls, file_path: Path) -> DocumentParser:
        extension = file_path.suffix.lower()
        if extension not in cls._parsers:
            raise UnsupportedFileTypeError(f"No parser for {extension}")
        return cls._parsers[extension]()

# Registration
ParserFactory.register('.pdf', PDFParser)
ParserFactory.register('.md', MarkdownParser)
ParserFactory.register('.txt', TextParser)
```

### 2.3 Builder Pattern

**Purpose**: Fluent configuration of complex Segmenta instances.

```python
class SegmentaBuilder:
    def __init__(self):
        self._config = SegmentaConfig()
        self._llm_provider = None
        self._embedding_provider = None
        self._output_formatter = None
    
    def with_config(self, config: SegmentaConfig) -> 'SegmentaBuilder':
        self._config = config
        return self
    
    def with_llm_provider(self, provider: LLMProvider) -> 'SegmentaBuilder':
        self._llm_provider = provider
        return self
    
    def with_embedding_provider(self, provider: EmbeddingProvider) -> 'SegmentaBuilder':
        self._embedding_provider = provider
        return self
    
    def with_output_formatter(self, formatter: OutputFormatter) -> 'SegmentaBuilder':
        self._output_formatter = formatter
        return self
    
    def build(self) -> 'Segmenta':
        return Segmenta(
            config=self._config,
            llm_provider=self._llm_provider or OpenAIProvider(),
            embedding_provider=self._embedding_provider or SentenceTransformerProvider(),
            output_formatter=self._output_formatter or MarkdownFormatter()
        )
```

### 2.4 Pipeline Pattern

**Purpose**: Process documents through a series of transformation stages.

```python
class Pipeline:
    def __init__(self):
        self._stages: List[PipelineStage] = []
    
    def add_stage(self, stage: PipelineStage) -> 'Pipeline':
        self._stages.append(stage)
        return self
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        for stage in self._stages:
            context = stage.process(context)
            if context.should_stop:
                break
        return context

class PipelineStage(ABC):
    @abstractmethod
    def process(self, context: PipelineContext) -> PipelineContext:
        pass
```

### 2.5 Template Method Pattern

**Purpose**: Define skeleton algorithm with customizable steps.

```python
class BaseChunkingStrategy(ABC):
    def chunk(self, document: Document) -> List[Chunk]:
        # Template method - defines the algorithm structure
        paragraphs = self.extract_paragraphs(document)
        boundaries = self.detect_boundaries(paragraphs)
        validated = self.validate_boundaries(boundaries, paragraphs)
        chunks = self.form_chunks(paragraphs, validated)
        enriched = self.enrich_chunks(chunks)
        return enriched
    
    @abstractmethod
    def detect_boundaries(self, paragraphs: List[Paragraph]) -> List[BoundaryProposal]:
        pass
    
    @abstractmethod
    def validate_boundaries(self, boundaries: List[BoundaryProposal], paragraphs: List[Paragraph]) -> List[BoundaryDecision]:
        pass
    
    # Default implementations for other steps
    def extract_paragraphs(self, document: Document) -> List[Paragraph]: ...
    def form_chunks(self, paragraphs: List[Paragraph], boundaries: List[BoundaryDecision]) -> List[Chunk]: ...
    def enrich_chunks(self, chunks: List[Chunk]) -> List[Chunk]: ...
```

---

## 3. Folder Structure

```
Segmenta/
│
├── __init__.py                     # Public API exports
├── Segmenta.py                       # Main Segmenta class (entry point)
├── config.py                       # Configuration dataclasses
├── exceptions.py                   # Custom exceptions
├── py.typed                        # PEP 561 marker for type hints
│
├── parsers/                        # Document Parsing Layer
│   ├── __init__.py                 # Export all parsers
│   ├── base.py                     # Abstract DocumentParser
│   ├── pdf_parser.py               # PDF parsing with PyMuPDF
│   ├── markdown_parser.py          # Markdown parsing with structure
│   ├── text_parser.py              # Plain text parsing
│   └── factory.py                  # Parser factory
│
├── segmenters/                     # Text Segmentation Layer
│   ├── __init__.py
│   ├── base.py                     # Abstract Segmenter
│   ├── structural.py               # Structural element detection
│   ├── paragraph.py                # Paragraph extraction
│   └── atomic.py                   # Atomic element handling
│
├── embeddings/                     # Embedding Layer (Stage 1)
│   ├── __init__.py
│   ├── base.py                     # Abstract EmbeddingProvider
│   ├── sentence_transformer.py     # Default ST implementation
│   └── similarity.py               # Similarity computation utilities
│
├── llm/                            # LLM Integration Layer (Stage 2)
│   ├── __init__.py
│   ├── base.py                     # Abstract LLMProvider
│   ├── openai_provider.py          # OpenAI implementation
│   ├── prompts/                    # Prompt templates
│   │   ├── __init__.py
│   │   ├── base.py                 # Abstract PromptTemplate
│   │   ├── validation.py           # Boundary validation prompts
│   │   └── enrichment.py           # Metadata enrichment prompts
│   └── factory.py                  # LLM provider factory
│
├── pipeline/                       # Orchestration Layer
│   ├── __init__.py
│   ├── base.py                     # Pipeline abstractions
│   ├── stages/                     # Pipeline stages
│   │   ├── __init__.py
│   │   ├── parse_stage.py
│   │   ├── segment_stage.py
│   │   ├── boundary_detect_stage.py
│   │   ├── boundary_validate_stage.py
│   │   ├── chunk_form_stage.py
│   │   ├── enrich_stage.py
│   │   └── output_stage.py
│   ├── context.py                  # Pipeline context object
│   └── orchestrator.py             # Main pipeline coordinator
│
├── models/                         # Data Models
│   ├── __init__.py                 # Export all models
│   ├── document.py                 # Document, Section, Paragraph
│   ├── chunk.py                    # Chunk, ChunkMetadata
│   ├── boundary.py                 # BoundaryProposal, BoundaryDecision
│   └── enums.py                    # Intent, ElementType enums
│
├── output/                         # Output Formatting Layer
│   ├── __init__.py
│   ├── base.py                     # Abstract OutputFormatter
│   └── markdown_formatter.py       # Default Markdown output
│
├── utils/                          # Utilities
│   ├── __init__.py
│   ├── token_counter.py            # Token counting with tiktoken
│   ├── validators.py               # Input validation helpers
│   ├── retry.py                    # Retry logic with backoff
│   └── logging.py                  # Logging configuration
│
└── cli/                            # Command Line Interface
    ├── __init__.py
    └── main.py                     # CLI entry point

tests/
├── __init__.py
├── conftest.py                     # Pytest fixtures
├── fixtures/                       # Test documents
│   ├── sample.pdf
│   ├── sample.md
│   └── sample.txt
├── unit/                           # Unit tests
│   ├── test_parsers/
│   ├── test_embeddings/
│   ├── test_llm/
│   ├── test_pipeline/
│   └── test_models/
├── integration/                    # Integration tests
│   └── test_full_pipeline.py
└── benchmark/                      # Performance tests
    └── test_performance.py

docs/
├── getting_started.md
├── api_reference.md
├── extending_Segmenta.md
└── examples/
    ├── basic_usage.py
    └── custom_provider.py

pyproject.toml                      # Project configuration
README.md                           # Project overview
LICENSE                             # License file
.github/
    └── workflows/
        └── ci.yml                  # CI/CD configuration
```

---

## 4. Data Models

### 4.1 Document Models

```python
# Segmenta/models/enums.py
from enum import Enum

class ElementType(str, Enum):
    PARAGRAPH = "paragraph"
    CODE_BLOCK = "code_block"
    TABLE = "table"
    LIST = "list"
    HEADER = "header"
    BLOCKQUOTE = "blockquote"

class Intent(str, Enum):
    EXPLAINS = "explains"
    LISTS = "lists"
    WARNS = "warns"
    DEFINES = "defines"
    INSTRUCTS = "instructs"
    COMPARES = "compares"
    UNKNOWN = "unknown"

class BoundaryVerdict(str, Enum):
    KEEP = "KEEP"
    MERGE = "MERGE"
    ADJUST = "ADJUST"
```

```python
# Segmenta/models/document.py
from dataclasses import dataclass, field
from typing import List, Optional
from .enums import ElementType

@dataclass
class Paragraph:
    """Represents a single paragraph or atomic element in a document."""
    text: str
    index: int
    element_type: ElementType = ElementType.PARAGRAPH
    is_atomic: bool = False  # True for code blocks, tables
    language: Optional[str] = None  # For code blocks
    
    @property
    def is_splittable(self) -> bool:
        return not self.is_atomic and self.element_type == ElementType.PARAGRAPH

@dataclass
class Section:
    """Represents a document section with optional hierarchy."""
    title: str
    level: int  # Header level (1-6 for markdown)
    paragraphs: List[Paragraph] = field(default_factory=list)
    subsections: List['Section'] = field(default_factory=list)
    
    @property
    def all_paragraphs(self) -> List[Paragraph]:
        """Recursively get all paragraphs including subsections."""
        result = list(self.paragraphs)
        for subsection in self.subsections:
            result.extend(subsection.all_paragraphs)
        return result

@dataclass
class Document:
    """Represents a parsed document."""
    filename: str
    file_type: str
    sections: List[Section] = field(default_factory=list)
    raw_text: str = ""
    
    @property
    def all_paragraphs(self) -> List[Paragraph]:
        """Get all paragraphs in document order."""
        result = []
        for section in self.sections:
            result.extend(section.all_paragraphs)
        return result
    
    @property
    def paragraph_count(self) -> int:
        return len(self.all_paragraphs)
```

### 4.2 Chunk Models

```python
# Segmenta/models/chunk.py
from dataclasses import dataclass, field
from typing import List, Optional
from .enums import Intent

@dataclass
class ChunkMetadata:
    """Metadata for a document chunk."""
    chunk_id: str
    title: str
    summary: str
    intent: Intent
    keywords: List[str]
    parent_section: str
    token_count: int
    
    def to_yaml_dict(self) -> dict:
        """Convert to dictionary for YAML serialization."""
        return {
            'chunk_id': self.chunk_id,
            'title': self.title,
            'summary': self.summary,
            'intent': self.intent.value,
            'keywords': self.keywords,
            'parent_section': self.parent_section,
            'token_count': self.token_count
        }

@dataclass
class Chunk:
    """A semantically coherent chunk of document content."""
    content: str
    metadata: Optional[ChunkMetadata] = None
    source_paragraphs: List[int] = field(default_factory=list)  # Original paragraph indices
    
    @property
    def is_enriched(self) -> bool:
        return self.metadata is not None
```

### 4.3 Boundary Models

```python
# Segmenta/models/boundary.py
from dataclasses import dataclass
from typing import Optional
from .document import Paragraph
from .enums import BoundaryVerdict

@dataclass
class BoundaryProposal:
    """A proposed chunk boundary from Stage 1 (embedding similarity)."""
    position: int  # Index in paragraph list where boundary should occur
    similarity_score: float  # Cosine similarity between adjacent paragraphs
    paragraph_before: Paragraph
    paragraph_after: Paragraph
    
    @property
    def context_window(self) -> str:
        """Get text context around the boundary for LLM validation."""
        return f"END OF CHUNK:\n{self.paragraph_before.text}\n\nSTART OF NEW CHUNK:\n{self.paragraph_after.text}"

@dataclass
class BoundaryDecision:
    """LLM decision on a boundary proposal from Stage 2."""
    proposal: BoundaryProposal
    verdict: BoundaryVerdict
    reason: str
    adjusted_position: Optional[int] = None  # Only if verdict is ADJUST
    confidence: float = 1.0  # LLM confidence if provided
    
    @property
    def final_position(self) -> Optional[int]:
        """Get the final boundary position after decision."""
        if self.verdict == BoundaryVerdict.MERGE:
            return None
        if self.verdict == BoundaryVerdict.ADJUST and self.adjusted_position is not None:
            return self.adjusted_position
        return self.proposal.position
```

---

## 5. Pipeline Architecture

### 5.1 Pipeline Context

```python
# Segmenta/pipeline/context.py
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict
from ..models import Document, Paragraph, Chunk, BoundaryProposal, BoundaryDecision
from ..config import SegmentaConfig

@dataclass
class PipelineContext:
    """Carries state through the pipeline stages."""
    # Input
    input_path: str
    output_dir: str
    config: SegmentaConfig
    
    # Stage outputs (populated as pipeline progresses)
    document: Optional[Document] = None
    paragraphs: List[Paragraph] = field(default_factory=list)
    boundary_proposals: List[BoundaryProposal] = field(default_factory=list)
    boundary_decisions: List[BoundaryDecision] = field(default_factory=list)
    chunks: List[Chunk] = field(default_factory=list)
    output_path: Optional[str] = None
    
    # Control flow
    should_stop: bool = False
    skip_boundary_detection: bool = False  # For short documents
    
    # Metrics
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Error tracking
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
```

### 5.2 Pipeline Stages

```python
# Segmenta/pipeline/base.py
from abc import ABC, abstractmethod
from .context import PipelineContext

class PipelineStage(ABC):
    """Base class for all pipeline stages."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Stage name for logging and metrics."""
        pass
    
    @abstractmethod
    def process(self, context: PipelineContext) -> PipelineContext:
        """Process the context and return updated context."""
        pass
    
    def should_skip(self, context: PipelineContext) -> bool:
        """Override to conditionally skip this stage."""
        return False
```

### 5.3 Stage Implementations

```python
# Segmenta/pipeline/stages/parse_stage.py
from ..base import PipelineStage
from ..context import PipelineContext
from ...parsers import ParserFactory

class ParseStage(PipelineStage):
    @property
    def name(self) -> str:
        return "parse"
    
    def process(self, context: PipelineContext) -> PipelineContext:
        parser = ParserFactory.create(context.input_path)
        context.document = parser.parse(context.input_path)
        context.metrics['parse_time'] = ...  # Track timing
        return context
```

```python
# Segmenta/pipeline/stages/boundary_detect_stage.py
from ..base import PipelineStage
from ..context import PipelineContext
from ...embeddings import EmbeddingProvider
from ...embeddings.similarity import compute_adjacent_similarities

class BoundaryDetectStage(PipelineStage):
    def __init__(self, embedding_provider: EmbeddingProvider, threshold: float = 0.5):
        self._embedding_provider = embedding_provider
        self._threshold = threshold
    
    @property
    def name(self) -> str:
        return "boundary_detect"
    
    def should_skip(self, context: PipelineContext) -> bool:
        # Skip for very short documents
        return context.skip_boundary_detection
    
    def process(self, context: PipelineContext) -> PipelineContext:
        # Get texts from non-atomic paragraphs
        texts = [p.text for p in context.paragraphs if p.is_splittable]
        
        # Compute embeddings
        embeddings = self._embedding_provider.embed(texts)
        
        # Compute similarities between adjacent paragraphs
        similarities = compute_adjacent_similarities(embeddings)
        
        # Propose boundaries where similarity drops below threshold
        for i, sim in enumerate(similarities):
            if sim < self._threshold:
                context.boundary_proposals.append(
                    BoundaryProposal(
                        position=i + 1,
                        similarity_score=sim,
                        paragraph_before=context.paragraphs[i],
                        paragraph_after=context.paragraphs[i + 1]
                    )
                )
        
        return context
```

### 5.4 Pipeline Orchestrator

```python
# Segmenta/pipeline/orchestrator.py
from typing import List, Optional, Callable
from .base import PipelineStage
from .context import PipelineContext
from ..config import SegmentaConfig
import logging

logger = logging.getLogger(__name__)

class PipelineOrchestrator:
    """Coordinates execution of pipeline stages."""
    
    def __init__(
        self,
        stages: List[PipelineStage],
        config: SegmentaConfig,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ):
        self._stages = stages
        self._config = config
        self._progress_callback = progress_callback
    
    def execute(self, input_path: str, output_dir: str) -> PipelineContext:
        context = PipelineContext(
            input_path=input_path,
            output_dir=output_dir,
            config=self._config
        )
        
        total_stages = len(self._stages)
        
        for i, stage in enumerate(self._stages):
            if context.should_stop:
                logger.warning(f"Pipeline stopped early at stage: {stage.name}")
                break
            
            if stage.should_skip(context):
                logger.info(f"Skipping stage: {stage.name}")
                continue
            
            logger.info(f"Executing stage: {stage.name}")
            
            try:
                context = stage.process(context)
            except Exception as e:
                logger.error(f"Error in stage {stage.name}: {e}")
                context.errors.append(f"{stage.name}: {str(e)}")
                if not self._config.continue_on_error:
                    context.should_stop = True
            
            if self._progress_callback:
                progress = (i + 1) / total_stages
                self._progress_callback(stage.name, progress)
        
        return context
```

### 5.5 Pipeline Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           Segmenta PIPELINE                                     │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: PARSE                                                               │
│ ───────────────                                                              │
│ Input:  File path (PDF/MD/TXT)                                               │
│ Action: Factory selects parser → Extract text and structure                  │
│ Output: Document model with Sections and Paragraphs                          │
│ Skip:   Never                                                                │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: SEGMENT                                                             │
│ ────────────────                                                             │
│ Input:  Document model                                                       │
│ Action: Flatten sections → Mark atomic elements → Number paragraphs          │
│ Output: Ordered list of Paragraphs with metadata                             │
│ Skip:   Never                                                                │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: BOUNDARY DETECTION (Embedding-based)                                │
│ ──────────────────────────────────────────────                               │
│ Input:  List of Paragraphs                                                   │
│ Action: Embed paragraphs → Compute adjacent similarity → Threshold filter    │
│ Output: List of BoundaryProposal objects                                     │
│ Skip:   If document has < 5 paragraphs                                       │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ STAGE 4: BOUNDARY VALIDATION (LLM-based)                                     │
│ ────────────────────────────────────────                                     │
│ Input:  List of BoundaryProposal objects                                     │
│ Action: For each proposal → Send context to LLM → Get verdict                │
│ Output: List of BoundaryDecision objects (KEEP/MERGE/ADJUST)                 │
│ Skip:   If no boundary proposals                                             │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ STAGE 5: CHUNK FORMATION                                                     │
│ ────────────────────────                                                     │
│ Input:  Paragraphs + BoundaryDecisions                                       │
│ Action: Group paragraphs by validated boundaries → Create raw chunks         │
│ Output: List of Chunk objects (content only, no metadata yet)                │
│ Skip:   Never                                                                │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ STAGE 6: ENRICHMENT (LLM-based)                                              │
│ ───────────────────────────────                                              │
│ Input:  List of raw Chunks                                                   │
│ Action: For each chunk → Send to LLM → Extract metadata                      │
│ Output: List of Chunks with ChunkMetadata (title, summary, intent, etc.)     │
│ Skip:   Never                                                                │
└──────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ STAGE 7: OUTPUT                                                              │
│ ───────────────                                                              │
│ Input:  List of enriched Chunks                                              │
│ Action: Format with YAML frontmatter → Write to output file                  │
│ Output: Segmenta_output_<filename>.md                                          │
│ Skip:   Never                                                                │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Public API Design

### 6.1 Simple API

```python
# Most common usage - minimal configuration
from Segmenta import Segmenta

# Initialize with defaults
Segmenta = Segmenta(
    openai_api_key="sk-...",
    model="gpt-4o"
)

# Process a single file
result = Segmenta.chunk(
    input_file="document.pdf",
    output_dir="./output"
)

print(f"Created: {result.output_path}")
print(f"Chunks generated: {len(result.chunks)}")
```

### 6.2 Advanced API with Builder

```python
from Segmenta import Segmenta, SegmentaConfig
from Segmenta.llm import OpenAIProvider
from Segmenta.embeddings import SentenceTransformerProvider
from Segmenta.output import MarkdownFormatter

# Custom configuration
config = SegmentaConfig(
    similarity_threshold=0.5,      # Embedding similarity threshold for boundaries
    min_chunk_tokens=50,           # Minimum tokens per chunk
    max_chunk_tokens=500,          # Maximum tokens per chunk (soft limit)
    retry_attempts=3,              # LLM retry attempts
    fallback_enabled=True,         # Use fallback on LLM failure
    continue_on_error=False,       # Stop pipeline on error
    verbose=True                   # Enable detailed logging
)

# Build with custom components
Segmenta = (
    Segmenta.builder()
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
            device="cuda"  # Use GPU if available
        )
    )
    .with_output_formatter(MarkdownFormatter())
    .build()
)

# Process with progress callback
def on_progress(stage: str, progress: float):
    print(f"[{progress:.0%}] {stage}")

result = Segmenta.chunk(
    input_file="document.pdf",
    output_dir="./output",
    progress_callback=on_progress
)
```

### 6.3 Configuration Dataclass

```python
# Segmenta/config.py
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class SegmentaConfig:
    """Configuration for Segmenta processing."""
    
    # Boundary detection
    similarity_threshold: float = 0.5
    """Similarity score below which a boundary is proposed (0.0 - 1.0)"""
    
    # Chunk size constraints
    min_chunk_tokens: int = 50
    """Minimum tokens per chunk. Smaller chunks will be merged."""
    
    max_chunk_tokens: int = 500
    """Maximum tokens per chunk (soft limit). Atomic elements may exceed."""
    
    # LLM behavior
    retry_attempts: int = 2
    """Number of retry attempts for LLM calls."""
    
    retry_delay: float = 1.0
    """Delay between retries in seconds."""
    
    fallback_enabled: bool = True
    """Use fallback metadata when LLM fails."""
    
    # Pipeline behavior
    continue_on_error: bool = False
    """Continue pipeline execution on stage errors."""
    
    # Logging
    verbose: bool = False
    """Enable verbose logging output."""
    
    # Short document handling
    short_document_threshold: int = 5
    """Documents with fewer paragraphs skip boundary detection."""
    
    # Embedding model (if using default provider)
    embedding_model: str = "all-MiniLM-L6-v2"
    """Sentence Transformer model name."""
    
    # Token counting
    token_model: str = "gpt-4"
    """Model name for tiktoken token counting."""
```

### 6.4 Result Object

```python
# Segmenta/models/result.py
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from .chunk import Chunk

@dataclass
class SegmentaResult:
    """Result of chunking operation."""
    
    success: bool
    """Whether chunking completed successfully."""
    
    output_path: Optional[str]
    """Path to the generated output file."""
    
    chunks: List[Chunk]
    """List of generated chunks with metadata."""
    
    metrics: Dict[str, Any] = field(default_factory=dict)
    """Processing metrics (timing, counts, etc.)."""
    
    warnings: List[str] = field(default_factory=list)
    """Non-fatal warnings during processing."""
    
    errors: List[str] = field(default_factory=list)
    """Errors encountered during processing."""
    
    @property
    def chunk_count(self) -> int:
        return len(self.chunks)
    
    @property
    def total_tokens(self) -> int:
        return sum(c.metadata.token_count for c in self.chunks if c.metadata)
```

---

## 7. Component Specifications

### 7.1 Document Parsers

#### Abstract Base

```python
# Segmenta/parsers/base.py
from abc import ABC, abstractmethod
from pathlib import Path
from ..models import Document

class DocumentParser(ABC):
    """Abstract base class for document parsers."""
    
    @abstractmethod
    def parse(self, file_path: Path) -> Document:
        """Parse a document file into a Document model.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Parsed Document model
            
        Raises:
            SegmentaParseError: If parsing fails
        """
        pass
    
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """Return list of supported file extensions."""
        pass
```

#### PDF Parser

```python
# Segmenta/parsers/pdf_parser.py
import fitz  # PyMuPDF
from pathlib import Path
from .base import DocumentParser
from ..models import Document, Section, Paragraph, ElementType
from ..exceptions import SegmentaParseError

class PDFParser(DocumentParser):
    """Parser for PDF documents using PyMuPDF."""
    
    def supported_extensions(self) -> List[str]:
        return ['.pdf']
    
    def parse(self, file_path: Path) -> Document:
        try:
            doc = fitz.open(file_path)
            sections = []
            paragraph_index = 0
            
            for page_num, page in enumerate(doc):
                # Extract text blocks
                blocks = page.get_text("dict")["blocks"]
                
                for block in blocks:
                    if block["type"] == 0:  # Text block
                        text = self._extract_block_text(block)
                        if text.strip():
                            # Detect if this might be a header
                            is_header = self._is_likely_header(block)
                            
                            if is_header:
                                sections.append(Section(
                                    title=text.strip(),
                                    level=self._estimate_header_level(block),
                                    paragraphs=[]
                                ))
                            else:
                                paragraph = Paragraph(
                                    text=text.strip(),
                                    index=paragraph_index,
                                    element_type=ElementType.PARAGRAPH
                                )
                                paragraph_index += 1
                                
                                if sections:
                                    sections[-1].paragraphs.append(paragraph)
                                else:
                                    # Create implicit section for content before first header
                                    sections.append(Section(
                                        title="",
                                        level=0,
                                        paragraphs=[paragraph]
                                    ))
            
            doc.close()
            
            return Document(
                filename=file_path.name,
                file_type="pdf",
                sections=sections,
                raw_text=self._get_raw_text(file_path)
            )
            
        except Exception as e:
            raise SegmentaParseError(f"Failed to parse PDF: {e}") from e
    
    def _extract_block_text(self, block: dict) -> str:
        """Extract text from a PDF block."""
        lines = []
        for line in block.get("lines", []):
            line_text = " ".join(span["text"] for span in line.get("spans", []))
            lines.append(line_text)
        return " ".join(lines)
    
    def _is_likely_header(self, block: dict) -> bool:
        """Heuristic to detect headers based on font size and style."""
        # Implementation based on font analysis
        ...
    
    def _estimate_header_level(self, block: dict) -> int:
        """Estimate header level based on font size."""
        ...
```

#### Markdown Parser

```python
# Segmenta/parsers/markdown_parser.py
from markdown_it import MarkdownIt
from pathlib import Path
from .base import DocumentParser
from ..models import Document, Section, Paragraph, ElementType
from ..exceptions import SegmentaParseError

class MarkdownParser(DocumentParser):
    """Parser for Markdown documents."""
    
    def __init__(self):
        self._md = MarkdownIt()
    
    def supported_extensions(self) -> List[str]:
        return ['.md', '.markdown']
    
    def parse(self, file_path: Path) -> Document:
        try:
            content = file_path.read_text(encoding='utf-8')
            tokens = self._md.parse(content)
            
            sections = []
            current_section = None
            section_stack = []  # For nested sections
            paragraph_index = 0
            
            i = 0
            while i < len(tokens):
                token = tokens[i]
                
                if token.type == 'heading_open':
                    level = int(token.tag[1])  # h1 -> 1, h2 -> 2, etc.
                    # Get heading content from next token
                    heading_content = tokens[i + 1].content if i + 1 < len(tokens) else ""
                    
                    new_section = Section(
                        title=heading_content,
                        level=level,
                        paragraphs=[]
                    )
                    
                    # Handle section hierarchy
                    self._add_section_to_hierarchy(sections, section_stack, new_section, level)
                    current_section = new_section
                    i += 3  # Skip heading_open, inline, heading_close
                    continue
                
                elif token.type == 'paragraph_open':
                    para_content = tokens[i + 1].content if i + 1 < len(tokens) else ""
                    paragraph = Paragraph(
                        text=para_content,
                        index=paragraph_index,
                        element_type=ElementType.PARAGRAPH
                    )
                    paragraph_index += 1
                    self._add_paragraph(sections, current_section, paragraph)
                    i += 3  # Skip paragraph_open, inline, paragraph_close
                    continue
                
                elif token.type == 'fence':  # Code block
                    paragraph = Paragraph(
                        text=token.content,
                        index=paragraph_index,
                        element_type=ElementType.CODE_BLOCK,
                        is_atomic=True,
                        language=token.info  # Language hint
                    )
                    paragraph_index += 1
                    self._add_paragraph(sections, current_section, paragraph)
                
                elif token.type == 'table_open':
                    table_content = self._extract_table(tokens, i)
                    paragraph = Paragraph(
                        text=table_content,
                        index=paragraph_index,
                        element_type=ElementType.TABLE,
                        is_atomic=True
                    )
                    paragraph_index += 1
                    self._add_paragraph(sections, current_section, paragraph)
                    # Skip to table_close
                    while i < len(tokens) and tokens[i].type != 'table_close':
                        i += 1
                
                i += 1
            
            return Document(
                filename=file_path.name,
                file_type="markdown",
                sections=sections,
                raw_text=content
            )
            
        except Exception as e:
            raise SegmentaParseError(f"Failed to parse Markdown: {e}") from e
```

### 7.2 Embedding Provider

```python
# Segmenta/embeddings/base.py
from abc import ABC, abstractmethod
from typing import List
import numpy as np

class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        pass
    
    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the dimensionality of embeddings."""
        pass

# Segmenta/embeddings/sentence_transformer.py
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import numpy as np
from .base import EmbeddingProvider

class SentenceTransformerProvider(EmbeddingProvider):
    """Embedding provider using Sentence Transformers."""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        batch_size: int = 32
    ):
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._model = None  # Lazy loading
    
    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the model on first use."""
        if self._model is None:
            self._model = SentenceTransformer(self._model_name, device=self._device)
        return self._model
    
    def embed(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
    
    @property
    def embedding_dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()
```

### 7.3 LLM Provider

```python
# Segmenta/llm/base.py
from abc import ABC, abstractmethod
from typing import Optional
from dataclasses import dataclass

@dataclass
class LLMResponse:
    """Response from an LLM call."""
    content: str
    tokens_used: int
    model: str
    success: bool = True
    error: Optional[str] = None

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def complete(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Generate a completion for the given prompt.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            
        Returns:
            LLMResponse with the generated content
        """
        pass
    
    @abstractmethod
    def complete_json(self, prompt: str, system_prompt: Optional[str] = None) -> dict:
        """Generate a JSON-structured completion.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            
        Returns:
            Parsed JSON response as a dictionary
        """
        pass

# Segmenta/llm/openai_provider.py
from openai import OpenAI
from typing import Optional
import json
from .base import LLMProvider, LLMResponse
from ..exceptions import SegmentaLLMError

class OpenAIProvider(LLMProvider):
    """LLM provider using OpenAI API."""
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        temperature: float = 0.1,
        max_tokens: int = 1000
    ):
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
    
    def complete(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens
            )
            
            return LLMResponse(
                content=response.choices[0].message.content,
                tokens_used=response.usage.total_tokens,
                model=self._model
            )
        except Exception as e:
            return LLMResponse(
                content="",
                tokens_used=0,
                model=self._model,
                success=False,
                error=str(e)
            )
    
    def complete_json(self, prompt: str, system_prompt: Optional[str] = None) -> dict:
        response = self.complete(prompt, system_prompt)
        if not response.success:
            raise SegmentaLLMError(f"LLM call failed: {response.error}")
        
        try:
            return json.loads(response.content)
        except json.JSONDecodeError as e:
            raise SegmentaLLMError(f"Failed to parse JSON response: {e}")
```

### 7.4 Prompt Templates

```python
# Segmenta/llm/prompts/validation.py

BOUNDARY_VALIDATION_SYSTEM = """You are an expert at analyzing document structure and logical coherence.
Your task is to determine if a proposed chunk boundary between two text segments is appropriate.
A good boundary separates distinct logical units or topics.
A bad boundary splits a continuous thought or related concepts."""

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
"""

# Segmenta/llm/prompts/enrichment.py

ENRICHMENT_SYSTEM = """You are an expert at analyzing text and extracting structured metadata.
Your task is to extract key information that summarizes and categorizes a chunk of text."""

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
- Keywords: Terms someone might search for to find this content
"""
```

---

## 8. Error Handling Strategy

### 8.1 Custom Exceptions

```python
# Segmenta/exceptions.py

class SegmentaError(Exception):
    """Base exception for all Segmenta errors."""
    pass

class SegmentaConfigError(SegmentaError):
    """Raised when configuration is invalid."""
    pass

class SegmentaParseError(SegmentaError):
    """Raised when document parsing fails."""
    def __init__(self, message: str, file_path: str = None, position: int = None):
        super().__init__(message)
        self.file_path = file_path
        self.position = position

class SegmentaLLMError(SegmentaError):
    """Raised when LLM operations fail."""
    def __init__(self, message: str, retries_attempted: int = 0):
        super().__init__(message)
        self.retries_attempted = retries_attempted

class SegmentaEmbeddingError(SegmentaError):
    """Raised when embedding operations fail."""
    pass

class UnsupportedFileTypeError(SegmentaError):
    """Raised when file type is not supported."""
    def __init__(self, file_type: str):
        super().__init__(f"Unsupported file type: {file_type}")
        self.file_type = file_type
```

### 8.2 Retry Logic

```python
# Segmenta/utils/retry.py
import time
import logging
from typing import TypeVar, Callable, Optional
from functools import wraps

T = TypeVar('T')
logger = logging.getLogger(__name__)

def retry_with_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exceptions: tuple = (Exception,)
):
    """Decorator for retry logic with exponential backoff."""
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        logger.warning(
                            f"Attempt {attempt + 1} failed: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
            
            raise last_exception
        
        return wrapper
    return decorator

class RetryHandler:
    """Handler for retry logic with callbacks."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        on_retry: Optional[Callable[[int, Exception], None]] = None
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.on_retry = on_retry
    
    def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(self.max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.max_attempts - 1:
                    if self.on_retry:
                        self.on_retry(attempt, e)
                    time.sleep(self.base_delay * (2 ** attempt))
        
        raise last_exception
```

### 8.3 Fallback Strategy

```python
# Segmenta/pipeline/stages/enrich_stage.py

class EnrichStage(PipelineStage):
    def __init__(
        self,
        llm_provider: LLMProvider,
        retry_handler: RetryHandler,
        fallback_enabled: bool = True
    ):
        self._llm = llm_provider
        self._retry = retry_handler
        self._fallback_enabled = fallback_enabled
    
    def enrich_chunk(self, chunk: Chunk, parent_section: str) -> ChunkMetadata:
        try:
            return self._retry.execute(
                self._enrich_with_llm,
                chunk,
                parent_section
            )
        except Exception as e:
            if self._fallback_enabled:
                logger.warning(f"LLM enrichment failed, using fallback: {e}")
                return self._fallback_metadata(chunk, parent_section)
            raise
    
    def _fallback_metadata(self, chunk: Chunk, parent_section: str) -> ChunkMetadata:
        """Generate fallback metadata when LLM fails."""
        # Extract first sentence as title
        sentences = chunk.content.split('. ')
        title = sentences[0][:50] + "..." if len(sentences[0]) > 50 else sentences[0]
        
        return ChunkMetadata(
            chunk_id=self._generate_chunk_id(),
            title=title,
            summary="Content summary unavailable",
            intent=Intent.UNKNOWN,
            keywords=self._extract_simple_keywords(chunk.content),
            parent_section=parent_section,
            token_count=self._count_tokens(chunk.content)
        )
    
    def _extract_simple_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """Simple keyword extraction as fallback."""
        # Remove common words and extract frequent terms
        # This is a simplified implementation
        words = text.lower().split()
        word_freq = {}
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being'}
        
        for word in words:
            word = ''.join(c for c in word if c.isalnum())
            if word and word not in stop_words and len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:max_keywords]]
```

---

## 9. Edge Case Handling

### 9.1 Edge Case Matrix

| Scenario | Detection | Handling |
|----------|-----------|----------|
| Very short document (<5 paragraphs) | `document.paragraph_count < config.short_document_threshold` | Skip boundary detection, single chunk |
| Empty document | `document.paragraph_count == 0` | Return empty result with warning |
| LLM returns invalid JSON | JSON parse exception | Retry, then fallback |
| LLM returns invalid verdict | Verdict not in enum | Default to KEEP |
| No semantic boundaries found | `len(boundary_proposals) == 0` | Use structural boundaries only |
| Single very long paragraph | Paragraph exceeds `max_chunk_tokens` | Sentence-level splitting |
| Code block exceeds max tokens | Atomic element too large | Keep whole (never split code) |
| Deeply nested sections | Section level > 6 | Flatten to level 6 |
| Empty sections | Section has no paragraphs | Skip, attach header to next content |
| Mixed language content | Non-English text | Process as-is (v1 limitation) |
| Malformed Markdown | Parse errors | Best-effort parsing with warnings |
| PDF with images only | No extractable text | Return empty with error |
| Rate limit exceeded | API returns 429 | Exponential backoff, max 5 retries |

### 9.2 Short Document Handling

```python
# In segment_stage.py

class SegmentStage(PipelineStage):
    def process(self, context: PipelineContext) -> PipelineContext:
        paragraphs = context.document.all_paragraphs
        context.paragraphs = paragraphs
        
        # Check for short document
        if len(paragraphs) < context.config.short_document_threshold:
            context.skip_boundary_detection = True
            context.warnings.append(
                f"Short document ({len(paragraphs)} paragraphs). "
                "Returning as single chunk."
            )
        
        return context
```

### 9.3 Long Paragraph Splitting

```python
# Segmenta/segmenters/paragraph.py

class ParagraphSplitter:
    """Handles splitting of very long paragraphs at sentence boundaries."""
    
    def __init__(self, max_tokens: int, token_counter: TokenCounter):
        self._max_tokens = max_tokens
        self._token_counter = token_counter
    
    def split_if_needed(self, paragraph: Paragraph) -> List[Paragraph]:
        """Split paragraph if it exceeds max tokens."""
        if paragraph.is_atomic:
            # Never split atomic elements
            return [paragraph]
        
        token_count = self._token_counter.count(paragraph.text)
        if token_count <= self._max_tokens:
            return [paragraph]
        
        # Split at sentence boundaries
        return self._split_by_sentences(paragraph)
    
    def _split_by_sentences(self, paragraph: Paragraph) -> List[Paragraph]:
        """Split paragraph into sentence groups that fit within token limit."""
        import re
        
        # Simple sentence splitting (could use nltk for better accuracy)
        sentences = re.split(r'(?<=[.!?])\s+', paragraph.text)
        
        result = []
        current_sentences = []
        current_tokens = 0
        base_index = paragraph.index
        
        for sentence in sentences:
            sentence_tokens = self._token_counter.count(sentence)
            
            if current_tokens + sentence_tokens > self._max_tokens and current_sentences:
                # Create paragraph from current sentences
                result.append(Paragraph(
                    text=' '.join(current_sentences),
                    index=base_index + len(result),
                    element_type=paragraph.element_type
                ))
                current_sentences = [sentence]
                current_tokens = sentence_tokens
            else:
                current_sentences.append(sentence)
                current_tokens += sentence_tokens
        
        # Add remaining sentences
        if current_sentences:
            result.append(Paragraph(
                text=' '.join(current_sentences),
                index=base_index + len(result),
                element_type=paragraph.element_type
            ))
        
        return result
```

---

## 10. Dependencies

### 10.1 Core Dependencies

```toml
# pyproject.toml
[project]
name = "Segmenta"
version = "1.0.0"
description = "Semantic document chunking library"
requires-python = ">=3.9"
dependencies = [
    # Embedding
    "sentence-transformers>=2.2.0",
    "torch>=2.0.0",
    
    # LLM
    "openai>=1.0.0",
    
    # Token counting
    "tiktoken>=0.5.0",
    
    # Document parsing
    "pymupdf>=1.23.0",       # PDF parsing
    "markdown-it-py>=3.0.0", # Markdown parsing
    
    # Data validation
    "pydantic>=2.0.0",
    
    # Utilities
    "python-dotenv>=1.0.0",
    "rich>=13.0.0",          # CLI output
    "click>=8.0.0",          # CLI framework
    
    # Type hints
    "typing-extensions>=4.0.0",
]

[project.optional-dependencies]
dev = [
    # Testing
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.0.0",
    "pytest-mock>=3.10.0",
    "hypothesis>=6.0.0",     # Property-based testing
    
    # Code quality
    "ruff>=0.1.0",
    "mypy>=1.0.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    
    # Documentation
    "mkdocs>=1.5.0",
    "mkdocs-material>=9.0.0",
    "mkdocstrings[python]>=0.24.0",
]

azure = [
    "openai[azure]>=1.0.0",  # Azure OpenAI support
]

all = [
    "Segmenta[dev,azure]",
]

[project.scripts]
Segmenta = "Segmenta.cli.main:cli"
```

### 10.2 Dependency Justification

| Dependency | Purpose | Alternatives Considered |
|------------|---------|------------------------|
| `sentence-transformers` | Embedding generation | `openai` embeddings (more expensive), `fastembed` (less accurate) |
| `pymupdf` | PDF parsing | `pdfplumber` (slower), `pypdf` (less accurate) |
| `markdown-it-py` | Markdown parsing | `mistune` (less compliant), `commonmark` (slower) |
| `tiktoken` | Token counting | Manual estimation (less accurate) |
| `pydantic` | Data validation | `dataclasses` + manual validation (more code) |
| `click` | CLI framework | `argparse` (more verbose), `typer` (additional dependency) |
| `rich` | CLI output | `colorama` (less features) |

---

## 11. Testing Strategy

### 11.1 Test Structure

```
tests/
├── conftest.py                 # Shared fixtures
├── fixtures/
│   ├── documents/
│   │   ├── simple.md
│   │   ├── complex.md
│   │   ├── with_code.md
│   │   ├── with_tables.md
│   │   ├── short.txt
│   │   └── sample.pdf
│   └── expected/
│       └── simple_chunks.json
├── unit/
│   ├── test_parsers/
│   │   ├── test_markdown_parser.py
│   │   ├── test_pdf_parser.py
│   │   └── test_text_parser.py
│   ├── test_embeddings/
│   │   ├── test_sentence_transformer.py
│   │   └── test_similarity.py
│   ├── test_llm/
│   │   ├── test_openai_provider.py
│   │   └── test_prompts.py
│   ├── test_pipeline/
│   │   ├── test_stages.py
│   │   └── test_orchestrator.py
│   └── test_models/
│       ├── test_document.py
│       ├── test_chunk.py
│       └── test_boundary.py
├── integration/
│   ├── test_full_pipeline.py
│   ├── test_error_recovery.py
│   └── test_edge_cases.py
└── benchmark/
    ├── test_performance.py
    └── test_memory.py
```

### 11.2 Test Fixtures

```python
# tests/conftest.py
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock
from Segmenta import SegmentaConfig
from Segmenta.models import Document, Section, Paragraph, ElementType

@pytest.fixture
def sample_config() -> SegmentaConfig:
    return SegmentaConfig(
        similarity_threshold=0.5,
        min_chunk_tokens=50,
        max_chunk_tokens=500,
        retry_attempts=1,
        fallback_enabled=True
    )

@pytest.fixture
def sample_document() -> Document:
    return Document(
        filename="test.md",
        file_type="markdown",
        sections=[
            Section(
                title="Introduction",
                level=1,
                paragraphs=[
                    Paragraph(text="This is the first paragraph.", index=0),
                    Paragraph(text="This is the second paragraph.", index=1),
                ]
            ),
            Section(
                title="Details",
                level=1,
                paragraphs=[
                    Paragraph(text="Here are some details.", index=2),
                    Paragraph(
                        text="```python\nprint('hello')\n```",
                        index=3,
                        element_type=ElementType.CODE_BLOCK,
                        is_atomic=True
                    ),
                ]
            )
        ]
    )

@pytest.fixture
def mock_llm_provider():
    mock = Mock()
    mock.complete_json.return_value = {
        "verdict": "KEEP",
        "reason": "Different topics",
        "confidence": 0.9
    }
    return mock

@pytest.fixture
def mock_embedding_provider():
    import numpy as np
    mock = Mock()
    mock.embed.return_value = np.random.rand(10, 384)
    mock.embedding_dim = 384
    return mock

@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"
```

### 11.3 Unit Test Examples

```python
# tests/unit/test_parsers/test_markdown_parser.py
import pytest
from Segmenta.parsers import MarkdownParser
from Segmenta.models import ElementType

class TestMarkdownParser:
    
    @pytest.fixture
    def parser(self):
        return MarkdownParser()
    
    def test_parse_simple_document(self, parser, fixtures_dir):
        doc = parser.parse(fixtures_dir / "documents/simple.md")
        
        assert doc.filename == "simple.md"
        assert doc.file_type == "markdown"
        assert len(doc.sections) > 0
    
    def test_parse_preserves_headers(self, parser, fixtures_dir):
        doc = parser.parse(fixtures_dir / "documents/simple.md")
        
        # Check that headers become sections
        section_titles = [s.title for s in doc.sections]
        assert "Introduction" in section_titles
    
    def test_parse_identifies_code_blocks(self, parser, fixtures_dir):
        doc = parser.parse(fixtures_dir / "documents/with_code.md")
        
        code_blocks = [p for p in doc.all_paragraphs 
                       if p.element_type == ElementType.CODE_BLOCK]
        
        assert len(code_blocks) > 0
        assert all(p.is_atomic for p in code_blocks)
    
    def test_parse_identifies_tables(self, parser, fixtures_dir):
        doc = parser.parse(fixtures_dir / "documents/with_tables.md")
        
        tables = [p for p in doc.all_paragraphs 
                  if p.element_type == ElementType.TABLE]
        
        assert len(tables) > 0
        assert all(p.is_atomic for p in tables)
    
    def test_parse_handles_empty_file(self, parser, tmp_path):
        empty_file = tmp_path / "empty.md"
        empty_file.write_text("")
        
        doc = parser.parse(empty_file)
        
        assert doc.paragraph_count == 0
```

### 11.4 Integration Tests

```python
# tests/integration/test_full_pipeline.py
import pytest
from Segmenta import Segmenta, SegmentaConfig

class TestFullPipeline:
    
    @pytest.fixture
    def Segmenta_instance(self, mock_llm_provider, mock_embedding_provider):
        return (
            Segmenta.builder()
            .with_config(SegmentaConfig(fallback_enabled=True))
            .with_llm_provider(mock_llm_provider)
            .with_embedding_provider(mock_embedding_provider)
            .build()
        )
    
    def test_end_to_end_markdown(self, Segmenta_instance, fixtures_dir, tmp_path):
        result = Segmenta_instance.chunk(
            input_file=str(fixtures_dir / "documents/simple.md"),
            output_dir=str(tmp_path)
        )
        
        assert result.success
        assert result.chunk_count > 0
        assert result.output_path is not None
        assert Path(result.output_path).exists()
    
    def test_short_document_single_chunk(self, Segmenta_instance, fixtures_dir, tmp_path):
        result = Segmenta_instance.chunk(
            input_file=str(fixtures_dir / "documents/short.txt"),
            output_dir=str(tmp_path)
        )
        
        assert result.success
        assert result.chunk_count == 1
    
    def test_preserves_atomic_elements(self, Segmenta_instance, fixtures_dir, tmp_path):
        result = Segmenta_instance.chunk(
            input_file=str(fixtures_dir / "documents/with_code.md"),
            output_dir=str(tmp_path)
        )
        
        # Verify code blocks are not split
        for chunk in result.chunks:
            if "```" in chunk.content:
                # Code block should be complete
                assert chunk.content.count("```") % 2 == 0
```

### 11.5 Property-Based Tests

```python
# tests/unit/test_models/test_boundary.py
from hypothesis import given, strategies as st
from Segmenta.models import BoundaryProposal, BoundaryDecision, BoundaryVerdict, Paragraph

class TestBoundaryDecision:
    
    @given(
        verdict=st.sampled_from(list(BoundaryVerdict)),
        position=st.integers(min_value=0, max_value=1000),
        adjusted=st.integers(min_value=0, max_value=1000)
    )
    def test_final_position_logic(self, verdict, position, adjusted):
        proposal = BoundaryProposal(
            position=position,
            similarity_score=0.5,
            paragraph_before=Paragraph(text="before", index=position-1),
            paragraph_after=Paragraph(text="after", index=position)
        )
        
        decision = BoundaryDecision(
            proposal=proposal,
            verdict=verdict,
            reason="test",
            adjusted_position=adjusted if verdict == BoundaryVerdict.ADJUST else None
        )
        
        if verdict == BoundaryVerdict.MERGE:
            assert decision.final_position is None
        elif verdict == BoundaryVerdict.ADJUST:
            assert decision.final_position == adjusted
        else:  # KEEP
            assert decision.final_position == position
```

---

## 12. Implementation Phases

### Phase 1: Foundation (Week 1)

**Goals:**
- Project scaffolding
- Core data models
- Base abstractions
- Configuration system

**Tasks:**
1. Initialize project with `pyproject.toml`
2. Set up folder structure
3. Implement all data models (`Document`, `Chunk`, `Boundary`)
4. Implement `SegmentaConfig` dataclass
5. Create custom exceptions
6. Set up logging infrastructure
7. Write model unit tests

**Deliverables:**
- Working project structure
- All data models with tests
- Configuration system

### Phase 2: Document Parsing (Week 2)

**Goals:**
- Implement all document parsers
- Parser factory pattern
- Structural element detection

**Tasks:**
1. Implement `DocumentParser` abstract base
2. Implement `TextParser` (simplest case)
3. Implement `MarkdownParser` with structure preservation
4. Implement `PDFParser` with PyMuPDF
5. Create `ParserFactory`
6. Write comprehensive parser tests
7. Test with various document samples

**Deliverables:**
- Working parsers for PDF, MD, TXT
- Factory for parser selection
- Test coverage > 90%

### Phase 3: Embedding & Similarity (Week 3)

**Goals:**
- Embedding provider implementation
- Similarity computation
- Boundary detection logic

**Tasks:**
1. Implement `EmbeddingProvider` abstract base
2. Implement `SentenceTransformerProvider`
3. Implement similarity computation utilities
4. Implement `BoundaryDetector`
5. Tune default similarity threshold
6. Write embedding and similarity tests
7. Performance benchmarks for embedding

**Deliverables:**
- Working embedding system
- Boundary detection with configurable threshold
- Performance baseline

### Phase 4: LLM Integration (Week 4)

**Goals:**
- LLM provider implementation
- Prompt templates
- Retry and fallback logic

**Tasks:**
1. Implement `LLMProvider` abstract base
2. Implement `OpenAIProvider`
3. Create prompt templates for validation
4. Create prompt templates for enrichment
5. Implement retry logic with backoff
6. Implement fallback metadata generation
7. Write LLM integration tests (mocked)

**Deliverables:**
- Working LLM integration
- Robust error handling
- Configurable retry behavior

### Phase 5: Pipeline Orchestration (Week 5)

**Goals:**
- Complete pipeline implementation
- All stages integrated
- Edge case handling

**Tasks:**
1. Implement `PipelineContext`
2. Implement all pipeline stages
3. Implement `PipelineOrchestrator`
4. Handle edge cases (short docs, long paragraphs)
5. Implement progress callbacks
6. Write integration tests
7. Test error recovery scenarios

**Deliverables:**
- Full working pipeline
- All edge cases handled
- Integration test coverage

### Phase 6: Output & Polish (Week 6)

**Goals:**
- Output formatting
- CLI tool
- Documentation
- Final polish

**Tasks:**
1. Implement `MarkdownFormatter`
2. Create CLI with Click
3. Write user documentation
4. Write API reference docs
5. Create usage examples
6. Performance optimization
7. Final test coverage review
8. Package and publish setup

**Deliverables:**
- CLI tool
- Complete documentation
- Published package (TestPyPI)

---

## 13. Extensibility Guide

### 13.1 Adding a New Document Parser

```python
# my_parsers/docx_parser.py
from Segmenta.parsers.base import DocumentParser
from Segmenta.parsers import ParserFactory
from Segmenta.models import Document
from pathlib import Path

class DocxParser(DocumentParser):
    """Parser for Word documents."""
    
    def supported_extensions(self) -> List[str]:
        return ['.docx']
    
    def parse(self, file_path: Path) -> Document:
        # Implementation using python-docx
        import docx
        doc = docx.Document(file_path)
        # ... conversion logic
        return Document(...)

# Register the parser
ParserFactory.register('.docx', DocxParser)
```

### 13.2 Adding a New LLM Provider

```python
# my_providers/anthropic_provider.py
from Segmenta.llm.base import LLMProvider, LLMResponse
import anthropic

class AnthropicProvider(LLMProvider):
    """LLM provider using Anthropic Claude."""
    
    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
    
    def complete(self, prompt: str, system_prompt: str = None) -> LLMResponse:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1000,
            system=system_prompt or "",
            messages=[{"role": "user", "content": prompt}]
        )
        return LLMResponse(
            content=response.content[0].text,
            tokens_used=response.usage.input_tokens + response.usage.output_tokens,
            model=self._model
        )
    
    def complete_json(self, prompt: str, system_prompt: str = None) -> dict:
        # ... implementation
        pass

# Usage
Segmenta = (
    Segmenta.builder()
    .with_llm_provider(AnthropicProvider(api_key="..."))
    .build()
)
```

### 13.3 Adding a New Output Formatter

```python
# my_formatters/json_formatter.py
from Segmenta.output.base import OutputFormatter
from Segmenta.models import Chunk
from typing import List
import json

class JSONFormatter(OutputFormatter):
    """Output formatter producing JSON files."""
    
    def format(self, chunks: List[Chunk], output_path: str) -> str:
        output_data = {
            "chunks": [
                {
                    "metadata": chunk.metadata.to_yaml_dict() if chunk.metadata else {},
                    "content": chunk.content
                }
                for chunk in chunks
            ]
        }
        
        output_file = output_path.replace('.md', '.json')
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        return output_file

# Usage
Segmenta = (
    Segmenta.builder()
    .with_output_formatter(JSONFormatter())
    .build()
)
```

### 13.4 Custom Prompt Templates

```python
# my_prompts/custom_enrichment.py
from Segmenta.llm.prompts.base import PromptTemplate

class CustomEnrichmentPrompt(PromptTemplate):
    """Custom prompt for domain-specific enrichment."""
    
    SYSTEM = "You are a technical documentation expert..."
    
    TEMPLATE = """
    Analyze this technical documentation chunk:
    
    {chunk_content}
    
    Extract:
    - title
    - summary  
    - complexity_level: beginner | intermediate | advanced
    - prerequisites: list of required knowledge
    """

# Usage
enricher = ChunkEnricher(
    llm_provider=provider,
    prompt_template=CustomEnrichmentPrompt()
)
```

---

## 14. Performance Considerations

### 14.1 Optimization Strategies

| Strategy | Implementation | Impact |
|----------|----------------|--------|
| Lazy model loading | Load embedding model on first use | Faster startup |
| Batch embeddings | Process all paragraphs in single call | 10x faster embedding |
| Parallel LLM calls | Use asyncio for concurrent calls | 3-5x faster enrichment |
| Embedding cache | Cache embeddings by content hash | Skip redundant computation |
| Streaming output | Write chunks as they're processed | Lower memory usage |

### 14.2 Lazy Loading

```python
class SentenceTransformerProvider(EmbeddingProvider):
    def __init__(self, model_name: str):
        self._model_name = model_name
        self._model = None  # Not loaded yet
    
    @property
    def model(self):
        if self._model is None:
            # Load only when first needed
            self._model = SentenceTransformer(self._model_name)
        return self._model
```

### 14.3 Batch Processing

```python
class BoundaryDetectStage(PipelineStage):
    def process(self, context: PipelineContext) -> PipelineContext:
        texts = [p.text for p in context.paragraphs]
        
        # Single batch call instead of individual calls
        embeddings = self._embedding_provider.embed(texts)
        
        # Compute all similarities at once
        similarities = np.sum(
            embeddings[:-1] * embeddings[1:], axis=1
        ) / (
            np.linalg.norm(embeddings[:-1], axis=1) *
            np.linalg.norm(embeddings[1:], axis=1)
        )
        
        return context
```

### 14.4 Memory Profiling Targets

| Document Size | Expected Memory | Target Time |
|---------------|-----------------|-------------|
| 1,000 tokens | < 100 MB | < 5 seconds |
| 10,000 tokens | < 500 MB | < 30 seconds |
| 100,000 tokens | < 2 GB | < 5 minutes |

---

## 15. CLI Specification

### 15.1 Command Structure

```bash
# Basic usage
Segmenta <input_file> [OPTIONS]

# Examples
Segmenta document.pdf
Segmenta document.md -o ./output
Segmenta document.txt --model gpt-4o --verbose
Segmenta document.pdf --config Segmenta.yaml
```

### 15.2 CLI Options

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

### 15.3 Configuration File

```yaml
# Segmenta.yaml
llm:
  provider: openai
  model: gpt-4o
  api_key: ${OPENAI_API_KEY}  # Environment variable
  temperature: 0.1

embedding:
  model: all-MiniLM-L6-v2
  device: cuda  # or cpu

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

### 15.4 CLI Implementation

```python
# Segmenta/cli/main.py
import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from pathlib import Path
from .. import Segmenta, SegmentaConfig

console = Console()

@click.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('-o', '--output-dir', default='./output', help='Output directory')
@click.option('-m', '--model', default='gpt-4o', help='OpenAI model name')
@click.option('-k', '--api-key', envvar='OPENAI_API_KEY', help='OpenAI API key')
@click.option('-c', '--config', type=click.Path(exists=True), help='Config file path')
@click.option('-t', '--threshold', default=0.5, help='Similarity threshold')
@click.option('--min-tokens', default=50, help='Min tokens per chunk')
@click.option('--max-tokens', default=500, help='Max tokens per chunk')
@click.option('-v', '--verbose', is_flag=True, help='Verbose output')
@click.option('--dry-run', is_flag=True, help='Parse without LLM calls')
@click.version_option()
def cli(input_file, output_dir, model, api_key, config, threshold, 
        min_tokens, max_tokens, verbose, dry_run):
    """Segmenta: Semantic document chunking library.
    
    Transform documents into semantically coherent, metadata-enriched chunks.
    """
    
    # Load config from file if provided
    if config:
        Segmenta_config = SegmentaConfig.from_yaml(config)
    else:
        Segmenta_config = SegmentaConfig(
            similarity_threshold=threshold,
            min_chunk_tokens=min_tokens,
            max_chunk_tokens=max_tokens,
            verbose=verbose
        )
    
    # Initialize Segmenta
    Segmenta = Segmenta(
        openai_api_key=api_key,
        model=model,
        config=Segmenta_config
    )
    
    # Process with progress indicator
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Processing...", total=None)
        
        def on_progress(stage: str, pct: float):
            progress.update(task, description=f"[cyan]{stage}[/cyan]")
        
        result = Segmenta.chunk(
            input_file=input_file,
            output_dir=output_dir,
            progress_callback=on_progress,
            dry_run=dry_run
        )
    
    # Display results
    if result.success:
        console.print(f"\n[green]✓[/green] Generated {result.chunk_count} chunks")
        console.print(f"[green]✓[/green] Output: {result.output_path}")
    else:
        console.print(f"\n[red]✗[/red] Processing failed")
        for error in result.errors:
            console.print(f"  [red]•[/red] {error}")

if __name__ == '__main__':
    cli()
```

---

## Appendix A: Sequence Diagrams

### A.1 Main Processing Flow

```
User                Segmenta              Parser           Embedder            LLM
 │                    │                   │                 │                 │
 │  chunk(file)       │                   │                 │                 │
 │───────────────────▶│                   │                 │                 │
 │                    │  parse(file)      │                 │                 │
 │                    │──────────────────▶│                 │                 │
 │                    │     Document      │                 │                 │
 │                    │◀──────────────────│                 │                 │
 │                    │                   │                 │                 │
 │                    │  embed(paragraphs)│                 │                 │
 │                    │──────────────────────────────────▶│                 │
 │                    │       embeddings  │                 │                 │
 │                    │◀──────────────────────────────────│                 │
 │                    │                   │                 │                 │
 │                    │  validate_boundary(context)        │                 │
 │                    │───────────────────────────────────────────────────▶│
 │                    │       KEEP/MERGE/ADJUST           │                 │
 │                    │◀───────────────────────────────────────────────────│
 │                    │                   │                 │                 │
 │                    │  enrich_chunk(content)             │                 │
 │                    │───────────────────────────────────────────────────▶│
 │                    │       metadata    │                 │                 │
 │                    │◀───────────────────────────────────────────────────│
 │                    │                   │                 │                 │
 │    SegmentaResult    │                   │                 │                 │
 │◀───────────────────│                   │                 │                 │
 │                    │                   │                 │                 │
```

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **Chunk** | A semantically coherent unit of text with metadata |
| **Boundary** | A proposed split point between paragraphs |
| **Atomic Element** | Content that should never be split (code blocks, tables) |
| **Stage 1** | Embedding-based boundary detection |
| **Stage 2** | LLM-based boundary validation |
| **Enrichment** | Process of extracting metadata from chunk content |
| **Intent** | The purpose of a chunk (explains, defines, warns, etc.) |
| **Similarity Threshold** | Score below which a boundary is proposed |
| **Fallback** | Default behavior when LLM fails |

---

## Appendix C: Sample Test Documents

### C.1 Simple Test (simple.md)

```markdown
# Introduction

This is a test document for Segmenta. It contains multiple sections.

# Section One

First paragraph of section one. It discusses topic A.

Second paragraph continues topic A with more details.

# Section Two

This section covers a completely different topic B.
```

### C.2 With Code Blocks (with_code.md)

```markdown
# Code Example

Here is some Python code:

\`\`\`python
def hello():
    print("Hello, World!")
\`\`\`

The function above prints a greeting.
```

---

*Document Version: 1.0*
*Last Updated: February 2026*
