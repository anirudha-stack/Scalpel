# Microtome: Semantic Document Chunking Library

## Business Requirements Document

---

## 1. What is Microtome?

A Python library that transforms unstructured documents (PDF, Markdown, Text) into semantically coherent, metadata-enriched chunks.

**Core Value Proposition**: Documents are split not by arbitrary character/token limits, but by logical and semantic boundaries — producing chunks that represent complete, self-contained ideas.

---

## 2. User Interaction

User provides:
- Input file (PDF / MD / TXT)
- Output directory
- LLM provider configuration (OpenAI API key, model name)

User receives:
- A single `.md` file: `Microtome_output_<original_filename>.md`

---

## 3. What Microtome Produces

### 3.1 Output File Structure

Each chunk in the output file contains:

1. **Metadata Block** (YAML frontmatter)
   - `chunk_id`: Unique identifier
   - `title`: Concise title describing chunk content
   - `summary`: 1-2 sentence summary
   - `intent`: What this chunk does (explains / lists / warns / defines / instructs / compares)
   - `keywords`: 3-7 relevant terms
   - `parent_section`: Original document section this belonged to
   - `token_count`: Size of chunk in tokens

2. **Content Block**: The actual text of the chunk

---

### 3.2 Example Output

**Input Document** (`motor_control.md`):
```
# Motor Control Basics

## PWM Fundamentals

PWM stands for Pulse Width Modulation. It controls power delivery by 
switching voltage on and off rapidly. The duty cycle determines average 
power output.

A 50% duty cycle means power is on half the time. Higher duty cycles 
deliver more power to the motor.

## Current Sensing

Measuring motor current is essential for torque control. Shunt resistors 
are commonly used for this purpose. The voltage across the shunt is 
proportional to current.
```

**Output** (`Microtome_output_motor_control.md`):
```
---
chunk_id: chunk_001
title: PWM Fundamentals and Duty Cycle
summary: Explains PWM as a power control method using rapid on-off switching, where duty cycle determines average power output.
intent: explains
keywords: [PWM, pulse width modulation, duty cycle, power control]
parent_section: PWM Fundamentals
token_count: 58
---

PWM stands for Pulse Width Modulation. It controls power delivery by switching voltage on and off rapidly. The duty cycle determines average power output.

A 50% duty cycle means power is on half the time. Higher duty cycles deliver more power to the motor.

---
chunk_id: chunk_002
title: Current Sensing for Torque Control
summary: Describes current measurement in motors using shunt resistors for torque control applications.
intent: explains
keywords: [current sensing, torque control, shunt resistor, motor current]
parent_section: Current Sensing
token_count: 41
---

Measuring motor current is essential for torque control. Shunt resistors are commonly used for this purpose. The voltage across the shunt is proportional to current.
```

---

## 4. Chunking Logic

### 4.1 Two-Stage Approach

**Stage 1 — Fast Approximation (Sentence Transformer)**
- Computes semantic similarity between adjacent paragraphs
- Proposes chunk boundaries where similarity drops significantly
- Fast, cheap, but imperfect

**Stage 2 — Intelligent Validation (LLM)**
- Reviews each proposed boundary
- Decides: Keep split / Merge / Adjust split point
- Ensures logical coherence, not just semantic similarity

---

### 4.2 Why Two Stages?

Embeddings fail on logical coherence:

| Text | Embedding Verdict | Correct Verdict |
|------|-------------------|-----------------|
| "The dog runs 10m/s. Running fast is difficult." | Same chunk (high similarity) | **Split** — different topics |
| "Python handles data well. Python is a snake in Asia." | Same chunk (high similarity) | **Split** — unrelated meanings |
| "This concludes PWM. However, one exception exists..." | Split (low similarity) | **Merge** — continuation |

LLM catches what embeddings miss.

---

### 4.3 Sentence Transformer Role

**Input**: Sequence of paragraphs from a document section

**Task**: Compute pairwise similarity between adjacent paragraphs

**Output**: Similarity scores

**Example**:

```
Paragraph 1: "PWM controls power by switching rapidly."
Paragraph 2: "Duty cycle is the ratio of on-time to total period."
Paragraph 3: "Shunt resistors measure current flow."

Similarities:
  P1 ↔ P2: 0.82 (high — same topic)
  P2 ↔ P3: 0.34 (low — topic shift)

Proposed boundary: Between P2 and P3
```

---

### 4.4 LLM Validation Role

**Input**: Text around each proposed boundary

**Task**: Validate if boundary is correct

**Output**: KEEP / MERGE / ADJUST

**Example Prompt**:
```
Below is a proposed chunk boundary. Should these be separate chunks?

END OF CHUNK 1:
"...duty cycle is the ratio of on-time to total period."

START OF CHUNK 2:
"Shunt resistors measure current flow..."

Options:
A) KEEP — These are different logical units
B) MERGE — These belong together
C) ADJUST — Split point should be elsewhere (specify)
```

**Expected Response**:
```
Verdict: KEEP
Reason: Chunk 1 discusses PWM duty cycle. Chunk 2 introduces current sensing, a distinct topic.
```

---

### 4.5 LLM Enrichment Role

**Input**: Complete chunk text

**Task**: Extract metadata

**Output**: title, summary, intent, keywords

**Example Prompt**:
```
Extract metadata for this text chunk:

"PWM stands for Pulse Width Modulation. It controls power delivery by 
switching voltage on and off rapidly. The duty cycle determines average 
power output. A 50% duty cycle means power is on half the time."

Provide:
- title: (concise, descriptive)
- summary: (1-2 sentences)
- intent: (one of: explains, lists, warns, defines, instructs, compares)
- keywords: (3-7 relevant terms)
```

**Expected Response**:
```
title: PWM Fundamentals and Duty Cycle
summary: Explains PWM as a power control method using rapid on-off switching, where duty cycle determines average power output.
intent: explains
keywords: [PWM, pulse width modulation, duty cycle, power control, switching]
```

---

## 5. Structural Rules

These elements are never split mid-way:

| Element | Rule |
|---------|------|
| Code blocks | Atomic — always kept whole |
| Tables | Atomic — always kept whole |
| Lists | Kept with their parent context |
| Headers with no body | Attached to following content |

---

## 6. Edge Cases

| Scenario | Behavior |
|----------|----------|
| Very short document (<5 paragraphs) | Single chunk, no splitting |
| LLM returns invalid response | Retry once, then use fallback (first sentence as summary, "unknown" intent) |
| No clear semantic boundaries | Respect structural boundaries only (headers) |
| Single long paragraph | Sentence-level analysis as fallback |

---

## 7. Success Criteria

A chunk is well-formed if:

1. **Self-contained**: Can be understood without reading other chunks
2. **Coherent**: All sentences relate to a single topic/idea
3. **Complete**: Doesn't cut off mid-explanation
4. **Metadata-accurate**: Title and summary genuinely reflect content

---

## 8. Out of Scope

- Chunk overlap configuration (consumer's responsibility)
- Embedding generation for RAG (consumer's responsibility)
- Vector database integration
- Multi-file batch processing (v1 handles single file)
- Non-English language optimization (v1 targets English)

---

## 9. Dependencies

| Component | Purpose |
|-----------|---------|
| Sentence Transformers | Local embedding for similarity computation |
| OpenAI API | LLM for validation and enrichment |
| PDF Parser | Extract text from PDFs |
| Markdown Parser | Parse MD structure |
| Token Counter | Accurate token counts in metadata |