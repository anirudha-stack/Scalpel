from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from segmenta.config import SegmentaConfig
from segmenta.llm.base import LLMProvider, LLMResponse
from segmenta.llm.prompts.granularity import (
    GRANULARITY_CRITIQUE_SYSTEM,
    GRANULARITY_SYSTEM,
)
from segmenta.llm.prompts.validation import (
    BOUNDARY_CRITIQUE_SYSTEM,
    BOUNDARY_VALIDATION_SYSTEM,
)
from segmenta.models import BoundaryProposal, BoundaryVerdict, Document, Paragraph, Section
from segmenta.pipeline.context import PipelineContext
from segmenta.pipeline.stages.boundary_validate_stage import BoundaryValidateStage
from segmenta.pipeline.stages.granularity_stage import GranularityStage


@dataclass
class FakeLLM(LLMProvider):
    json_by_system: Dict[str, Dict[str, Any]]
    text_by_system: Dict[str, str]
    model: str = "fake"

    def complete(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        content = self.text_by_system.get(system_prompt or "", "")
        return LLMResponse(
            content=content,
            tokens_used=0,
            model=self.model,
            success=True,
        )

    def complete_json(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        return self.json_by_system.get(system_prompt or "", {})

    @property
    def model_name(self) -> str:
        return self.model


def _ctx_with_doc(*, config: SegmentaConfig) -> PipelineContext:
    paras = [
        Paragraph(text="Alpha. Beta. Gamma. Delta.", index=0),
        Paragraph(text="Epsilon. Zeta. Eta. Theta.", index=1),
        Paragraph(text="Iota. Kappa. Lambda. Mu.", index=2),
    ]
    doc = Document(
        filename="x.txt",
        file_type="txt",
        sections=[Section(title="Root", level=0, paragraphs=paras)],
        raw_text="",
    )
    return PipelineContext(
        input_path="x.txt",
        output_dir="out",
        config=config,
        document=doc,
        paragraphs=list(paras),
    )


def test_granularity_plan_rejected_by_critique() -> None:
    llm = FakeLLM(
        json_by_system={
            GRANULARITY_SYSTEM: {
                "topics": ["t1", "t2"],
                "expected_chunk_count": 10,
                "expected_chunk_count_range": [8, 12],
                "atomize_sentences_per_paragraph": 2,
                "atomize_min_sentences": 6,
                "rationale": "x",
                "confidence": 0.9,
            },
            GRANULARITY_CRITIQUE_SYSTEM: {
                "verdict": "REJECT",
                "issues": ["looks inconsistent"],
                "confidence": 0.8,
            },
        },
        text_by_system={},
    )
    cfg = SegmentaConfig(
        granularity_planning_enabled=True,
        granularity_critique_enabled=True,
    )
    ctx = _ctx_with_doc(config=cfg)

    out = GranularityStage(llm).process(ctx)
    assert out.metrics["granularity_plan_applied"] is False
    assert out.atomize_sentences_per_paragraph_override is None
    assert out.atomize_min_sentences_override is None


def test_granularity_plan_accepted_by_critique() -> None:
    llm = FakeLLM(
        json_by_system={
            GRANULARITY_SYSTEM: {
                "topics": ["t1"],
                "expected_chunk_count": 5,
                "expected_chunk_count_range": [4, 6],
                "atomize_sentences_per_paragraph": 2,
                "atomize_min_sentences": 6,
                "rationale": "x",
                "confidence": 0.9,
            },
            GRANULARITY_CRITIQUE_SYSTEM: {
                "verdict": "ACCEPT",
                "issues": [],
                "confidence": 0.9,
            },
        },
        text_by_system={},
    )
    cfg = SegmentaConfig(
        granularity_planning_enabled=True,
        granularity_critique_enabled=True,
    )
    ctx = _ctx_with_doc(config=cfg)

    out = GranularityStage(llm).process(ctx)
    assert out.metrics["granularity_plan_applied"] is True
    plan = out.metrics["granularity_plan"]
    assert out.atomize_sentences_per_paragraph_override == plan["atomize_sentences_per_paragraph"]
    assert out.atomize_min_sentences_override == plan["atomize_min_sentences"]


def test_boundary_critique_vetoes_keep() -> None:
    llm = FakeLLM(
        json_by_system={
            BOUNDARY_VALIDATION_SYSTEM: {
                "verdict": "KEEP",
                "reason": "x",
                "confidence": 0.9,
            }
        },
        text_by_system={BOUNDARY_CRITIQUE_SYSTEM: "NO"},
    )
    cfg = SegmentaConfig(boundary_validation_critique_enabled=True)
    ctx = _ctx_with_doc(config=cfg)

    before = ctx.paragraphs[0]
    after = ctx.paragraphs[1]
    ctx.boundary_proposals = [
        BoundaryProposal(
            position=1,
            similarity_score=0.2,
            paragraph_before=before,
            paragraph_after=after,
        )
    ]

    out = BoundaryValidateStage(llm_provider=llm).process(ctx)
    assert out.metrics["boundary_critique_calls"] == 1
    assert out.metrics["boundary_critique_vetoes"] == 1
    assert out.boundary_decisions[0].verdict == BoundaryVerdict.MERGE


def test_boundary_critique_not_called_on_merge() -> None:
    llm = FakeLLM(
        json_by_system={
            BOUNDARY_VALIDATION_SYSTEM: {
                "verdict": "MERGE",
                "reason": "x",
                "confidence": 0.9,
            }
        },
        text_by_system={BOUNDARY_CRITIQUE_SYSTEM: "NO"},
    )
    cfg = SegmentaConfig(boundary_validation_critique_enabled=True)
    ctx = _ctx_with_doc(config=cfg)

    before = ctx.paragraphs[0]
    after = ctx.paragraphs[1]
    ctx.boundary_proposals = [
        BoundaryProposal(
            position=1,
            similarity_score=0.2,
            paragraph_before=before,
            paragraph_after=after,
        )
    ]

    out = BoundaryValidateStage(llm_provider=llm).process(ctx)
    assert out.metrics["boundary_critique_calls"] == 0
    assert out.boundary_decisions[0].verdict == BoundaryVerdict.MERGE


def test_boundary_critique_disabled() -> None:
    llm = FakeLLM(
        json_by_system={
            BOUNDARY_VALIDATION_SYSTEM: {
                "verdict": "KEEP",
                "reason": "x",
                "confidence": 0.9,
            }
        },
        text_by_system={BOUNDARY_CRITIQUE_SYSTEM: "NO"},
    )
    cfg = SegmentaConfig(boundary_validation_critique_enabled=False)
    ctx = _ctx_with_doc(config=cfg)

    before = ctx.paragraphs[0]
    after = ctx.paragraphs[1]
    ctx.boundary_proposals = [
        BoundaryProposal(
            position=1,
            similarity_score=0.2,
            paragraph_before=before,
            paragraph_after=after,
        )
    ]

    out = BoundaryValidateStage(llm_provider=llm).process(ctx)
    assert out.metrics["boundary_critique_calls"] == 0
    assert out.boundary_decisions[0].verdict == BoundaryVerdict.KEEP
