"""Boundary validation stage - uses LLM to validate proposed boundaries."""

import time
import logging
import re
from typing import Optional

from segmenta.pipeline.base import PipelineStage
from segmenta.pipeline.context import PipelineContext
from segmenta.llm.base import LLMProvider
from segmenta.llm.prompts.validation import (
    BOUNDARY_CRITIQUE_SYSTEM,
    BOUNDARY_VALIDATION_SYSTEM,
    format_boundary_critique_prompt,
    format_validation_prompt,
)
from segmenta.models import BoundaryDecision, BoundaryVerdict
from segmenta.utils.retry import RetryHandler
from segmenta.exceptions import SegmentaLLMError

logger = logging.getLogger(__name__)


class BoundaryValidateStage(PipelineStage):
    """Stage that validates boundaries using LLM."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        retry_handler: Optional[RetryHandler] = None,
    ) -> None:
        """Initialize the boundary validation stage.

        Args:
            llm_provider: LLM provider for validation
            retry_handler: Optional retry handler for LLM calls
        """
        self._llm = llm_provider
        self._retry = retry_handler

    @property
    def name(self) -> str:
        return "boundary_validate"

    def should_skip(self, context: PipelineContext) -> bool:
        """Skip if no boundary proposals or in dry run mode."""
        if context.dry_run:
            return True
        return len(context.boundary_proposals) == 0

    def process(self, context: PipelineContext) -> PipelineContext:
        """Validate boundaries using LLM.

        Args:
            context: Pipeline context

        Returns:
            Updated context with boundary decisions
        """
        start_time = time.time()
        critique_enabled = bool(
            getattr(context.config, "boundary_validation_critique_enabled", True)
        )
        context.add_metric("boundary_critique_enabled", critique_enabled)
        context.add_metric("boundary_critique_calls", 0)
        context.add_metric("boundary_critique_vetoes", 0)
        context.add_metric("boundary_critique_failed", 0)

        for proposal in context.boundary_proposals:
            try:
                decision = self._validate_boundary(proposal, context, critique_enabled)
                context.boundary_decisions.append(decision)

            except SegmentaLLMError as e:
                logger.warning(f"LLM validation failed: {e}")
                # Default to KEEP on failure
                context.boundary_decisions.append(
                    BoundaryDecision(
                        proposal=proposal,
                        verdict=BoundaryVerdict.KEEP,
                        reason="LLM validation failed, defaulting to KEEP",
                        confidence=0.5,
                    )
                )
                context.add_warning(f"Boundary validation failed: {e}")

        # Record metrics
        context.add_metric("boundary_validate_time", time.time() - start_time)
        context.add_metric("boundaries_validated", len(context.boundary_decisions))
        context.add_metric(
            "boundaries_kept",
            sum(1 for d in context.boundary_decisions if d.verdict == BoundaryVerdict.KEEP),
        )
        context.add_metric(
            "boundaries_merged",
            sum(1 for d in context.boundary_decisions if d.verdict == BoundaryVerdict.MERGE),
        )

        return context

    def _validate_boundary(
        self,
        proposal,
        context: PipelineContext,
        critique_enabled: bool,
    ) -> BoundaryDecision:
        """Validate a single boundary proposal.

        Args:
            proposal: Boundary proposal to validate
            context: Pipeline context (for configuration and metrics)
            critique_enabled: Whether the binary critique gate is enabled

        Returns:
            BoundaryDecision with the LLM verdict
        """
        ctx_block = self._build_context_block(proposal, context)
        prompt = format_validation_prompt(
            text_before=proposal.get_text_before(max_chars=500),
            text_after=proposal.get_text_after(max_chars=500),
            context_block=ctx_block,
        )

        if self._retry:
            result = self._retry.execute(
                self._llm.complete_json,
                prompt,
                BOUNDARY_VALIDATION_SYSTEM,
            )
        else:
            result = self._llm.complete_json(prompt, BOUNDARY_VALIDATION_SYSTEM)

        # Parse the result
        verdict_str = result.get("verdict", "KEEP")
        verdict = BoundaryVerdict.from_string(verdict_str)
        reason = result.get("reason", "")
        confidence = float(result.get("confidence", 1.0))

        # Binary critique veto: only runs when the worker would keep/split.
        if critique_enabled and verdict != BoundaryVerdict.MERGE:
            context.increment_metric("boundary_critique_calls", 1)
            critique_prompt = format_boundary_critique_prompt(
                text_before=proposal.get_text_before(max_chars=500),
                text_after=proposal.get_text_after(max_chars=500),
                context_block=ctx_block,
            )
            try:
                critique_resp = self._llm.complete(
                    critique_prompt, BOUNDARY_CRITIQUE_SYSTEM
                )
                if critique_resp.failed:
                    context.increment_metric("boundary_critique_failed", 1)
                else:
                    vote = self._parse_yes_no(critique_resp.content)
                    if vote == "NO":
                        context.increment_metric("boundary_critique_vetoes", 1)
                        verdict = BoundaryVerdict.MERGE
                        # Preserve worker output for auditing.
                        reason = (
                            "Critique vetoed boundary (NO). "
                            f"Worker verdict={verdict_str!r}. Worker reason={reason!r}"
                        )
                        confidence = min(confidence, 0.5)
            except Exception as e:
                # Never fail the stage due to critique issues; fall back to worker verdict.
                context.increment_metric("boundary_critique_failed", 1)
                context.add_warning(f"Boundary critique error: {e}. Using worker verdict.")

        return BoundaryDecision(
            proposal=proposal,
            verdict=verdict,
            reason=reason,
            confidence=confidence,
        )

    @staticmethod
    def _build_context_block(proposal, context: PipelineContext) -> str:
        """Build optional global context for both worker and critique prompts."""
        lines = [
            "Context:",
            f"- similarity_score: {getattr(proposal, 'similarity_score', 0.0):.3f}",
            f"- min_chunk_tokens: {context.config.min_chunk_tokens}",
            f"- max_chunk_tokens: {context.config.max_chunk_tokens}",
        ]

        plan = context.metrics.get("granularity_plan")
        if isinstance(plan, dict):
            topics = plan.get("topics")
            if isinstance(topics, list):
                topics_clean = [
                    t.strip()
                    for t in topics
                    if isinstance(t, str) and t.strip()
                ][:12]
                if topics_clean:
                    joined = ", ".join(topics_clean)
                    lines.append(f"- document_topics (approx): {joined}")

        return "\n".join(lines) + "\n\n"

    _YES_NO_RE = re.compile(r"\b(YES|NO)\b", re.IGNORECASE)

    @classmethod
    def _parse_yes_no(cls, text: str) -> str:
        """Parse a strict YES/NO vote from an LLM response."""
        if not isinstance(text, str):
            return ""
        stripped = text.strip().upper()
        if stripped in {"YES", "NO"}:
            return stripped
        match = cls._YES_NO_RE.search(text)
        if match:
            return match.group(1).upper()
        return ""
