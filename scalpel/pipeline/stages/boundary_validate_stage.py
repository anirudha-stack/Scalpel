"""Boundary validation stage - uses LLM to validate proposed boundaries."""

import time
import logging
from typing import Optional

from scalpel.pipeline.base import PipelineStage
from scalpel.pipeline.context import PipelineContext
from scalpel.llm.base import LLMProvider
from scalpel.llm.prompts.validation import (
    BOUNDARY_VALIDATION_SYSTEM,
    format_validation_prompt,
)
from scalpel.models import BoundaryDecision, BoundaryVerdict
from scalpel.utils.retry import RetryHandler
from scalpel.exceptions import ScalpelLLMError

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
        llm_tokens_used = 0

        for proposal in context.boundary_proposals:
            try:
                decision = self._validate_boundary(proposal)
                context.boundary_decisions.append(decision)

            except ScalpelLLMError as e:
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

    def _validate_boundary(self, proposal) -> BoundaryDecision:
        """Validate a single boundary proposal.

        Args:
            proposal: Boundary proposal to validate

        Returns:
            BoundaryDecision with the LLM verdict
        """
        prompt = format_validation_prompt(
            text_before=proposal.get_text_before(max_chars=500),
            text_after=proposal.get_text_after(max_chars=500),
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

        return BoundaryDecision(
            proposal=proposal,
            verdict=verdict,
            reason=reason,
            confidence=confidence,
        )
