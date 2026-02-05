"""Granularity planning stage - uses an LLM to adapt atomization settings.

PDFs frequently collapse multiple semantic ideas into a single extracted paragraph.
Atomization (sentence-group splitting) helps expose boundaries, but a fixed setting
can over-split or under-split depending on the document.

This stage sends a compact document sample to the LLM to estimate:
- topics covered
- expected chunk count
- recommended atomization settings
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional

from scalpel.pipeline.base import PipelineStage
from scalpel.pipeline.context import PipelineContext
from scalpel.llm.base import LLMProvider
from scalpel.llm.prompts.granularity import (
    GRANULARITY_SYSTEM,
    format_granularity_prompt,
)
from scalpel.exceptions import ScalpelLLMError

logger = logging.getLogger(__name__)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


class GranularityStage(PipelineStage):
    """Stage that plans chunking granularity and atomization settings."""

    def __init__(self, llm_provider: LLMProvider) -> None:
        self._llm = llm_provider

    @property
    def name(self) -> str:
        return "granularity"

    def should_skip(self, context: PipelineContext) -> bool:
        if context.dry_run:
            return True
        return not context.config.granularity_planning_enabled

    def process(self, context: PipelineContext) -> PipelineContext:
        start_time = time.time()

        if context.document is None or not context.paragraphs:
            context.add_metric("granularity_time", time.time() - start_time)
            context.add_metric("granularity_planning_skipped", True)
            return context

        try:
            paragraph_lines = self._build_paragraph_lines(context)
            prompt = format_granularity_prompt(
                file_type=context.document.file_type,
                paragraph_count=context.paragraph_count,
                section_count=len(context.document.sections),
                section_titles=self._format_section_titles(context),
                min_chunk_tokens=context.config.min_chunk_tokens,
                max_chunk_tokens=context.config.max_chunk_tokens,
                similarity_threshold=context.config.similarity_threshold,
                paragraph_lines=paragraph_lines,
            )

            raw = self._llm.complete_json(prompt, GRANULARITY_SYSTEM)
            plan = self._normalize_plan(raw, context)
            plan = self._apply_safety_adjustments(plan, context)

            # Persist in metrics for inspection / downstream use.
            context.add_metric("granularity_plan", plan)
            context.add_metric("granularity_planning_skipped", False)

            # Apply overrides for the atomizer.
            context.atomize_sentences_per_paragraph_override = plan.get(
                "atomize_sentences_per_paragraph"
            )
            context.atomize_min_sentences_override = plan.get(
                "atomize_min_sentences"
            )

        except ScalpelLLMError as e:
            context.add_warning(
                f"Granularity planning failed: {e}. Using configured atomization defaults."
            )
            context.add_metric("granularity_planning_failed", True)
        except Exception as e:
            # Never fail the pipeline due to planning issues.
            context.add_warning(
                f"Granularity planning error: {e}. Using configured atomization defaults."
            )
            context.add_metric("granularity_planning_failed", True)

        context.add_metric("granularity_time", time.time() - start_time)
        return context

    def _apply_safety_adjustments(
        self, plan: Dict[str, Any], context: PipelineContext
    ) -> Dict[str, Any]:
        """Apply light heuristics so the plan is likely to be effective.

        The LLM can recommend settings that look reasonable in isolation, but end up
        being no-ops on the actual extracted paragraph shapes (e.g., min_sentences
        higher than any paragraph's sentence count). Here we:
        - avoid extreme over-atomization (too many atoms)
        - avoid under-atomization (too few atoms to reach the expected chunk range)
        """
        group_size = plan.get("atomize_sentences_per_paragraph")
        min_sentences = plan.get("atomize_min_sentences")
        expected_range = plan.get("expected_chunk_count_range", [None, None])
        if not isinstance(group_size, int) or group_size <= 0:
            return plan
        if not isinstance(min_sentences, int):
            min_sentences = context.config.atomize_min_sentences

        try:
            expected_lo = (
                int(expected_range[0])
                if expected_range and len(expected_range) >= 2
                else None
            )
            expected_hi = (
                int(expected_range[1])
                if expected_range and len(expected_range) >= 2
                else None
            )
        except Exception:
            expected_lo = None
            expected_hi = None

        if (
            expected_lo is None
            or expected_hi is None
            or expected_lo <= 0
            or expected_hi <= 0
        ):
            return plan

        def _estimate_atoms(gs: int, ms: int) -> int:
            # Mirrors AtomizeStage behavior (including tail-merge).
            atoms = 0
            for para in context.paragraphs:
                if not getattr(para, "is_splittable", False):
                    atoms += 1
                    continue
                text = (getattr(para, "text", "") or "").strip()
                sent_count = self._count_sentences(text)
                if sent_count < max(ms, gs + 1):
                    atoms += 1
                else:
                    # Tail sentences are merged into the last group, so the
                    # number of produced atoms is floor(n/gs), not ceil.
                    atoms += max(1, sent_count // gs)
            return atoms

        # Atomization must be able to produce at least the low end of the expected range.
        atoms_estimate = _estimate_atoms(group_size, min_sentences)

        if atoms_estimate < expected_lo:
            # Under-atomization: coarsest parameters that reach expected_lo.
            # First, clamp min_sentences to the observed max sentence count so the
            # setting doesn't accidentally disable atomization for the whole doc.
            sent_counts = []
            for para in context.paragraphs:
                if getattr(para, "is_splittable", False):
                    text = (getattr(para, "text", "") or "").strip()
                    sent_counts.append(self._count_sentences(text))
            max_sent = max(sent_counts) if sent_counts else 0

            start_min = min_sentences
            if max_sent > 0:
                start_min = min(start_min, max_sent)

            chosen_gs = group_size
            chosen_ms = start_min
            found = False

            # Prefer keeping group_size (cost) and only lowering min_sentences.
            # If that isn't enough, lower group_size (finer atoms) as a last resort.
            for gs in range(max(1, group_size), 1, -1):
                for ms in range(start_min, -1, -1):
                    if _estimate_atoms(gs, ms) >= expected_lo:
                        chosen_gs = gs
                        chosen_ms = ms
                        found = True
                        break
                if found:
                    break

            if not found:
                # Last resort: allow sentence-level groups to hit the target.
                for ms in range(start_min, -1, -1):
                    if _estimate_atoms(1, ms) >= expected_lo:
                        chosen_gs = 1
                        chosen_ms = ms
                        found = True
                        break

            if found and (chosen_gs != group_size or chosen_ms != min_sentences):
                plan = dict(plan)
                plan["atomize_sentences_per_paragraph"] = chosen_gs
                plan["atomize_min_sentences"] = chosen_ms
                plan["rationale"] = (
                    (plan.get("rationale", "") + " ").strip()
                    + (
                        " (Adjusted atomization to avoid under-splitting: "
                        f"{chosen_gs} sentences/group, min_sentences={chosen_ms}.)"
                    )
                ).strip()
                group_size = chosen_gs
                min_sentences = chosen_ms
                atoms_estimate = _estimate_atoms(group_size, min_sentences)

        # If the plan expects substantially more chunks than the raw paragraph count,
        # prefer finer groups (when safe) to better expose intra-paragraph topic shifts.
        expected = plan.get("expected_chunk_count")
        if (
            isinstance(expected, int)
            and context.paragraph_count > 0
            and expected > context.paragraph_count
        ):
            ratio = expected / max(1, context.paragraph_count)
            if ratio >= 1.5 and group_size > 2:
                candidate_atoms = _estimate_atoms(2, min_sentences)
                if candidate_atoms <= (expected_hi * 3):
                    plan = dict(plan)
                    plan["atomize_sentences_per_paragraph"] = 2
                    plan["rationale"] = (
                        (plan.get("rationale", "") + " ").strip()
                        + (
                            " (Adjusted atomization to 2 sentences/group to better match the "
                            "expected chunk granularity.)"
                        )
                    ).strip()
                    group_size = 2
                    atoms_estimate = candidate_atoms

        # If we're way above the expected chunk range, coarsen atomization.
        # This reduces cost in later boundary validation without disabling atomization entirely.
        cap = expected_hi * 3
        adjusted = group_size
        while atoms_estimate > cap and adjusted < 6:
            adjusted += 1
            atoms_estimate = _estimate_atoms(adjusted, min_sentences)

        if adjusted != group_size:
            plan = dict(plan)
            plan["atomize_sentences_per_paragraph"] = adjusted
            plan["rationale"] = (
                (plan.get("rationale", "") + " ").strip()
                + f" (Adjusted atomization to {adjusted} sentences/group to avoid over-splitting.)"
            ).strip()

        return plan

    @staticmethod
    def _format_section_titles(context: PipelineContext, max_titles: int = 20) -> str:
        if context.document is None:
            return "[]"

        titles = []
        for section in context.document.all_sections_flat:
            title = (section.title or "").strip()
            if title:
                titles.append(title)

        if not titles:
            return "[]"

        truncated = len(titles) > max_titles
        titles = titles[:max_titles]
        # Render as a compact JSON-ish list for readability in the prompt.
        joined = ", ".join([f"\"{t}\"" for t in titles])
        suffix = ", ..." if truncated else ""
        return f"[{joined}{suffix}]"

    def _build_paragraph_lines(self, context: PipelineContext) -> str:
        paragraphs = context.paragraphs
        max_paragraphs = context.config.granularity_max_paragraphs
        max_chars = context.config.granularity_max_chars_per_paragraph

        sampled = self._sample_paragraphs(paragraphs, max_paragraphs)

        lines: List[str] = []
        for para in sampled:
            text = (para.text or "").strip().replace("\n", " ")
            sent_count = self._count_sentences(text)
            snippet = text[:max_chars]
            if len(text) > max_chars:
                snippet = snippet.rstrip() + "..."
            lines.append(
                f"- P{para.index} sentences={sent_count} chars={len(text)}: {snippet}"
            )

        if len(sampled) < len(paragraphs):
            lines.append(
                f"- ... (sampled {len(sampled)}/{len(paragraphs)} paragraphs)"
            )

        return "\n".join(lines)

    @staticmethod
    def _count_sentences(text: str) -> int:
        if not text:
            return 0
        return len([s for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()])

    @staticmethod
    def _sample_paragraphs(paragraphs: List[Any], max_paragraphs: int) -> List[Any]:
        if max_paragraphs <= 0 or len(paragraphs) <= max_paragraphs:
            return list(paragraphs)

        # Take a stable sample: start, middle, end.
        third = max_paragraphs // 3
        head = paragraphs[:third]
        tail = paragraphs[-third:]
        remaining = max_paragraphs - (len(head) + len(tail))
        if remaining <= 0:
            return list(head) + list(tail)

        mid_start = max((len(paragraphs) // 2) - (remaining // 2), len(head))
        mid_end = min(mid_start + remaining, len(paragraphs) - len(tail))
        middle = paragraphs[mid_start:mid_end]
        return list(head) + list(middle) + list(tail)

    @staticmethod
    def _normalize_plan(raw: Dict[str, Any], context: PipelineContext) -> Dict[str, Any]:
        def _to_int(value: Any) -> Optional[int]:
            if value is None:
                return None
            if isinstance(value, bool):
                return int(value)
            if isinstance(value, (int, float)):
                return int(round(value))
            if isinstance(value, str):
                try:
                    return int(round(float(value.strip())))
                except Exception:
                    return None
            return None

        def _clamp_int(value: Optional[int], lo: int, hi: int) -> Optional[int]:
            if value is None:
                return None
            return max(lo, min(hi, value))

        topics = raw.get("topics", [])
        if isinstance(topics, str):
            topics = [t.strip() for t in topics.split(",") if t.strip()]
        if not isinstance(topics, list):
            topics = []

        expected = _to_int(raw.get("expected_chunk_count"))
        expected = _clamp_int(expected, 1, 100000) or max(1, context.paragraph_count)

        rng = raw.get("expected_chunk_count_range")
        if isinstance(rng, list) and len(rng) >= 2:
            lo = _to_int(rng[0])
            hi = _to_int(rng[1])
        else:
            lo, hi = None, None

        lo = _clamp_int(lo, 1, 100000) if lo is not None else max(1, expected - 2)
        hi = _clamp_int(hi, 1, 100000) if hi is not None else expected + 2
        if lo > hi:
            lo, hi = hi, lo

        group_size = _to_int(raw.get("atomize_sentences_per_paragraph"))
        group_size = _clamp_int(group_size, 0, 10)
        if group_size is None:
            # Heuristic fallback: if expected chunks significantly exceed paragraph
            # count, atomize; otherwise disable.
            group_size = 2 if expected > context.paragraph_count else 0

        min_sentences = _to_int(raw.get("atomize_min_sentences"))
        min_sentences = _clamp_int(min_sentences, 0, 50)
        if min_sentences is None:
            min_sentences = context.config.atomize_min_sentences

        rationale = raw.get("rationale", "")
        if not isinstance(rationale, str):
            rationale = ""

        confidence = raw.get("confidence", 0.0)
        try:
            confidence_f = float(confidence)
        except Exception:
            confidence_f = 0.0
        confidence_f = max(0.0, min(1.0, confidence_f))

        plan: Dict[str, Any] = {
            "topics": topics[:50],
            "expected_chunk_count": expected,
            "expected_chunk_count_range": [lo, hi],
            "atomize_sentences_per_paragraph": group_size,
            "atomize_min_sentences": min_sentences,
            "rationale": rationale,
            "confidence": confidence_f,
        }

        if context.config.verbose:
            logger.info("Granularity plan: %s", plan)

        return plan
