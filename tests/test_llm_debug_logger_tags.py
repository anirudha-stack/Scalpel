from __future__ import annotations

import json
from pathlib import Path

from segmenta.utils.llm_debug_logger import LLMDebugLogger


def test_llm_debug_logger_surfaces_call_tags(tmp_path: Path) -> None:
    log_path = tmp_path / "trace.jsonl"
    logger = LLMDebugLogger(
        log_path=log_path,
        run_id="run",
        input_path="x",
        model="m",
    )

    logger.log_call(
        prompt="p",
        system_prompt="s",
        response_content="r",
        tokens_used=1,
        success=True,
        extra={
            "pipeline_stage": "boundary_validate",
            "call_role": "critique",
            "call_kind": "boundary_validate_critique",
            "other": 123,
        },
    )

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 2

    event = json.loads(lines[-1])
    assert event["type"] == "llm_call"
    assert event["pipeline_stage"] == "boundary_validate"
    assert event["call_role"] == "critique"
    assert event["call_kind"] == "boundary_validate_critique"
    assert event["extra"]["other"] == 123

