"""Debug logger for LLM prompt/response traces."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Optional, Dict, Any


@dataclass
class LLMDebugLogger:
    """Append-only JSONL logger for LLM calls."""

    log_path: Path
    run_id: str
    input_path: str
    model: str

    def __post_init__(self) -> None:
        """Ensure directory exists and write run header."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_event(
            {
                "type": "run_start",
                "run_id": self.run_id,
                "timestamp": self._timestamp(),
                "input_path": self.input_path,
                "model": self.model,
                "log_path": str(self.log_path),
            }
        )

    def log_call(
        self,
        *,
        prompt: str,
        system_prompt: Optional[str],
        response_content: str,
        tokens_used: int,
        success: bool,
        error: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a single LLM call."""
        event: Dict[str, Any] = {
            "type": "llm_call",
            "run_id": self.run_id,
            "timestamp": self._timestamp(),
            "prompt": prompt,
            "system_prompt": system_prompt,
            "response": response_content,
            "tokens_used": tokens_used,
            "success": success,
            "error": error,
        }
        if extra:
            event["extra"] = extra
        self._write_event(event)

    def _write_event(self, event: Dict[str, Any]) -> None:
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=True) + "\n")

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()
