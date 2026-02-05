"""Configuration for Scalpel."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any
import os

import yaml

from scalpel.exceptions import ScalpelConfigError


@dataclass
class ScalpelConfig:
    """Configuration for Scalpel processing."""

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

    embedding_device: Optional[str] = None
    """Device for embedding model (cuda, cpu, or None for auto)."""

    # Token counting
    token_model: str = "gpt-4"
    """Model name for tiktoken token counting."""

    # LLM settings
    llm_model: str = "gpt-4o"
    """Default LLM model name."""

    llm_temperature: float = 0.1
    """Temperature for LLM calls."""

    llm_max_tokens: int = 1000
    """Maximum tokens for LLM response."""

    # Fine-grained segmentation (optional)
    atomize_sentences_per_paragraph: int = 2
    """If >0, split long paragraphs into sentence groups of this size before boundary detection."""

    atomize_min_sentences: int = 6
    """Only atomize paragraphs that have at least this many sentences."""

    granularity_planning_enabled: bool = True
    """If True, run an initial LLM pass to plan atomization granularity."""

    granularity_max_paragraphs: int = 60
    """Maximum number of paragraphs to include in the granularity planning sample."""

    granularity_max_chars_per_paragraph: int = 280
    """Maximum characters per paragraph included in the granularity planning sample."""

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate()

    def _validate(self) -> None:
        """Validate configuration values."""
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ScalpelConfigError(
                f"similarity_threshold must be between 0.0 and 1.0, got {self.similarity_threshold}",
                "similarity_threshold",
            )

        if self.min_chunk_tokens < 0:
            raise ScalpelConfigError(
                f"min_chunk_tokens must be non-negative, got {self.min_chunk_tokens}",
                "min_chunk_tokens",
            )

        if self.max_chunk_tokens < self.min_chunk_tokens:
            raise ScalpelConfigError(
                f"max_chunk_tokens ({self.max_chunk_tokens}) must be >= min_chunk_tokens ({self.min_chunk_tokens})",
                "max_chunk_tokens",
            )

        if self.retry_attempts < 0:
            raise ScalpelConfigError(
                f"retry_attempts must be non-negative, got {self.retry_attempts}",
                "retry_attempts",
            )

        if self.short_document_threshold < 1:
            raise ScalpelConfigError(
                f"short_document_threshold must be at least 1, got {self.short_document_threshold}",
                "short_document_threshold",
            )

        if self.atomize_sentences_per_paragraph < 0:
            raise ScalpelConfigError(
                f"atomize_sentences_per_paragraph must be >= 0, got {self.atomize_sentences_per_paragraph}",
                "atomize_sentences_per_paragraph",
            )

        if self.atomize_min_sentences < 0:
            raise ScalpelConfigError(
                f"atomize_min_sentences must be >= 0, got {self.atomize_min_sentences}",
                "atomize_min_sentences",
            )

        if self.granularity_max_paragraphs < 1:
            raise ScalpelConfigError(
                f"granularity_max_paragraphs must be >= 1, got {self.granularity_max_paragraphs}",
                "granularity_max_paragraphs",
            )

        if self.granularity_max_chars_per_paragraph < 1:
            raise ScalpelConfigError(
                f"granularity_max_chars_per_paragraph must be >= 1, got {self.granularity_max_chars_per_paragraph}",
                "granularity_max_chars_per_paragraph",
            )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScalpelConfig":
        """Create configuration from dictionary."""
        # Handle nested structure from YAML
        flat_data = {}

        # Extract from nested sections if present
        if "chunking" in data:
            chunking = data["chunking"]
            flat_data["similarity_threshold"] = chunking.get(
                "similarity_threshold", 0.5
            )
            flat_data["min_chunk_tokens"] = chunking.get("min_tokens", 50)
            flat_data["max_chunk_tokens"] = chunking.get("max_tokens", 500)
            flat_data["atomize_sentences_per_paragraph"] = chunking.get(
                "atomize_sentences_per_paragraph", 2
            )
            flat_data["atomize_min_sentences"] = chunking.get(
                "atomize_min_sentences", 6
            )
            flat_data["granularity_planning_enabled"] = chunking.get(
                "granularity_planning_enabled", True
            )
            flat_data["granularity_max_paragraphs"] = chunking.get(
                "granularity_max_paragraphs", 60
            )
            flat_data["granularity_max_chars_per_paragraph"] = chunking.get(
                "granularity_max_chars_per_paragraph", 280
            )

        if "behavior" in data:
            behavior = data["behavior"]
            flat_data["retry_attempts"] = behavior.get("retry_attempts", 2)
            flat_data["fallback_enabled"] = behavior.get("fallback_enabled", True)
            flat_data["continue_on_error"] = behavior.get("continue_on_error", False)
            flat_data["verbose"] = behavior.get("verbose", False)

        if "embedding" in data:
            embedding = data["embedding"]
            flat_data["embedding_model"] = embedding.get("model", "all-MiniLM-L6-v2")
            flat_data["embedding_device"] = embedding.get("device")

        if "llm" in data:
            llm = data["llm"]
            flat_data["llm_model"] = llm.get("model", "gpt-4o")
            flat_data["llm_temperature"] = llm.get("temperature", 0.1)

        # Also accept flat keys
        for key in [
            "similarity_threshold",
            "min_chunk_tokens",
            "max_chunk_tokens",
            "retry_attempts",
            "retry_delay",
            "fallback_enabled",
            "continue_on_error",
            "verbose",
            "short_document_threshold",
            "embedding_model",
            "embedding_device",
            "token_model",
            "llm_model",
            "llm_temperature",
            "llm_max_tokens",
            "atomize_sentences_per_paragraph",
            "atomize_min_sentences",
            "granularity_planning_enabled",
            "granularity_max_paragraphs",
            "granularity_max_chars_per_paragraph",
        ]:
            if key in data and key not in flat_data:
                flat_data[key] = data[key]

        return cls(**flat_data)

    @classmethod
    def from_yaml(cls, path: str) -> "ScalpelConfig":
        """Load configuration from YAML file."""
        file_path = Path(path)
        if not file_path.exists():
            raise ScalpelConfigError(f"Config file not found: {path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ScalpelConfigError(f"Invalid YAML in config file: {e}")

        if data is None:
            data = {}

        # Handle environment variable substitution
        data = cls._substitute_env_vars(data)

        return cls.from_dict(data)

    @classmethod
    def _substitute_env_vars(cls, data: Any) -> Any:
        """Recursively substitute environment variables in config."""
        if isinstance(data, dict):
            return {k: cls._substitute_env_vars(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [cls._substitute_env_vars(item) for item in data]
        elif isinstance(data, str) and data.startswith("${") and data.endswith("}"):
            env_var = data[2:-1]
            return os.environ.get(env_var, "")
        return data

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "similarity_threshold": self.similarity_threshold,
            "min_chunk_tokens": self.min_chunk_tokens,
            "max_chunk_tokens": self.max_chunk_tokens,
            "retry_attempts": self.retry_attempts,
            "retry_delay": self.retry_delay,
            "fallback_enabled": self.fallback_enabled,
            "continue_on_error": self.continue_on_error,
            "verbose": self.verbose,
            "short_document_threshold": self.short_document_threshold,
            "embedding_model": self.embedding_model,
            "embedding_device": self.embedding_device,
            "token_model": self.token_model,
            "llm_model": self.llm_model,
            "llm_temperature": self.llm_temperature,
            "llm_max_tokens": self.llm_max_tokens,
            "atomize_sentences_per_paragraph": self.atomize_sentences_per_paragraph,
            "atomize_min_sentences": self.atomize_min_sentences,
            "granularity_planning_enabled": self.granularity_planning_enabled,
            "granularity_max_paragraphs": self.granularity_max_paragraphs,
            "granularity_max_chars_per_paragraph": self.granularity_max_chars_per_paragraph,
        }
