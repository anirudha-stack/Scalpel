"""Enumerations for Microtome models."""

from enum import Enum


class ElementType(str, Enum):
    """Type of document element."""

    PARAGRAPH = "paragraph"
    CODE_BLOCK = "code_block"
    TABLE = "table"
    LIST = "list"
    HEADER = "header"
    BLOCKQUOTE = "blockquote"


class Intent(str, Enum):
    """Intent/purpose of a chunk."""

    EXPLAINS = "explains"
    LISTS = "lists"
    WARNS = "warns"
    DEFINES = "defines"
    INSTRUCTS = "instructs"
    COMPARES = "compares"
    UNKNOWN = "unknown"

    @classmethod
    def from_string(cls, value: str) -> "Intent":
        """Convert string to Intent, defaulting to UNKNOWN."""
        try:
            return cls(value.lower())
        except ValueError:
            return cls.UNKNOWN


class BoundaryVerdict(str, Enum):
    """LLM decision on a boundary proposal."""

    KEEP = "KEEP"
    MERGE = "MERGE"
    ADJUST = "ADJUST"

    @classmethod
    def from_string(cls, value: str) -> "BoundaryVerdict":
        """Convert string to BoundaryVerdict, defaulting to KEEP."""
        try:
            return cls(value.upper())
        except ValueError:
            return cls.KEEP
