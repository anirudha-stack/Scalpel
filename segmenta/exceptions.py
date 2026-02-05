"""Custom exceptions for Segmenta."""

from typing import Optional


class SegmentaError(Exception):
    """Base exception for all Segmenta errors."""

    pass


class SegmentaConfigError(SegmentaError):
    """Raised when configuration is invalid."""

    def __init__(self, message: str, config_key: Optional[str] = None):
        super().__init__(message)
        self.config_key = config_key


class SegmentaParseError(SegmentaError):
    """Raised when document parsing fails."""

    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        position: Optional[int] = None,
    ):
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

    def __init__(self, message: str, model_name: Optional[str] = None):
        super().__init__(message)
        self.model_name = model_name


class UnsupportedFileTypeError(SegmentaError):
    """Raised when file type is not supported."""

    def __init__(self, file_type: str, supported_types: Optional[list] = None):
        message = f"Unsupported file type: {file_type}"
        if supported_types:
            message += f". Supported types: {', '.join(supported_types)}"
        super().__init__(message)
        self.file_type = file_type
        self.supported_types = supported_types or []


class SegmentaPipelineError(SegmentaError):
    """Raised when pipeline execution fails."""

    def __init__(self, message: str, stage_name: Optional[str] = None):
        super().__init__(message)
        self.stage_name = stage_name


class SegmentaValidationError(SegmentaError):
    """Raised when validation fails."""

    def __init__(self, message: str, field_name: Optional[str] = None):
        super().__init__(message)
        self.field_name = field_name
