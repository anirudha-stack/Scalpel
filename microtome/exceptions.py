"""Custom exceptions for Microtome."""

from typing import Optional


class MicrotomeError(Exception):
    """Base exception for all Microtome errors."""

    pass


class MicrotomeConfigError(MicrotomeError):
    """Raised when configuration is invalid."""

    def __init__(self, message: str, config_key: Optional[str] = None):
        super().__init__(message)
        self.config_key = config_key


class MicrotomeParseError(MicrotomeError):
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


class MicrotomeLLMError(MicrotomeError):
    """Raised when LLM operations fail."""

    def __init__(self, message: str, retries_attempted: int = 0):
        super().__init__(message)
        self.retries_attempted = retries_attempted


class MicrotomeEmbeddingError(MicrotomeError):
    """Raised when embedding operations fail."""

    def __init__(self, message: str, model_name: Optional[str] = None):
        super().__init__(message)
        self.model_name = model_name


class UnsupportedFileTypeError(MicrotomeError):
    """Raised when file type is not supported."""

    def __init__(self, file_type: str, supported_types: Optional[list] = None):
        message = f"Unsupported file type: {file_type}"
        if supported_types:
            message += f". Supported types: {', '.join(supported_types)}"
        super().__init__(message)
        self.file_type = file_type
        self.supported_types = supported_types or []


class MicrotomePipelineError(MicrotomeError):
    """Raised when pipeline execution fails."""

    def __init__(self, message: str, stage_name: Optional[str] = None):
        super().__init__(message)
        self.stage_name = stage_name


class MicrotomeValidationError(MicrotomeError):
    """Raised when validation fails."""

    def __init__(self, message: str, field_name: Optional[str] = None):
        super().__init__(message)
        self.field_name = field_name
