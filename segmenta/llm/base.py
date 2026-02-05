"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class LLMResponse:
    """Response from an LLM call."""

    content: str
    tokens_used: int
    model: str
    success: bool = True
    error: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None

    @property
    def failed(self) -> bool:
        """Check if the response indicates failure."""
        return not self.success


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def complete(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """Generate a completion for the given prompt.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt

        Returns:
            LLMResponse with the generated content
        """
        pass

    @abstractmethod
    def complete_json(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate a JSON-structured completion.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt

        Returns:
            Parsed JSON response as a dictionary

        Raises:
            SegmentaLLMError: If the call fails or response is not valid JSON
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name being used."""
        pass
