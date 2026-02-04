"""OpenAI LLM provider."""

import json
import re
from typing import Optional, Dict, Any

from scalpel.llm.base import LLMProvider, LLMResponse
from scalpel.exceptions import ScalpelLLMError


class OpenAIProvider(LLMProvider):
    """LLM provider using OpenAI API."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        temperature: float = 0.1,
        max_tokens: int = 1000,
        base_url: Optional[str] = None,
    ) -> None:
        """Initialize the OpenAI provider.

        Args:
            api_key: OpenAI API key
            model: Model name to use
            temperature: Temperature for generation
            max_tokens: Maximum tokens in response
            base_url: Optional custom base URL for API
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ScalpelLLMError(
                "openai is not installed. Install it with: pip install openai"
            )

        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self._client = OpenAI(**client_kwargs)

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
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )

            content = response.choices[0].message.content or ""
            tokens_used = response.usage.total_tokens if response.usage else 0

            return LLMResponse(
                content=content,
                tokens_used=tokens_used,
                model=self._model,
                success=True,
            )

        except Exception as e:
            return LLMResponse(
                content="",
                tokens_used=0,
                model=self._model,
                success=False,
                error=str(e),
            )

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
            ScalpelLLMError: If the call fails or response is not valid JSON
        """
        response = self.complete(prompt, system_prompt)

        if not response.success:
            raise ScalpelLLMError(f"LLM call failed: {response.error}")

        try:
            # Try to extract JSON from the response
            content = response.content.strip()

            # Handle markdown code blocks
            if content.startswith("```"):
                # Extract content between code fences
                match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
                if match:
                    content = match.group(1)

            return json.loads(content)

        except json.JSONDecodeError as e:
            raise ScalpelLLMError(
                f"Failed to parse JSON response: {e}\nResponse was: {response.content[:500]}"
            )

    @property
    def model_name(self) -> str:
        """Return the model name being used."""
        return self._model
