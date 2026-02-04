"""Token counting utilities."""

from typing import Optional


class TokenCounter:
    """Token counter using tiktoken."""

    def __init__(self, model: str = "gpt-4") -> None:
        """Initialize the token counter.

        Args:
            model: Model name for tiktoken encoding
        """
        self._model = model
        self._encoding = None  # Lazy loading

    @property
    def encoding(self):
        """Lazy load the encoding."""
        if self._encoding is None:
            try:
                import tiktoken
                try:
                    self._encoding = tiktoken.encoding_for_model(self._model)
                except KeyError:
                    # Fallback to cl100k_base for unknown models
                    self._encoding = tiktoken.get_encoding("cl100k_base")
            except ImportError:
                raise ImportError(
                    "tiktoken is not installed. Install it with: pip install tiktoken"
                )
        return self._encoding

    def count(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        if not text:
            return 0

        try:
            return len(self.encoding.encode(text))
        except Exception:
            # Fallback to approximate word count
            return self._approximate_count(text)

    def count_many(self, texts: list) -> list:
        """Count tokens for multiple texts.

        Args:
            texts: List of texts to count

        Returns:
            List of token counts
        """
        return [self.count(text) for text in texts]

    def _approximate_count(self, text: str) -> int:
        """Approximate token count based on word count.

        Rough approximation: ~1.3 tokens per word for English.

        Args:
            text: Text to count

        Returns:
            Approximate token count
        """
        words = len(text.split())
        return int(words * 1.3)

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within max tokens.

        Args:
            text: Text to truncate
            max_tokens: Maximum number of tokens

        Returns:
            Truncated text
        """
        if not text:
            return ""

        try:
            tokens = self.encoding.encode(text)
            if len(tokens) <= max_tokens:
                return text
            truncated_tokens = tokens[:max_tokens]
            return self.encoding.decode(truncated_tokens)
        except Exception:
            # Fallback to word-based truncation
            words = text.split()
            max_words = int(max_tokens / 1.3)
            return " ".join(words[:max_words])
