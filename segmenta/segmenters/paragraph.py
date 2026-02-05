"""Paragraph splitter for handling long paragraphs."""

import re
from typing import List, TYPE_CHECKING

from segmenta.models import Paragraph, ElementType

if TYPE_CHECKING:
    from segmenta.utils.token_counter import TokenCounter


class ParagraphSplitter:
    """Handles splitting of very long paragraphs at sentence boundaries."""

    def __init__(self, max_tokens: int, token_counter: "TokenCounter") -> None:
        """Initialize the paragraph splitter.

        Args:
            max_tokens: Maximum tokens per paragraph
            token_counter: Token counter instance
        """
        self._max_tokens = max_tokens
        self._token_counter = token_counter

    def split_if_needed(self, paragraph: Paragraph) -> List[Paragraph]:
        """Split paragraph if it exceeds max tokens.

        Args:
            paragraph: Paragraph to potentially split

        Returns:
            List of paragraphs (original if no split needed, or split parts)
        """
        if paragraph.is_atomic:
            # Never split atomic elements (code blocks, tables)
            return [paragraph]

        token_count = self._token_counter.count(paragraph.text)
        if token_count <= self._max_tokens:
            return [paragraph]

        # Split at sentence boundaries
        return self._split_by_sentences(paragraph)

    def _split_by_sentences(self, paragraph: Paragraph) -> List[Paragraph]:
        """Split paragraph into sentence groups that fit within token limit.

        Args:
            paragraph: Paragraph to split

        Returns:
            List of split paragraphs
        """
        # Sentence splitting pattern
        # Handles periods, exclamation marks, question marks followed by space or end
        sentences = re.split(r"(?<=[.!?])\s+", paragraph.text)

        if len(sentences) <= 1:
            # Can't split further, return as-is
            return [paragraph]

        result: List[Paragraph] = []
        current_sentences: List[str] = []
        current_tokens = 0
        base_index = paragraph.index

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_tokens = self._token_counter.count(sentence)

            if (
                current_tokens + sentence_tokens > self._max_tokens
                and current_sentences
            ):
                # Create paragraph from current sentences
                result.append(
                    Paragraph(
                        text=" ".join(current_sentences),
                        index=base_index + len(result),
                        element_type=paragraph.element_type,
                        is_atomic=False,
                    )
                )
                current_sentences = [sentence]
                current_tokens = sentence_tokens
            else:
                current_sentences.append(sentence)
                current_tokens += sentence_tokens

        # Add remaining sentences
        if current_sentences:
            result.append(
                Paragraph(
                    text=" ".join(current_sentences),
                    index=base_index + len(result),
                    element_type=paragraph.element_type,
                    is_atomic=False,
                )
            )

        return result if result else [paragraph]

    def process_paragraphs(self, paragraphs: List[Paragraph]) -> List[Paragraph]:
        """Process a list of paragraphs, splitting any that are too long.

        Args:
            paragraphs: List of paragraphs to process

        Returns:
            List of paragraphs with long ones split
        """
        result: List[Paragraph] = []
        current_index = 0

        for paragraph in paragraphs:
            split_paragraphs = self.split_if_needed(paragraph)

            # Update indices
            for split_para in split_paragraphs:
                result.append(
                    Paragraph(
                        text=split_para.text,
                        index=current_index,
                        element_type=split_para.element_type,
                        is_atomic=split_para.is_atomic,
                        language=split_para.language,
                    )
                )
                current_index += 1

        return result
