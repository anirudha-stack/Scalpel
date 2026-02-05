"""Plain text document parser."""

from pathlib import Path
from typing import List

from segmenta.models import Document, Section, Paragraph, ElementType
from segmenta.parsers.base import DocumentParser
from segmenta.exceptions import SegmentaParseError


class TextParser(DocumentParser):
    """Parser for plain text documents."""

    def supported_extensions(self) -> List[str]:
        """Return supported file extensions."""
        return [".txt", ".text"]

    def parse(self, file_path: Path) -> Document:
        """Parse a plain text file into a Document model.

        Args:
            file_path: Path to the text file

        Returns:
            Parsed Document model

        Raises:
            SegmentaParseError: If parsing fails
        """
        file_path = Path(file_path)

        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Try with latin-1 as fallback
            try:
                content = file_path.read_text(encoding="latin-1")
            except Exception as e:
                raise SegmentaParseError(
                    f"Failed to read text file: {e}",
                    file_path=str(file_path),
                )
        except Exception as e:
            raise SegmentaParseError(
                f"Failed to read text file: {e}",
                file_path=str(file_path),
            )

        # Split into paragraphs by double newlines
        raw_paragraphs = content.split("\n\n")
        paragraphs = []
        paragraph_index = 0

        for raw_para in raw_paragraphs:
            # Clean up the paragraph
            text = raw_para.strip()
            if not text:
                continue

            # Replace single newlines with spaces
            text = " ".join(text.split())

            paragraphs.append(
                Paragraph(
                    text=text,
                    index=paragraph_index,
                    element_type=ElementType.PARAGRAPH,
                    is_atomic=False,
                )
            )
            paragraph_index += 1

        # Create a single implicit section for plain text
        section = Section(
            title="",
            level=0,
            paragraphs=paragraphs,
        )

        return Document(
            filename=file_path.name,
            file_type="text",
            sections=[section] if paragraphs else [],
            raw_text=content,
        )
