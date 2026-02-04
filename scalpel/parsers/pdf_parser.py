"""PDF document parser."""

from pathlib import Path
from typing import List, Optional, Dict, Any

import fitz  # PyMuPDF

from scalpel.models import Document, Section, Paragraph, ElementType
from scalpel.parsers.base import DocumentParser
from scalpel.exceptions import ScalpelParseError


class PDFParser(DocumentParser):
    """Parser for PDF documents using PyMuPDF."""

    # Font size thresholds for header detection
    HEADER_SIZE_THRESHOLD = 14.0
    SUBHEADER_SIZE_THRESHOLD = 12.0

    def supported_extensions(self) -> List[str]:
        """Return supported file extensions."""
        return [".pdf"]

    def parse(self, file_path: Path) -> Document:
        """Parse a PDF file into a Document model.

        Args:
            file_path: Path to the PDF file

        Returns:
            Parsed Document model

        Raises:
            ScalpelParseError: If parsing fails
        """
        file_path = Path(file_path)

        try:
            doc = fitz.open(file_path)
        except Exception as e:
            raise ScalpelParseError(
                f"Failed to open PDF file: {e}",
                file_path=str(file_path),
            )

        try:
            sections: List[Section] = []
            paragraph_index = 0
            current_section: Optional[Section] = None

            for page_num in range(len(doc)):
                page = doc[page_num]

                # Extract text blocks with detailed info
                blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)[
                    "blocks"
                ]

                for block in blocks:
                    if block.get("type") != 0:  # Skip non-text blocks (images, etc.)
                        continue

                    text, font_info = self._extract_block_text(block)
                    if not text.strip():
                        continue

                    # Detect if this might be a header
                    is_header, header_level = self._analyze_header(text, font_info)

                    if is_header:
                        new_section = Section(
                            title=text.strip(),
                            level=header_level,
                            paragraphs=[],
                        )

                        if header_level == 1 or current_section is None:
                            sections.append(new_section)
                        else:
                            # Add as subsection or sibling based on level
                            self._add_section(sections, new_section, header_level)

                        current_section = new_section
                    else:
                        paragraph = Paragraph(
                            text=text.strip(),
                            index=paragraph_index,
                            element_type=ElementType.PARAGRAPH,
                            is_atomic=False,
                        )
                        paragraph_index += 1

                        if current_section is not None:
                            current_section.paragraphs.append(paragraph)
                        else:
                            # Create implicit section
                            if not sections:
                                sections.append(
                                    Section(title="", level=0, paragraphs=[])
                                )
                            sections[-1].paragraphs.append(paragraph)

            # Get raw text before closing
            raw_text = self._get_raw_text(file_path)

            return Document(
                filename=file_path.name,
                file_type="pdf",
                sections=sections,
                raw_text=raw_text,
            )

        except ScalpelParseError:
            raise
        except Exception as e:
            raise ScalpelParseError(
                f"Failed to parse PDF: {e}",
                file_path=str(file_path),
            )
        finally:
            try:
                doc.close()
            except Exception:
                pass  # Already closed or never opened

    def _extract_block_text(self, block: Dict[str, Any]) -> tuple:
        """Extract text and font information from a PDF block.

        Returns:
            Tuple of (text, font_info dict)
        """
        lines = []
        font_sizes = []
        is_bold = False

        for line in block.get("lines", []):
            line_text_parts = []
            for span in line.get("spans", []):
                text = span.get("text", "")
                line_text_parts.append(text)

                # Track font info
                font_sizes.append(span.get("size", 12.0))
                flags = span.get("flags", 0)
                if flags & 2**4:  # Bold flag
                    is_bold = True

            line_text = " ".join(line_text_parts)
            lines.append(line_text)

        text = " ".join(lines)

        font_info = {
            "avg_size": sum(font_sizes) / len(font_sizes) if font_sizes else 12.0,
            "max_size": max(font_sizes) if font_sizes else 12.0,
            "is_bold": is_bold,
        }

        return text, font_info

    def _analyze_header(
        self, text: str, font_info: Dict[str, Any]
    ) -> tuple:
        """Analyze if text is likely a header based on font and content.

        Returns:
            Tuple of (is_header, header_level)
        """
        # Check font size
        avg_size = font_info.get("avg_size", 12.0)
        is_bold = font_info.get("is_bold", False)

        # Heuristics for header detection
        is_short = len(text.split()) <= 10
        no_punctuation = not text.rstrip().endswith((".", ",", ";", ":"))

        if avg_size >= self.HEADER_SIZE_THRESHOLD and is_short:
            return True, 1
        elif avg_size >= self.SUBHEADER_SIZE_THRESHOLD and is_bold and is_short:
            return True, 2
        elif is_bold and is_short and no_punctuation:
            return True, 3

        return False, 0

    def _add_section(
        self, sections: List[Section], new_section: Section, level: int
    ) -> None:
        """Add a section at the appropriate hierarchy level."""
        if not sections:
            sections.append(new_section)
            return

        # Find the right parent
        last_section = sections[-1]

        if level > last_section.level:
            # Add as subsection
            last_section.subsections.append(new_section)
        else:
            # Add as sibling
            sections.append(new_section)

    def _get_raw_text(self, file_path: Path) -> str:
        """Extract raw text from PDF for reference."""
        try:
            doc = fitz.open(file_path)
            text_parts = []
            for page in doc:
                text_parts.append(page.get_text())
            doc.close()
            return "\n".join(text_parts)
        except Exception:
            return ""
