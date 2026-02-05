"""Markdown document parser."""

from pathlib import Path
from typing import List, Optional, Tuple

from markdown_it import MarkdownIt

from segmenta.models import Document, Section, Paragraph, ElementType
from segmenta.parsers.base import DocumentParser
from segmenta.exceptions import SegmentaParseError


class MarkdownParser(DocumentParser):
    """Parser for Markdown documents."""

    def __init__(self) -> None:
        """Initialize the Markdown parser."""
        self._md = MarkdownIt()

    def supported_extensions(self) -> List[str]:
        """Return supported file extensions."""
        return [".md", ".markdown", ".mdown", ".mkd"]

    def parse(self, file_path: Path) -> Document:
        """Parse a Markdown file into a Document model.

        Args:
            file_path: Path to the Markdown file

        Returns:
            Parsed Document model

        Raises:
            SegmentaParseError: If parsing fails
        """
        file_path = Path(file_path)

        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as e:
            raise SegmentaParseError(
                f"Failed to read Markdown file: {e}",
                file_path=str(file_path),
            )

        try:
            tokens = self._md.parse(content)
        except Exception as e:
            raise SegmentaParseError(
                f"Failed to parse Markdown: {e}",
                file_path=str(file_path),
            )

        sections: List[Section] = []
        section_stack: List[Tuple[int, Section]] = []  # (level, section)
        current_section: Optional[Section] = None
        paragraph_index = 0

        i = 0
        while i < len(tokens):
            token = tokens[i]

            if token.type == "heading_open":
                level = int(token.tag[1])  # h1 -> 1, h2 -> 2, etc.

                # Get heading content from next token
                heading_content = ""
                if i + 1 < len(tokens) and tokens[i + 1].type == "inline":
                    heading_content = tokens[i + 1].content

                new_section = Section(
                    title=heading_content,
                    level=level,
                    paragraphs=[],
                )

                # Handle section hierarchy
                self._add_section_to_hierarchy(
                    sections, section_stack, new_section, level
                )
                current_section = new_section

                # Skip heading_open, inline, heading_close
                i += 3
                continue

            elif token.type == "paragraph_open":
                # Get paragraph content from next token
                para_content = ""
                if i + 1 < len(tokens) and tokens[i + 1].type == "inline":
                    para_content = tokens[i + 1].content

                if para_content.strip():
                    paragraph = Paragraph(
                        text=para_content,
                        index=paragraph_index,
                        element_type=ElementType.PARAGRAPH,
                        is_atomic=False,
                    )
                    paragraph_index += 1
                    self._add_paragraph(sections, current_section, paragraph)

                # Skip paragraph_open, inline, paragraph_close
                i += 3
                continue

            elif token.type == "fence":  # Code block
                paragraph = Paragraph(
                    text=token.content.rstrip(),
                    index=paragraph_index,
                    element_type=ElementType.CODE_BLOCK,
                    is_atomic=True,
                    language=token.info if token.info else None,
                )
                paragraph_index += 1
                self._add_paragraph(sections, current_section, paragraph)
                i += 1
                continue

            elif token.type == "code_block":  # Indented code block
                paragraph = Paragraph(
                    text=token.content.rstrip(),
                    index=paragraph_index,
                    element_type=ElementType.CODE_BLOCK,
                    is_atomic=True,
                )
                paragraph_index += 1
                self._add_paragraph(sections, current_section, paragraph)
                i += 1
                continue

            elif token.type == "table_open":
                table_content, end_index = self._extract_table(tokens, i, content)
                paragraph = Paragraph(
                    text=table_content,
                    index=paragraph_index,
                    element_type=ElementType.TABLE,
                    is_atomic=True,
                )
                paragraph_index += 1
                self._add_paragraph(sections, current_section, paragraph)
                i = end_index + 1
                continue

            elif token.type == "blockquote_open":
                quote_content, end_index = self._extract_blockquote(tokens, i)
                if quote_content.strip():
                    paragraph = Paragraph(
                        text=quote_content,
                        index=paragraph_index,
                        element_type=ElementType.BLOCKQUOTE,
                        is_atomic=False,
                    )
                    paragraph_index += 1
                    self._add_paragraph(sections, current_section, paragraph)
                i = end_index + 1
                continue

            elif token.type in ("bullet_list_open", "ordered_list_open"):
                list_content, end_index = self._extract_list(tokens, i)
                if list_content.strip():
                    paragraph = Paragraph(
                        text=list_content,
                        index=paragraph_index,
                        element_type=ElementType.LIST,
                        is_atomic=False,
                    )
                    paragraph_index += 1
                    self._add_paragraph(sections, current_section, paragraph)
                i = end_index + 1
                continue

            i += 1

        return Document(
            filename=file_path.name,
            file_type="markdown",
            sections=sections,
            raw_text=content,
        )

    def _add_section_to_hierarchy(
        self,
        sections: List[Section],
        section_stack: List[Tuple[int, Section]],
        new_section: Section,
        level: int,
    ) -> None:
        """Add a section to the hierarchy based on heading level."""
        # Pop sections from stack that are at same or higher level
        while section_stack and section_stack[-1][0] >= level:
            section_stack.pop()

        if section_stack:
            # Add as subsection of the section on top of stack
            parent = section_stack[-1][1]
            parent.subsections.append(new_section)
        else:
            # Add as top-level section
            sections.append(new_section)

        # Push new section onto stack
        section_stack.append((level, new_section))

    def _add_paragraph(
        self,
        sections: List[Section],
        current_section: Optional[Section],
        paragraph: Paragraph,
    ) -> None:
        """Add a paragraph to the current section or create implicit section."""
        if current_section is not None:
            current_section.paragraphs.append(paragraph)
        else:
            # Create implicit section for content before first header
            if not sections or sections[0].title:
                implicit_section = Section(title="", level=0, paragraphs=[])
                sections.insert(0, implicit_section)
            sections[0].paragraphs.append(paragraph)

    def _extract_table(
        self, tokens: list, start_index: int, raw_content: str
    ) -> Tuple[str, int]:
        """Extract table content from tokens."""
        # Find table_close
        end_index = start_index
        for i in range(start_index, len(tokens)):
            if tokens[i].type == "table_close":
                end_index = i
                break

        # Build table content from tokens
        rows = []
        current_row = []
        in_row = False

        for i in range(start_index, end_index + 1):
            token = tokens[i]
            if token.type == "tr_open":
                in_row = True
                current_row = []
            elif token.type == "tr_close":
                in_row = False
                if current_row:
                    rows.append(current_row)
            elif token.type == "inline" and in_row:
                current_row.append(token.content)

        # Format as markdown table
        if not rows:
            return "", end_index

        # Build table string
        table_lines = []
        for i, row in enumerate(rows):
            table_lines.append("| " + " | ".join(row) + " |")
            if i == 0:
                # Add separator after header
                table_lines.append("|" + "|".join(["---"] * len(row)) + "|")

        return "\n".join(table_lines), end_index

    def _extract_blockquote(self, tokens: list, start_index: int) -> Tuple[str, int]:
        """Extract blockquote content from tokens."""
        end_index = start_index
        depth = 1

        for i in range(start_index + 1, len(tokens)):
            if tokens[i].type == "blockquote_open":
                depth += 1
            elif tokens[i].type == "blockquote_close":
                depth -= 1
                if depth == 0:
                    end_index = i
                    break

        # Extract content
        content_parts = []
        for i in range(start_index, end_index + 1):
            if tokens[i].type == "inline":
                content_parts.append(tokens[i].content)

        return "> " + "\n> ".join(content_parts), end_index

    def _extract_list(self, tokens: list, start_index: int) -> Tuple[str, int]:
        """Extract list content from tokens."""
        is_ordered = tokens[start_index].type == "ordered_list_open"
        close_type = "ordered_list_close" if is_ordered else "bullet_list_close"

        end_index = start_index
        depth = 1

        for i in range(start_index + 1, len(tokens)):
            if tokens[i].type in ("bullet_list_open", "ordered_list_open"):
                depth += 1
            elif tokens[i].type in ("bullet_list_close", "ordered_list_close"):
                depth -= 1
                if depth == 0:
                    end_index = i
                    break

        # Extract list items
        items = []
        item_num = 1

        for i in range(start_index, end_index + 1):
            if tokens[i].type == "inline":
                if is_ordered:
                    items.append(f"{item_num}. {tokens[i].content}")
                    item_num += 1
                else:
                    items.append(f"- {tokens[i].content}")

        return "\n".join(items), end_index
