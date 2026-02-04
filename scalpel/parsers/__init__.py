"""Document parsers for Scalpel."""

from scalpel.parsers.base import DocumentParser
from scalpel.parsers.text_parser import TextParser
from scalpel.parsers.markdown_parser import MarkdownParser
from scalpel.parsers.pdf_parser import PDFParser
from scalpel.parsers.factory import ParserFactory

__all__ = [
    "DocumentParser",
    "TextParser",
    "MarkdownParser",
    "PDFParser",
    "ParserFactory",
]
