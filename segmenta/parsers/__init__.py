"""Document parsers for Segmenta."""

from segmenta.parsers.base import DocumentParser
from segmenta.parsers.text_parser import TextParser
from segmenta.parsers.markdown_parser import MarkdownParser
from segmenta.parsers.pdf_parser import PDFParser
from segmenta.parsers.factory import ParserFactory

__all__ = [
    "DocumentParser",
    "TextParser",
    "MarkdownParser",
    "PDFParser",
    "ParserFactory",
]
