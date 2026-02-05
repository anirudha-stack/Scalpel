"""Parser factory for Segmenta."""

from pathlib import Path
from typing import Dict, Type, Optional, List

from segmenta.parsers.base import DocumentParser
from segmenta.parsers.text_parser import TextParser
from segmenta.parsers.markdown_parser import MarkdownParser
from segmenta.parsers.pdf_parser import PDFParser
from segmenta.exceptions import UnsupportedFileTypeError


class ParserFactory:
    """Factory for creating document parsers based on file type."""

    _parsers: Dict[str, Type[DocumentParser]] = {}
    _initialized: bool = False

    @classmethod
    def _ensure_initialized(cls) -> None:
        """Ensure default parsers are registered."""
        if not cls._initialized:
            cls._register_defaults()
            cls._initialized = True

    @classmethod
    def _register_defaults(cls) -> None:
        """Register default parsers."""
        # Text parser
        for ext in TextParser().supported_extensions():
            cls._parsers[ext.lower()] = TextParser

        # Markdown parser
        for ext in MarkdownParser().supported_extensions():
            cls._parsers[ext.lower()] = MarkdownParser

        # PDF parser
        for ext in PDFParser().supported_extensions():
            cls._parsers[ext.lower()] = PDFParser

    @classmethod
    def register(cls, extension: str, parser_class: Type[DocumentParser]) -> None:
        """Register a parser class for a file extension.

        Args:
            extension: File extension (e.g., '.docx')
            parser_class: Parser class to use for this extension
        """
        cls._ensure_initialized()
        cls._parsers[extension.lower()] = parser_class

    @classmethod
    def unregister(cls, extension: str) -> None:
        """Unregister a parser for a file extension.

        Args:
            extension: File extension to unregister
        """
        cls._ensure_initialized()
        ext_lower = extension.lower()
        if ext_lower in cls._parsers:
            del cls._parsers[ext_lower]

    @classmethod
    def create(cls, file_path: str | Path) -> DocumentParser:
        """Create a parser for the given file.

        Args:
            file_path: Path to the file to parse

        Returns:
            Appropriate DocumentParser instance

        Raises:
            UnsupportedFileTypeError: If no parser is registered for the file type
        """
        cls._ensure_initialized()

        path = Path(file_path)
        extension = path.suffix.lower()

        if extension not in cls._parsers:
            raise UnsupportedFileTypeError(
                extension,
                supported_types=list(cls._parsers.keys()),
            )

        return cls._parsers[extension]()

    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """Get list of all supported file extensions.

        Returns:
            List of supported extensions
        """
        cls._ensure_initialized()
        return list(cls._parsers.keys())

    @classmethod
    def is_supported(cls, file_path: str | Path) -> bool:
        """Check if a file type is supported.

        Args:
            file_path: Path to the file

        Returns:
            True if supported, False otherwise
        """
        cls._ensure_initialized()
        path = Path(file_path)
        return path.suffix.lower() in cls._parsers

    @classmethod
    def get_parser_class(cls, extension: str) -> Optional[Type[DocumentParser]]:
        """Get the parser class for a file extension.

        Args:
            extension: File extension

        Returns:
            Parser class or None if not found
        """
        cls._ensure_initialized()
        return cls._parsers.get(extension.lower())

    @classmethod
    def reset(cls) -> None:
        """Reset factory to default state (useful for testing)."""
        cls._parsers.clear()
        cls._initialized = False
