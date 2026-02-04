"""Abstract base class for document parsers."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from scalpel.models import Document


class DocumentParser(ABC):
    """Abstract base class for document parsers."""

    @abstractmethod
    def parse(self, file_path: Path) -> Document:
        """Parse a document file into a Document model.

        Args:
            file_path: Path to the document file

        Returns:
            Parsed Document model

        Raises:
            ScalpelParseError: If parsing fails
        """
        pass

    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """Return list of supported file extensions.

        Returns:
            List of extensions like ['.pdf', '.PDF']
        """
        pass

    def supports(self, file_extension: str) -> bool:
        """Check if this parser supports the given file extension.

        Args:
            file_extension: File extension including the dot (e.g., '.pdf')

        Returns:
            True if supported, False otherwise
        """
        return file_extension.lower() in [ext.lower() for ext in self.supported_extensions()]
