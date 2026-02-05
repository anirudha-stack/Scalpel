"""Document models for Segmenta."""

from dataclasses import dataclass, field
from typing import List, Optional

from segmenta.models.enums import ElementType


@dataclass
class Paragraph:
    """Represents a single paragraph or atomic element in a document."""

    text: str
    index: int
    element_type: ElementType = ElementType.PARAGRAPH
    is_atomic: bool = False
    language: Optional[str] = None  # For code blocks
    source_index: Optional[int] = None  # Original paragraph index (after atomization)

    @property
    def is_splittable(self) -> bool:
        """Check if this paragraph can be split."""
        return not self.is_atomic and self.element_type == ElementType.PARAGRAPH

    def __len__(self) -> int:
        """Return the length of the text."""
        return len(self.text)


@dataclass
class Section:
    """Represents a document section with optional hierarchy."""

    title: str
    level: int  # Header level (1-6 for markdown, 0 for implicit)
    paragraphs: List[Paragraph] = field(default_factory=list)
    subsections: List["Section"] = field(default_factory=list)

    @property
    def all_paragraphs(self) -> List[Paragraph]:
        """Recursively get all paragraphs including subsections."""
        result = list(self.paragraphs)
        for subsection in self.subsections:
            result.extend(subsection.all_paragraphs)
        return result

    @property
    def paragraph_count(self) -> int:
        """Get total paragraph count including subsections."""
        return len(self.all_paragraphs)

    def add_paragraph(self, paragraph: Paragraph) -> None:
        """Add a paragraph to this section."""
        self.paragraphs.append(paragraph)

    def add_subsection(self, subsection: "Section") -> None:
        """Add a subsection to this section."""
        self.subsections.append(subsection)


@dataclass
class Document:
    """Represents a parsed document."""

    filename: str
    file_type: str
    sections: List[Section] = field(default_factory=list)
    raw_text: str = ""

    @property
    def all_paragraphs(self) -> List[Paragraph]:
        """Get all paragraphs in document order."""
        result = []
        for section in self.sections:
            result.extend(section.all_paragraphs)
        return result

    @property
    def paragraph_count(self) -> int:
        """Get total paragraph count."""
        return len(self.all_paragraphs)

    @property
    def all_sections_flat(self) -> List[Section]:
        """Get all sections flattened (including nested)."""
        result = []

        def collect_sections(sections: List[Section]) -> None:
            for section in sections:
                result.append(section)
                collect_sections(section.subsections)

        collect_sections(self.sections)
        return result

    def get_section_for_paragraph(self, paragraph_index: int) -> Optional[Section]:
        """Find the section that contains a paragraph by index."""
        for section in self.all_sections_flat:
            for para in section.paragraphs:
                if para.index == paragraph_index:
                    return section
        return None
