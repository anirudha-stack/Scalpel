"""Parse stage - parses input document."""

import time
from pathlib import Path

from segmenta.pipeline.base import PipelineStage
from segmenta.pipeline.context import PipelineContext
from segmenta.parsers import ParserFactory
from segmenta.exceptions import SegmentaParseError


class ParseStage(PipelineStage):
    """Stage that parses the input document."""

    @property
    def name(self) -> str:
        return "parse"

    def process(self, context: PipelineContext) -> PipelineContext:
        """Parse the input document.

        Args:
            context: Pipeline context

        Returns:
            Updated context with parsed document
        """
        start_time = time.time()

        try:
            input_path = Path(context.input_path)

            if not input_path.exists():
                raise SegmentaParseError(
                    f"Input file not found: {context.input_path}",
                    file_path=context.input_path,
                )

            parser = ParserFactory.create(input_path)
            context.document = parser.parse(input_path)

            # Record metrics
            context.add_metric("parse_time", time.time() - start_time)
            context.add_metric("file_type", context.document.file_type)
            context.add_metric("section_count", len(context.document.sections))

        except SegmentaParseError:
            raise
        except Exception as e:
            raise SegmentaParseError(
                f"Failed to parse document: {e}",
                file_path=context.input_path,
            )

        return context
