"""Parse stage - parses input document."""

import time
from pathlib import Path

from microtome.pipeline.base import PipelineStage
from microtome.pipeline.context import PipelineContext
from microtome.parsers import ParserFactory
from microtome.exceptions import MicrotomeParseError


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
                raise MicrotomeParseError(
                    f"Input file not found: {context.input_path}",
                    file_path=context.input_path,
                )

            parser = ParserFactory.create(input_path)
            context.document = parser.parse(input_path)

            # Record metrics
            context.add_metric("parse_time", time.time() - start_time)
            context.add_metric("file_type", context.document.file_type)
            context.add_metric("section_count", len(context.document.sections))

        except MicrotomeParseError:
            raise
        except Exception as e:
            raise MicrotomeParseError(
                f"Failed to parse document: {e}",
                file_path=context.input_path,
            )

        return context
