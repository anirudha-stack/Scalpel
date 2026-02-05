"""Output stage - writes chunks to output file."""

import time
from pathlib import Path
from typing import TYPE_CHECKING

from segmenta.pipeline.base import PipelineStage
from segmenta.pipeline.context import PipelineContext

if TYPE_CHECKING:
    from segmenta.output.base import OutputFormatter


class OutputStage(PipelineStage):
    """Stage that writes chunks to output file."""

    def __init__(self, formatter: "OutputFormatter") -> None:
        """Initialize the output stage.

        Args:
            formatter: Output formatter to use
        """
        self._formatter = formatter

    @property
    def name(self) -> str:
        return "output"

    def process(self, context: PipelineContext) -> PipelineContext:
        """Write chunks to output file.

        Args:
            context: Pipeline context

        Returns:
            Updated context with output path
        """
        start_time = time.time()

        if not context.chunks:
            context.add_warning("No chunks to write")
            context.add_metric("output_time", time.time() - start_time)
            return context

        # Ensure output directory exists
        output_dir = Path(context.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate output filename
        if context.document:
            input_name = Path(context.document.filename).stem
        else:
            input_name = Path(context.input_path).stem

        output_filename = f"Segmenta_output_{input_name}.md"
        output_path = output_dir / output_filename

        # Format and write output
        context.output_path = self._formatter.format(
            chunks=context.chunks,
            output_path=str(output_path),
        )

        # Record metrics
        context.add_metric("output_time", time.time() - start_time)
        context.add_metric("output_path", context.output_path)

        return context
