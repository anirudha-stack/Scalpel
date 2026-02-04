"""Pipeline orchestrator - coordinates execution of pipeline stages."""

import logging
import time
from typing import List, Optional, Callable

from scalpel.pipeline.base import PipelineStage
from scalpel.pipeline.context import PipelineContext
from scalpel.config import ScalpelConfig

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Coordinates execution of pipeline stages."""

    def __init__(
        self,
        stages: List[PipelineStage],
        config: ScalpelConfig,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> None:
        """Initialize the pipeline orchestrator.

        Args:
            stages: List of pipeline stages to execute
            config: Scalpel configuration
            progress_callback: Optional callback for progress updates
        """
        self._stages = stages
        self._config = config
        self._progress_callback = progress_callback

    def execute(
        self,
        input_path: str,
        output_dir: str,
        dry_run: bool = False,
    ) -> PipelineContext:
        """Execute the pipeline on an input file.

        Args:
            input_path: Path to input file
            output_dir: Output directory
            dry_run: If True, skip LLM calls

        Returns:
            Pipeline context with results
        """
        start_time = time.time()

        context = PipelineContext(
            input_path=input_path,
            output_dir=output_dir,
            config=self._config,
            dry_run=dry_run,
        )

        total_stages = len(self._stages)

        for i, stage in enumerate(self._stages):
            if context.should_stop:
                logger.warning(f"Pipeline stopped early at stage: {stage.name}")
                break

            if stage.should_skip(context):
                logger.info(f"Skipping stage: {stage.name}")
                if self._progress_callback:
                    progress = (i + 1) / total_stages
                    self._progress_callback(f"Skipped: {stage.name}", progress)
                continue

            logger.info(f"Executing stage: {stage.name}")

            try:
                stage_start = time.time()
                context = stage.process(context)
                stage_time = time.time() - stage_start

                if self._config.verbose:
                    logger.info(f"Stage {stage.name} completed in {stage_time:.2f}s")

            except Exception as e:
                error_msg = f"{stage.name}: {str(e)}"
                logger.error(f"Error in stage {stage.name}: {e}")
                context.add_error(error_msg)

                if not self._config.continue_on_error:
                    context.should_stop = True
                    break

            if self._progress_callback:
                progress = (i + 1) / total_stages
                self._progress_callback(stage.name, progress)

        # Record total time
        context.add_metric("total_time", time.time() - start_time)

        return context

    def add_stage(self, stage: PipelineStage) -> None:
        """Add a stage to the pipeline.

        Args:
            stage: Stage to add
        """
        self._stages.append(stage)

    def insert_stage(self, index: int, stage: PipelineStage) -> None:
        """Insert a stage at a specific position.

        Args:
            index: Position to insert at
            stage: Stage to insert
        """
        self._stages.insert(index, stage)

    def remove_stage(self, stage_name: str) -> bool:
        """Remove a stage by name.

        Args:
            stage_name: Name of stage to remove

        Returns:
            True if stage was found and removed
        """
        for i, stage in enumerate(self._stages):
            if stage.name == stage_name:
                self._stages.pop(i)
                return True
        return False

    @property
    def stage_names(self) -> List[str]:
        """Get list of stage names in order."""
        return [stage.name for stage in self._stages]
