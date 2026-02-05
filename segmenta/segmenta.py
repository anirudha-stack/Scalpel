"""Main Segmenta class - entry point for the library."""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Optional, Callable, List

from segmenta.config import SegmentaConfig
from segmenta.models import SegmentaResult, Chunk
from segmenta.llm.base import LLMProvider
from segmenta.llm.openai_provider import OpenAIProvider
from segmenta.embeddings.base import EmbeddingProvider
from segmenta.embeddings.sentence_transformer import SentenceTransformerProvider
from segmenta.output.base import OutputFormatter
from segmenta.output.markdown_formatter import MarkdownFormatter
from segmenta.utils.token_counter import TokenCounter
from segmenta.utils.retry import RetryHandler
from segmenta.utils.llm_debug_logger import LLMDebugLogger
from segmenta.pipeline.orchestrator import PipelineOrchestrator
from segmenta.pipeline.stages import (
    ParseStage,
    SegmentStage,
    GranularityStage,
    AtomizeStage,
    BoundaryDetectStage,
    BoundaryValidateStage,
    ChunkFormStage,
    EnrichStage,
    OutputStage,
)
from segmenta.exceptions import SegmentaConfigError

logger = logging.getLogger(__name__)

class Segmenta:
    """Main Segmenta class for semantic document chunking."""

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model: str = "gpt-4o",
        config: Optional[SegmentaConfig] = None,
        llm_provider: Optional[LLMProvider] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        output_formatter: Optional[OutputFormatter] = None,
    ) -> None:
        """Initialize Segmenta.

        Args:
            openai_api_key: OpenAI API key (required if not using custom llm_provider)
            model: OpenAI model name
            config: Configuration object
            llm_provider: Custom LLM provider (overrides openai_api_key/model)
            embedding_provider: Custom embedding provider
            output_formatter: Custom output formatter
        """
        self._config = config or SegmentaConfig()

        # Set up LLM provider
        if llm_provider:
            self._llm_provider = llm_provider
        elif openai_api_key:
            self._llm_provider = OpenAIProvider(
                api_key=openai_api_key,
                model=model,
                temperature=self._config.llm_temperature,
                max_tokens=self._config.llm_max_tokens,
            )
        else:
            raise SegmentaConfigError(
                "Either openai_api_key or llm_provider must be provided"
            )

        # Set up embedding provider
        if embedding_provider:
            self._embedding_provider = embedding_provider
        else:
            self._embedding_provider = SentenceTransformerProvider(
                model_name=self._config.embedding_model,
                device=self._config.embedding_device,
            )

        # Set up output formatter
        self._output_formatter = output_formatter or MarkdownFormatter()

        # Set up token counter
        self._token_counter = TokenCounter(model=self._config.token_model)

        # Set up retry handler
        self._retry_handler = RetryHandler(
            max_attempts=self._config.retry_attempts,
            base_delay=self._config.retry_delay,
        )

    @classmethod
    def builder(cls) -> "SegmentaBuilder":
        """Create a builder for fluent configuration.

        Returns:
            SegmentaBuilder instance
        """
        return SegmentaBuilder()

    def chunk(
        self,
        input_file: str,
        output_dir: str = "./output",
        progress_callback: Optional[Callable[[str, float], None]] = None,
        dry_run: bool = False,
    ) -> SegmentaResult:
        """Process a document and generate semantic chunks.

        Args:
            input_file: Path to input document
            output_dir: Output directory for results
            progress_callback: Optional callback for progress updates
            dry_run: If True, skip LLM calls

        Returns:
            SegmentaResult with chunks and metadata
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        input_name = Path(input_file).stem or "input"
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_id = f"{input_name}_{timestamp}"
        debug_log_path = output_path / f"Segmenta_llm_debug_{run_id}.jsonl"
        debug_logger = LLMDebugLogger(
            log_path=debug_log_path,
            run_id=run_id,
            input_path=input_file,
            model=self._llm_provider.model_name,
        )
        if hasattr(self._llm_provider, "set_debug_logger"):
            self._llm_provider.set_debug_logger(debug_logger)
        else:
            logger.info(
                "LLM provider does not support debug logging; skipping LLM prompt log."
            )

        # Build pipeline stages
        stages = [
            ParseStage(),
            SegmentStage(),
            GranularityStage(llm_provider=self._llm_provider),
            AtomizeStage(),
            BoundaryDetectStage(
                embedding_provider=self._embedding_provider,
                threshold=self._config.similarity_threshold,
            ),
            BoundaryValidateStage(
                llm_provider=self._llm_provider,
                retry_handler=self._retry_handler,
            ),
            ChunkFormStage(),
            EnrichStage(
                llm_provider=self._llm_provider,
                token_counter=self._token_counter,
                retry_handler=self._retry_handler,
                fallback_enabled=self._config.fallback_enabled,
            ),
            OutputStage(formatter=self._output_formatter),
        ]

        # Create and execute pipeline
        orchestrator = PipelineOrchestrator(
            stages=stages,
            config=self._config,
            progress_callback=progress_callback,
        )

        context = orchestrator.execute(
            input_path=input_file,
            output_dir=output_dir,
            dry_run=dry_run,
        )

        # Write granularity plan (if generated) to output_dir for auditability.
        plan = context.metrics.get("granularity_plan")
        if isinstance(plan, dict):
            plan_path = output_path / f"Segmenta_granularity_plan_{run_id}.json"
            try:
                plan_path.write_text(
                    json.dumps(plan, indent=2, ensure_ascii=True) + "\n",
                    encoding="utf-8",
                )
                context.add_metric("granularity_plan_path", str(plan_path))
            except Exception as e:
                context.add_warning(f"Failed to write granularity plan file: {e}")

        # Build result
        return SegmentaResult(
            success=not context.has_errors,
            output_path=context.output_path,
            chunks=context.chunks,
            metrics=context.metrics,
            warnings=context.warnings,
            errors=context.errors,
        )

    def chunk_text(
        self,
        text: str,
        filename: str = "input.txt",
    ) -> List[Chunk]:
        """Process text directly and return chunks.

        Args:
            text: Text content to chunk
            filename: Virtual filename for the content

        Returns:
            List of chunks
        """
        import tempfile
        import os

        # Write text to temporary file
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".txt",
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write(text)
            temp_path = f.name

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                result = self.chunk(
                    input_file=temp_path,
                    output_dir=temp_dir,
                )
                return result.chunks
        finally:
            os.unlink(temp_path)

    @property
    def config(self) -> SegmentaConfig:
        """Get the configuration."""
        return self._config


class SegmentaBuilder:
    """Builder for fluent Segmenta configuration."""

    def __init__(self) -> None:
        """Initialize the builder."""
        self._config: Optional[SegmentaConfig] = None
        self._llm_provider: Optional[LLMProvider] = None
        self._embedding_provider: Optional[EmbeddingProvider] = None
        self._output_formatter: Optional[OutputFormatter] = None
        self._openai_api_key: Optional[str] = None
        self._model: str = "gpt-4o"

    def with_config(self, config: SegmentaConfig) -> "SegmentaBuilder":
        """Set configuration.

        Args:
            config: SegmentaConfig instance

        Returns:
            Self for chaining
        """
        self._config = config
        return self

    def with_llm_provider(self, provider: LLMProvider) -> "SegmentaBuilder":
        """Set LLM provider.

        Args:
            provider: LLMProvider instance

        Returns:
            Self for chaining
        """
        self._llm_provider = provider
        return self

    def with_embedding_provider(
        self, provider: EmbeddingProvider
    ) -> "SegmentaBuilder":
        """Set embedding provider.

        Args:
            provider: EmbeddingProvider instance

        Returns:
            Self for chaining
        """
        self._embedding_provider = provider
        return self

    def with_output_formatter(self, formatter: OutputFormatter) -> "SegmentaBuilder":
        """Set output formatter.

        Args:
            formatter: OutputFormatter instance

        Returns:
            Self for chaining
        """
        self._output_formatter = formatter
        return self

    def with_openai(self, api_key: str, model: str = "gpt-4o") -> "SegmentaBuilder":
        """Configure OpenAI provider.

        Args:
            api_key: OpenAI API key
            model: Model name

        Returns:
            Self for chaining
        """
        self._openai_api_key = api_key
        self._model = model
        return self

    def build(self) -> Segmenta:
        """Build the Segmenta instance.

        Returns:
            Configured Segmenta instance
        """
        return Segmenta(
            openai_api_key=self._openai_api_key,
            model=self._model,
            config=self._config,
            llm_provider=self._llm_provider,
            embedding_provider=self._embedding_provider,
            output_formatter=self._output_formatter,
        )
