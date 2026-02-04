"""Main Scalpel class - entry point for the library."""

from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Optional, Callable, List

from scalpel.config import ScalpelConfig
from scalpel.models import ScalpelResult, Chunk
from scalpel.llm.base import LLMProvider
from scalpel.llm.openai_provider import OpenAIProvider
from scalpel.embeddings.base import EmbeddingProvider
from scalpel.embeddings.sentence_transformer import SentenceTransformerProvider
from scalpel.output.base import OutputFormatter
from scalpel.output.markdown_formatter import MarkdownFormatter
from scalpel.utils.token_counter import TokenCounter
from scalpel.utils.retry import RetryHandler
from scalpel.utils.llm_debug_logger import LLMDebugLogger
from scalpel.pipeline.orchestrator import PipelineOrchestrator
from scalpel.pipeline.stages import (
    ParseStage,
    SegmentStage,
    AtomizeStage,
    BoundaryDetectStage,
    BoundaryValidateStage,
    ChunkFormStage,
    EnrichStage,
    OutputStage,
)
from scalpel.exceptions import ScalpelConfigError

logger = logging.getLogger(__name__)

class Scalpel:
    """Main Scalpel class for semantic document chunking."""

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model: str = "gpt-4o",
        config: Optional[ScalpelConfig] = None,
        llm_provider: Optional[LLMProvider] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        output_formatter: Optional[OutputFormatter] = None,
    ) -> None:
        """Initialize Scalpel.

        Args:
            openai_api_key: OpenAI API key (required if not using custom llm_provider)
            model: OpenAI model name
            config: Configuration object
            llm_provider: Custom LLM provider (overrides openai_api_key/model)
            embedding_provider: Custom embedding provider
            output_formatter: Custom output formatter
        """
        self._config = config or ScalpelConfig()

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
            raise ScalpelConfigError(
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
    def builder(cls) -> "ScalpelBuilder":
        """Create a builder for fluent configuration.

        Returns:
            ScalpelBuilder instance
        """
        return ScalpelBuilder()

    def chunk(
        self,
        input_file: str,
        output_dir: str = "./output",
        progress_callback: Optional[Callable[[str, float], None]] = None,
        dry_run: bool = False,
    ) -> ScalpelResult:
        """Process a document and generate semantic chunks.

        Args:
            input_file: Path to input document
            output_dir: Output directory for results
            progress_callback: Optional callback for progress updates
            dry_run: If True, skip LLM calls

        Returns:
            ScalpelResult with chunks and metadata
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        input_name = Path(input_file).stem or "input"
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_id = f"{input_name}_{timestamp}"
        debug_log_path = output_path / f"Scalpel_llm_debug_{run_id}.jsonl"
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

        # Build result
        return ScalpelResult(
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
    def config(self) -> ScalpelConfig:
        """Get the configuration."""
        return self._config


class ScalpelBuilder:
    """Builder for fluent Scalpel configuration."""

    def __init__(self) -> None:
        """Initialize the builder."""
        self._config: Optional[ScalpelConfig] = None
        self._llm_provider: Optional[LLMProvider] = None
        self._embedding_provider: Optional[EmbeddingProvider] = None
        self._output_formatter: Optional[OutputFormatter] = None
        self._openai_api_key: Optional[str] = None
        self._model: str = "gpt-4o"

    def with_config(self, config: ScalpelConfig) -> "ScalpelBuilder":
        """Set configuration.

        Args:
            config: ScalpelConfig instance

        Returns:
            Self for chaining
        """
        self._config = config
        return self

    def with_llm_provider(self, provider: LLMProvider) -> "ScalpelBuilder":
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
    ) -> "ScalpelBuilder":
        """Set embedding provider.

        Args:
            provider: EmbeddingProvider instance

        Returns:
            Self for chaining
        """
        self._embedding_provider = provider
        return self

    def with_output_formatter(self, formatter: OutputFormatter) -> "ScalpelBuilder":
        """Set output formatter.

        Args:
            formatter: OutputFormatter instance

        Returns:
            Self for chaining
        """
        self._output_formatter = formatter
        return self

    def with_openai(self, api_key: str, model: str = "gpt-4o") -> "ScalpelBuilder":
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

    def build(self) -> Scalpel:
        """Build the Scalpel instance.

        Returns:
            Configured Scalpel instance
        """
        return Scalpel(
            openai_api_key=self._openai_api_key,
            model=self._model,
            config=self._config,
            llm_provider=self._llm_provider,
            embedding_provider=self._embedding_provider,
            output_formatter=self._output_formatter,
        )
