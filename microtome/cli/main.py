"""CLI entry point for Microtome."""

import os
import sys
import logging

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.panel import Panel

from microtome import Microtome, MicrotomeConfig, __version__
from microtome.exceptions import MicrotomeError

console = Console()


def setup_logging(verbose: bool) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


@click.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "-o",
    "--output-dir",
    default="./output",
    help="Output directory",
    type=click.Path(),
)
@click.option(
    "-m",
    "--model",
    default="gpt-4o",
    help="OpenAI model name",
)
@click.option(
    "-k",
    "--api-key",
    envvar="OPENAI_API_KEY",
    help="OpenAI API key (or set OPENAI_API_KEY env var)",
)
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True),
    help="Path to configuration YAML file",
)
@click.option(
    "-t",
    "--threshold",
    default=0.5,
    type=float,
    help="Similarity threshold for boundaries (0.0-1.0)",
)
@click.option(
    "--min-tokens",
    default=50,
    type=int,
    help="Minimum tokens per chunk",
)
@click.option(
    "--max-tokens",
    default=500,
    type=int,
    help="Maximum tokens per chunk",
)
@click.option(
    "--plan-granularity/--no-plan-granularity",
    default=True,
    help="Use an initial LLM pass to plan atomizer granularity (recommended for PDFs).",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Enable verbose output",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Parse document without LLM calls",
)
@click.version_option(version=__version__)
def cli(
    input_file: str,
    output_dir: str,
    model: str,
    api_key: str,
    config: str,
    threshold: float,
    min_tokens: int,
    max_tokens: int,
    plan_granularity: bool,
    verbose: bool,
    dry_run: bool,
) -> None:
    """Microtome: Semantic document chunking library.

    Transform documents into semantically coherent, metadata-enriched chunks.

    INPUT_FILE: Path to input document (PDF, MD, or TXT)
    """
    setup_logging(verbose)

    # Validate API key
    if not api_key and not dry_run:
        console.print(
            "[red]Error:[/red] OpenAI API key required. "
            "Set OPENAI_API_KEY environment variable or use --api-key option."
        )
        sys.exit(1)

    # Load config from file or create from options
    try:
        if config:
            microtome_config = MicrotomeConfig.from_yaml(config)
        else:
            microtome_config = MicrotomeConfig(
                similarity_threshold=threshold,
                min_chunk_tokens=min_tokens,
                max_chunk_tokens=max_tokens,
                granularity_planning_enabled=plan_granularity,
                verbose=verbose,
            )
    except MicrotomeError as e:
        console.print(f"[red]Configuration error:[/red] {e}")
        sys.exit(1)

    # Display header
    console.print(
        Panel.fit(
            f"[bold blue]Microtome v{__version__}[/bold blue]\n"
            f"Semantic Document Chunking",
            border_style="blue",
        )
    )
    console.print()

    # Initialize Microtome
    try:
        microtome = Microtome(
            openai_api_key=api_key or "dummy-key-for-dry-run",
            model=model,
            config=microtome_config,
        )
    except MicrotomeError as e:
        console.print(f"[red]Initialization error:[/red] {e}")
        sys.exit(1)

    # Process with progress indicator
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Processing...", total=100)

        def on_progress(stage: str, pct: float) -> None:
            progress.update(task, completed=int(pct * 100), description=f"[cyan]{stage}")

        try:
            result = microtome.chunk(
                input_file=input_file,
                output_dir=output_dir,
                progress_callback=on_progress,
                dry_run=dry_run,
            )
        except MicrotomeError as e:
            console.print(f"\n[red]Processing error:[/red] {e}")
            sys.exit(1)
        except Exception as e:
            console.print(f"\n[red]Unexpected error:[/red] {e}")
            if verbose:
                console.print_exception()
            sys.exit(1)

    console.print()

    # Display results
    if result.success:
        console.print(f"[green]✓[/green] Generated {result.chunk_count} chunks")
        if result.output_path:
            console.print(f"[green]✓[/green] Output: {result.output_path}")

        # Show metrics if verbose
        if verbose and result.metrics:
            console.print()
            metrics_table = Table(title="Metrics", show_header=True)
            metrics_table.add_column("Metric", style="cyan")
            metrics_table.add_column("Value", style="green")

            for key, value in result.metrics.items():
                if isinstance(value, float):
                    metrics_table.add_row(key, f"{value:.3f}")
                elif isinstance(value, dict):
                    metrics_table.add_row(key, str(value))
                else:
                    metrics_table.add_row(key, str(value))

            console.print(metrics_table)

        # Show warnings
        if result.warnings:
            console.print()
            for warning in result.warnings:
                console.print(f"[yellow]Warning:[/yellow] {warning}")

    else:
        console.print("[red]✗[/red] Processing failed")
        for error in result.errors:
            console.print(f"  [red]•[/red] {error}")
        sys.exit(1)


def main() -> None:
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
