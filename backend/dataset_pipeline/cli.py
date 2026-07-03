"""Typer-based CLI for the dataset pipeline.

Commands:
  dss dataset ingest   — Full pipeline
  dss dataset catalog  — Catalog stage only
  dss dataset map      — Ontology mapping stage only
  dss dataset convert  — Conversion stage only
  dss dataset quality  — Quality stage only
  dss dataset train    — Training stage only
  dss dataset pipeline — Full pipeline (alias for ingest)
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

import typer

from backend.dataset_pipeline.config import pipeline_config
from backend.dataset_pipeline.models import PipelineResult
from backend.dataset_pipeline.service import PipelineService

app = typer.Typer(
    name="dss",
    help="DSSPrototype dataset pipeline CLI",
    no_args_is_help=True,
)

dataset_app = typer.Typer(
    name="dataset",
    help="Dataset ingestion commands",
    no_args_is_help=True,
)
app.add_typer(dataset_app)

logger = logging.getLogger("dss.dataset_pipeline.cli")


def _setup_logging(verbose: bool) -> None:
    if verbose:
        level = logging.DEBUG
    else:
        level = getattr(logging, pipeline_config.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format=pipeline_config.log_format,
        stream=sys.stdout,
    )


def _build_service() -> PipelineService:
    return PipelineService()


@dataset_app.callback()
def dataset_callback() -> None:
    pass


@dataset_app.command("ingest")
def cmd_ingest(
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset name"),
    source: str = typer.Option(..., "--source", "-s", help="Path to raw dataset"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory"),
    skip_quality: bool = typer.Option(False, "--skip-quality", help="Skip quality checks"),
    skip_training: bool = typer.Option(False, "--skip-training", help="Skip training"),
    continue_on_error: bool = typer.Option(
        False,
        "--continue-on-error",
        help="Continue on stage failure",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate without executing"),
    force: bool = typer.Option(False, "--force", help="Force re-create catalog entry"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
    config_path: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Config file path (not yet implemented)",
    ),
) -> None:
    """Ingest a dataset through the full pipeline."""
    _setup_logging(verbose)
    service = _build_service()
    source_path = Path(source)

    typer.echo(f"Ingesting dataset '{dataset}' from {source_path}...")

    result = asyncio.run(
        service.ingest_dataset(
            dataset_name=dataset,
            source_path=source_path,
            skip_quality=skip_quality,
            skip_training=skip_training,
            continue_on_error=continue_on_error,
            dry_run=dry_run,
            force=force,
            output_dir=Path(output) if output else None,
        ),
    )

    _print_result(result)


@dataset_app.command("catalog")
def cmd_catalog(
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset name"),
    source: str = typer.Option(..., "--source", "-s", help="Path to raw dataset"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate without executing"),
    force: bool = typer.Option(False, "--force", help="Force re-create catalog entry"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
) -> None:
    """Register a dataset in the catalog."""
    _setup_logging(verbose)
    service = _build_service()
    result = asyncio.run(
        service.run_catalog(
            dataset_name=dataset,
            source_path=Path(source),
            dry_run=dry_run,
            force=force,
        ),
    )
    _print_result(result)


@dataset_app.command("map")
def cmd_map(
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset name"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate without executing"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
) -> None:
    """Run ontology mapping for a dataset."""
    _setup_logging(verbose)
    service = _build_service()
    result = asyncio.run(
        service.run_mapping(dataset_name=dataset, dry_run=dry_run),
    )
    _print_result(result)


@dataset_app.command("convert")
def cmd_convert(
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset name"),
    source: str = typer.Option(..., "--source", "-s", help="Path to raw dataset"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate without executing"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
) -> None:
    """Convert a dataset to canonical format."""
    _setup_logging(verbose)
    service = _build_service()
    result = asyncio.run(
        service.run_conversion(dataset_name=dataset, source_path=Path(source), dry_run=dry_run),
    )
    _print_result(result)


@dataset_app.command("quality")
def cmd_quality(
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset name"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate without executing"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
) -> None:
    """Run quality checks on a dataset."""
    _setup_logging(verbose)
    service = _build_service()
    result = asyncio.run(
        service.run_quality(dataset_name=dataset, dry_run=dry_run),
    )
    _print_result(result)


@dataset_app.command("train")
def cmd_train(
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset name"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate without executing"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
) -> None:
    """Train a model using a dataset."""
    _setup_logging(verbose)
    service = _build_service()
    result = asyncio.run(
        service.run_training(dataset_name=dataset, dry_run=dry_run),
    )
    _print_result(result)


@dataset_app.command("pipeline")
def cmd_pipeline(
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset name"),
    source: str = typer.Option(..., "--source", "-s", help="Path to raw dataset"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory"),
    skip_quality: bool = typer.Option(False, "--skip-quality", help="Skip quality checks"),
    skip_training: bool = typer.Option(False, "--skip-training", help="Skip training"),
    continue_on_error: bool = typer.Option(
        False,
        "--continue-on-error",
        help="Continue on stage failure",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate without executing"),
    force: bool = typer.Option(False, "--force", help="Force re-create catalog entry"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
    config_path: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Config file path (not yet implemented)",
    ),
) -> None:
    """Run the full dataset pipeline (alias for ingest)."""
    _setup_logging(verbose)
    service = _build_service()
    source_path = Path(source)

    result = asyncio.run(
        service.run_full_pipeline(
            dataset_name=dataset,
            source_path=source_path,
            skip_quality=skip_quality,
            skip_training=skip_training,
            continue_on_error=continue_on_error,
            dry_run=dry_run,
            force=force,
            output_dir=Path(output) if output else None,
        ),
    )

    _print_result(result)


def _print_result(result: PipelineResult) -> None:
    typer.echo("")
    typer.echo("=" * 60)
    typer.echo(f"Pipeline Result: {result.status.value.upper()}")
    typer.echo(f"Dataset: {result.dataset_name}")
    if result.error:
        typer.echo(f"Error: {result.error}")
    typer.echo("-" * 60)

    for stage in result.stages:
        icon = {
            "completed": "✓",
            "failed": "✗",
            "skipped": "−",
            "running": "▶",
            "pending": "○",
        }.get(stage.status.value, "?")
        dur = f" ({stage.duration_seconds:.2f}s)" if stage.duration_seconds else ""
        typer.echo(f"  {icon} {stage.stage}: {stage.status.value}{dur}")
        if stage.error:
            typer.echo(f"     Error: {stage.error}")

    typer.echo("-" * 60)
    typer.echo(
        f"Summary: {result.summary.stages_completed} completed, "
        f"{result.summary.stages_failed} failed, "
        f"{result.summary.stages_skipped} skipped "
        f"in {result.summary.total_duration_seconds:.2f}s",
    )
    typer.echo("=" * 60)


if __name__ == "__main__":
    app()
