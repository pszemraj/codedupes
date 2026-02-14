"""Command-line interface for codedupes."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Literal, cast

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from codedupes import __version__
from codedupes.analyzer import AnalyzerConfig, CodeAnalyzer
from codedupes.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_C2LLM_REVISION,
    DEFAULT_MIN_SEMANTIC_LINES,
    DEFAULT_MODEL,
    DEFAULT_SEMANTIC_THRESHOLD,
    DEFAULT_TOP_K,
    DEFAULT_TRADITIONAL_THRESHOLD,
)
from codedupes.models import AnalysisResult, CodeUnit, DuplicatePair, HybridDuplicate

DEFAULT_THRESHOLD = DEFAULT_SEMANTIC_THRESHOLD
DEFAULT_MIN_LINES = DEFAULT_MIN_SEMANTIC_LINES
DEFAULT_OUTPUT_WIDTH = 160
MIN_OUTPUT_WIDTH = 80

console = Console(width=DEFAULT_OUTPUT_WIDTH)

_NOISY_EXTERNAL_LOGGERS = (
    "deepspeed",
    "httpx",
    "huggingface_hub",
    "jax",
    "numexpr",
    "sentence_transformers",
    "tensorflow",
    "torch.utils.cpp_extension",
    "transformers",
    "urllib3",
)


class _CodedupesLogFilter(logging.Filter):
    """Filter log records so non-codedupes INFO chatter is hidden by default."""

    def __init__(self, *, include_external_info: bool) -> None:
        super().__init__()
        self.include_external_info = include_external_info

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name.startswith("codedupes"):
            return True
        if self.include_external_info:
            return True
        return record.levelno >= logging.WARNING


def _set_console(output_width: int) -> None:
    """Set global console used by all rich output helpers."""
    global console
    console = Console(width=output_width)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging with rich handler."""
    level = logging.DEBUG if verbose else logging.INFO
    handler = RichHandler(console=console, show_time=False, show_path=False)
    handler.addFilter(_CodedupesLogFilter(include_external_info=verbose))
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[handler],
        force=True,
    )
    quiet_level = logging.DEBUG if verbose else logging.WARNING
    for logger_name in _NOISY_EXTERNAL_LOGGERS:
        logging.getLogger(logger_name).setLevel(quiet_level)


def _validate_threshold(
    _ctx: click.Context, _param: click.Parameter, value: float | None
) -> float | None:
    if value is None:
        return None
    if not 0.0 <= value <= 1.0:
        raise click.BadParameter("must be in [0.0, 1.0]")
    return value


def _validate_positive_int(_ctx: click.Context, _param: click.Parameter, value: int) -> int:
    if value <= 0:
        raise click.BadParameter("must be > 0")
    return value


def _validate_non_negative_int(_ctx: click.Context, _param: click.Parameter, value: int) -> int:
    if value < 0:
        raise click.BadParameter("must be >= 0")
    return value


def _validate_output_width(_ctx: click.Context, _param: click.Parameter, value: int) -> int:
    if value < MIN_OUTPUT_WIDTH:
        raise click.BadParameter(f"must be >= {MIN_OUTPUT_WIDTH}")
    return value


def _resolve_threshold(base: float, override: float | None) -> float:
    return override if override is not None else base


def _resolve_check_thresholds(
    threshold: float,
    semantic_threshold: float | None,
    traditional_threshold: float | None,
) -> tuple[float, float]:
    return (
        _resolve_threshold(threshold, semantic_threshold),
        _resolve_threshold(threshold, traditional_threshold),
    )


def format_location(unit: CodeUnit) -> str:
    """Format file:line location."""
    return f"{unit.file_path.name}:{unit.lineno}"


def truncate_source(source: str, max_lines: int = 5) -> str:
    """Truncate source code for display."""
    lines = source.strip().split("\n")
    if len(lines) <= max_lines:
        return source.strip()
    return "\n".join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more lines)"


def print_summary(
    result: AnalysisResult,
    *,
    mode: Literal["combined", "traditional", "semantic"],
) -> None:
    """Print analysis summary."""
    console.print()

    summary = Table(title="Analysis Summary", show_header=False, box=None)
    summary.add_column(style="bold cyan", no_wrap=True)
    summary.add_column(style="white", no_wrap=True)

    summary.add_row("Total code units", str(len(result.units)))
    summary.add_row(
        "  Functions",
        str(sum(1 for unit in result.units if unit.unit_type.name.lower() == "function")),
    )
    summary.add_row(
        "  Methods",
        str(sum(1 for unit in result.units if unit.unit_type.name.lower() == "method")),
    )
    summary.add_row(
        "  Classes",
        str(sum(1 for unit in result.units if unit.unit_type.name.lower() == "class")),
    )
    summary.add_row("", "")

    if mode == "combined":
        summary.add_row("Hybrid duplicates", str(len(result.hybrid_duplicates)))
        summary.add_row("Likely dead code", str(len(result.potentially_unused)))
        summary.add_row("", "")
        summary.add_row("Raw traditional duplicates", str(len(result.traditional_duplicates)))
        summary.add_row("Raw semantic duplicates", str(len(result.semantic_duplicates)))
        summary.add_row("Filtered raw duplicates", str(result.filtered_raw_duplicates))
    elif mode == "traditional":
        summary.add_row("Traditional duplicates", str(len(result.traditional_duplicates)))
        summary.add_row("Potentially unused", str(len(result.potentially_unused)))
    else:
        summary.add_row("Semantic duplicates", str(len(result.semantic_duplicates)))
        summary.add_row("Potentially unused", str(len(result.potentially_unused)))

    console.print(summary)
    console.print()


def _unit_to_dict(unit: CodeUnit) -> dict[str, Any]:
    return {
        "name": unit.name,
        "qualified_name": unit.qualified_name,
        "type": unit.unit_type.name.lower(),
        "file": str(unit.file_path),
        "line": unit.lineno,
        "end_line": unit.end_lineno,
        "is_public": unit.is_public,
        "is_exported": unit.is_exported,
    }


def _dup_to_dict(dup: DuplicatePair) -> dict[str, Any]:
    return {
        "unit_a": _unit_to_dict(dup.unit_a),
        "unit_b": _unit_to_dict(dup.unit_b),
        "similarity": dup.similarity,
        "method": dup.method,
    }


def _hybrid_dup_to_dict(dup: HybridDuplicate) -> dict[str, Any]:
    return {
        "unit_a": _unit_to_dict(dup.unit_a),
        "unit_b": _unit_to_dict(dup.unit_b),
        "tier": dup.tier,
        "confidence": dup.confidence,
        "has_exact": dup.has_exact,
        "semantic_similarity": dup.semantic_similarity,
        "jaccard_similarity": dup.jaccard_similarity,
        "weak_identifier_jaccard": dup.weak_identifier_jaccard,
        "statement_count_ratio": dup.statement_count_ratio,
    }


def print_check_json_combined(result: AnalysisResult, *, show_all: bool) -> None:
    """Output combined-mode check results as JSON."""
    output: dict[str, Any] = {
        "summary": {
            "total_units": len(result.units),
            "hybrid_duplicates": len(result.hybrid_duplicates),
            "potentially_unused": len(result.potentially_unused),
            "raw_traditional_duplicates": len(result.traditional_duplicates),
            "raw_semantic_duplicates": len(result.semantic_duplicates),
            "filtered_raw_duplicates": result.filtered_raw_duplicates,
        },
        "hybrid_duplicates": [
            _hybrid_dup_to_dict(duplicate) for duplicate in result.hybrid_duplicates
        ],
        "potentially_unused": [_unit_to_dict(unit) for unit in result.potentially_unused],
    }
    if show_all:
        output["traditional_duplicates"] = [
            _dup_to_dict(duplicate) for duplicate in result.traditional_duplicates
        ]
        output["semantic_duplicates"] = [
            _dup_to_dict(duplicate) for duplicate in result.semantic_duplicates
        ]

    print(json.dumps(output, indent=2, sort_keys=True))


def print_check_json_raw(result: AnalysisResult) -> None:
    """Output raw single-method check results as JSON."""
    output = {
        "summary": {
            "total_units": len(result.units),
            "traditional_duplicates": len(result.traditional_duplicates),
            "semantic_duplicates": len(result.semantic_duplicates),
            "potentially_unused": len(result.potentially_unused),
        },
        "traditional_duplicates": [
            _dup_to_dict(duplicate) for duplicate in result.traditional_duplicates
        ],
        "semantic_duplicates": [
            _dup_to_dict(duplicate) for duplicate in result.semantic_duplicates
        ],
        "potentially_unused": [_unit_to_dict(unit) for unit in result.potentially_unused],
    }
    print(json.dumps(output, indent=2, sort_keys=True))


def print_search_json(query: str, results: list[tuple[CodeUnit, float]]) -> None:
    """Output search output as JSON."""
    payload = {
        "query": query,
        "results": [{"score": float(score), **_unit_to_dict(unit)} for unit, score in results],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def _build_duplicates_table(*, hybrid: bool = False) -> Table:
    table = Table(show_header=True, header_style="bold")
    if hybrid:
        table.add_column("Confidence", style="green", width=10, no_wrap=True)
        table.add_column("Tier", style="magenta", no_wrap=True)
        table.add_column("Semantic", style="green", width=10, no_wrap=True)
        table.add_column("Jaccard", style="green", width=10, no_wrap=True)
        table.add_column("Unit A", style="cyan", no_wrap=True)
        table.add_column("Unit B", style="cyan", no_wrap=True)
    else:
        table.add_column("Similarity", style="green", width=10, no_wrap=True)
        table.add_column("Unit A", style="cyan", no_wrap=True)
        table.add_column("Unit B", style="cyan", no_wrap=True)
        table.add_column("Method", style="dim", no_wrap=True)
    return table


def _print_source_panels(unit_a: CodeUnit, unit_b: CodeUnit) -> None:
    console.print(
        Panel(
            Syntax(truncate_source(unit_a.source), "python", theme="monokai"),
            title=f"[cyan]{unit_a.qualified_name}[/cyan]",
            border_style="dim",
        )
    )
    console.print(
        Panel(
            Syntax(truncate_source(unit_b.source), "python", theme="monokai"),
            title=f"[cyan]{unit_b.qualified_name}[/cyan]",
            border_style="dim",
        )
    )


def _print_duplicate_table(
    duplicates: list[DuplicatePair] | list[HybridDuplicate],
    *,
    title: str,
    show_source: bool,
    max_items: int,
    hybrid: bool,
) -> None:
    """Render duplicate pairs in either raw or hybrid layout."""
    if not duplicates:
        return

    console.print(f"\n[bold yellow]{title}[/bold yellow] ({len(duplicates)} pairs)")
    table = _build_duplicates_table(hybrid=hybrid)

    for duplicate in duplicates[:max_items]:
        if hybrid:
            pair = cast(HybridDuplicate, duplicate)
            semantic = (
                f"{pair.semantic_similarity:.2%}" if pair.semantic_similarity is not None else "-"
            )
            jaccard = (
                f"{pair.jaccard_similarity:.2%}" if pair.jaccard_similarity is not None else "-"
            )
            table.add_row(
                f"{pair.confidence:.2%}",
                pair.tier,
                semantic,
                jaccard,
                f"{pair.unit_a.name}\n[dim]{format_location(pair.unit_a)}[/dim]",
                f"{pair.unit_b.name}\n[dim]{format_location(pair.unit_b)}[/dim]",
            )
            unit_a = pair.unit_a
            unit_b = pair.unit_b
        else:
            pair = cast(DuplicatePair, duplicate)
            table.add_row(
                f"{pair.similarity:.2%}",
                f"{pair.unit_a.name}\n[dim]{format_location(pair.unit_a)}[/dim]",
                f"{pair.unit_b.name}\n[dim]{format_location(pair.unit_b)}[/dim]",
                pair.method,
            )
            unit_a = pair.unit_a
            unit_b = pair.unit_b

        if show_source:
            console.print(table)
            _print_source_panels(unit_a, unit_b)
            table = _build_duplicates_table(hybrid=hybrid)

    if not show_source:
        console.print(table)

    if len(duplicates) > max_items:
        console.print(f"[dim]... and {len(duplicates) - max_items} more[/dim]")


def print_duplicates(
    duplicates: list[DuplicatePair],
    title: str,
    show_source: bool = False,
    max_items: int = 20,
) -> None:
    """Print duplicate pairs in a table."""
    _print_duplicate_table(
        duplicates,
        title=title,
        show_source=show_source,
        max_items=max_items,
        hybrid=False,
    )


def print_hybrid_duplicates(
    duplicates: list[HybridDuplicate],
    show_source: bool = False,
    max_items: int = 20,
) -> None:
    """Print synthesized hybrid duplicate pairs."""
    _print_duplicate_table(
        duplicates,
        title="Hybrid Duplicates",
        show_source=show_source,
        max_items=max_items,
        hybrid=True,
    )


def print_unused(
    unused: list[CodeUnit], max_items: int = 20, title: str = "Potentially Unused"
) -> None:
    """Print potentially unused code units."""
    if not unused:
        return

    console.print(f"\n[bold yellow]{title}[/bold yellow] ({len(unused)} units)")
    console.print("[dim]These have no detected references and don't appear to be public API.[/dim]")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Type", style="dim", no_wrap=True)
    table.add_column("Location", style="dim", no_wrap=True)

    for unit in unused[:max_items]:
        table.add_row(
            unit.name,
            unit.unit_type.name.lower(),
            format_location(unit),
        )

    console.print(table)

    if len(unused) > max_items:
        console.print(f"[dim]... and {len(unused) - max_items} more[/dim]")


def print_search_results(results: list[tuple[CodeUnit, float]]) -> None:
    """Print search results in a simple rank table."""
    if not results:
        console.print("[yellow]No matches found.[/yellow]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("Rank", justify="right", no_wrap=True)
    table.add_column("Score", style="green", width=10, no_wrap=True)
    table.add_column("Name", no_wrap=True)
    table.add_column("Location", style="dim", no_wrap=True)

    for idx, (unit, score) in enumerate(results, start=1):
        table.add_row(str(idx), f"{score:.2%}", unit.name, format_location(unit))

    console.print(table)


def _add_common_analysis_options(func: Callable[..., Any]) -> Callable[..., Any]:
    options = [
        click.option(
            "--no-private",
            is_flag=True,
            help="Exclude private functions/classes",
        ),
        click.option(
            "--min-lines",
            type=int,
            default=DEFAULT_MIN_LINES,
            show_default=True,
            callback=_validate_non_negative_int,
            help="Skip semantic comparison for functions with fewer body statements",
        ),
        click.option(
            "--model",
            default=DEFAULT_MODEL,
            show_default=True,
            help="HuggingFace embedding model",
        ),
        click.option(
            "--instruction-prefix",
            default=None,
            help="Custom instruction prefix prepended to semantic inputs",
        ),
        click.option(
            "--model-revision",
            default=DEFAULT_C2LLM_REVISION,
            show_default=True,
            help="Model revision/commit (default pins C2LLM for reproducibility)",
        ),
        click.option(
            "--trust-remote-code/--no-trust-remote-code",
            default=None,
            help="Allow execution of model-provided remote code during model loading",
        ),
        click.option(
            "--batch-size",
            type=int,
            default=DEFAULT_BATCH_SIZE,
            show_default=True,
            callback=_validate_positive_int,
            help="Batch size for embeddings",
        ),
        click.option(
            "--json",
            "as_json",
            is_flag=True,
            help="Output JSON instead of rich tables",
        ),
        click.option("--verbose", "-v", is_flag=True, help="Verbose logging"),
        click.option(
            "--exclude",
            multiple=True,
            help="Glob patterns to exclude (repeat option for multiple patterns)",
        ),
        click.option("--include-stubs", is_flag=True, help="Include .pyi files"),
        click.option(
            "--output-width",
            type=int,
            default=DEFAULT_OUTPUT_WIDTH,
            show_default=True,
            callback=_validate_output_width,
            help="Width used for rich terminal rendering",
        ),
    ]

    for option in reversed(options):
        func = option(func)
    return func


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="codedupes")
def cli() -> None:
    """Detect duplicate and unused Python code using AST and semantic analysis."""


@cli.command("check", help="Run duplicate + unused analysis")
@click.argument("path", type=click.Path(path_type=Path))
@click.option(
    "-t",
    "--threshold",
    type=float,
    default=DEFAULT_THRESHOLD,
    show_default=True,
    callback=_validate_threshold,
    help="Similarity threshold for both methods",
)
@click.option(
    "--semantic-threshold",
    type=float,
    callback=_validate_threshold,
    help="Override semantic similarity threshold",
)
@click.option(
    "--traditional-threshold",
    type=float,
    callback=_validate_threshold,
    help="Override traditional (Jaccard) threshold",
)
@click.option("--semantic-only", is_flag=True, help="Only run semantic analysis")
@click.option(
    "--traditional-only",
    is_flag=True,
    help="Only run traditional (AST/token) analysis",
)
@click.option("--no-unused", is_flag=True, help="Skip unused code detection")
@click.option("--strict-unused", is_flag=True, help="Do not skip public functions")
@click.option(
    "--suppress-test-semantic",
    is_flag=True,
    help="Suppress semantic duplicate matches involving test_* functions",
)
@click.option(
    "--show-all",
    is_flag=True,
    help="Show raw traditional/semantic duplicate lists alongside hybrid output",
)
@click.option("--show-source", is_flag=True, help="Show source code snippets")
@_add_common_analysis_options
def check_command(
    path: Path,
    threshold: float,
    semantic_threshold: float | None,
    traditional_threshold: float | None,
    semantic_only: bool,
    traditional_only: bool,
    no_unused: bool,
    strict_unused: bool,
    suppress_test_semantic: bool,
    show_all: bool,
    show_source: bool,
    no_private: bool,
    min_lines: int,
    model: str,
    instruction_prefix: str | None,
    model_revision: str | None,
    trust_remote_code: bool | None,
    batch_size: int,
    as_json: bool,
    verbose: bool,
    exclude: tuple[str, ...],
    include_stubs: bool,
    output_width: int,
) -> None:
    """Run duplicate and unused-code analysis."""
    if semantic_only and traditional_only:
        raise click.UsageError("Cannot use both --semantic-only and --traditional-only.")

    _set_console(output_width)
    if not as_json:
        setup_logging(verbose)

    combined_mode = not semantic_only and not traditional_only

    semantic_thresh, traditional_thresh = _resolve_check_thresholds(
        threshold,
        semantic_threshold,
        traditional_threshold,
    )

    config = AnalyzerConfig(
        exclude_patterns=list(exclude) or None,
        include_private=not no_private,
        jaccard_threshold=traditional_thresh,
        semantic_threshold=semantic_thresh,
        model_name=model,
        instruction_prefix=instruction_prefix,
        model_revision=model_revision,
        trust_remote_code=trust_remote_code,
        run_traditional=not semantic_only,
        run_semantic=not traditional_only,
        run_unused=not no_unused,
        min_semantic_lines=min_lines,
        strict_unused=strict_unused,
        suppress_test_semantic_matches=suppress_test_semantic,
        batch_size=batch_size,
        include_stubs=include_stubs,
    )

    try:
        analyzer = CodeAnalyzer(config)
        result = analyzer.analyze(path)
    except FileNotFoundError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise click.exceptions.Exit(1) from exc
    except Exception as exc:
        console.print(f"[red]Error during analysis:[/red] {exc}")
        if verbose:
            console.print_exception()
        raise click.exceptions.Exit(1) from exc

    if as_json:
        if combined_mode:
            print_check_json_combined(result, show_all=show_all)
        else:
            print_check_json_raw(result)
    else:
        if combined_mode:
            print_summary(result, mode="combined")
            print_hybrid_duplicates(result.hybrid_duplicates, show_source=show_source)
            print_unused(result.potentially_unused, title="Likely Dead Code")

            if show_all:
                console.print(
                    f"[dim]Filtered out {result.filtered_raw_duplicates} raw duplicate pairs "
                    "from default hybrid output.[/dim]"
                )
                print_duplicates(
                    result.traditional_duplicates,
                    "Traditional Duplicates (Raw AST/Token/Jaccard)",
                    show_source=show_source,
                )
                print_duplicates(
                    result.semantic_duplicates,
                    "Semantic Duplicates (Raw Embedding)",
                    show_source=show_source,
                )
        elif semantic_only:
            print_summary(result, mode="semantic")
            print_duplicates(
                result.semantic_duplicates,
                "Semantic Duplicates (Embedding)",
                show_source=show_source,
            )
            print_unused(result.potentially_unused)
        else:
            print_summary(result, mode="traditional")
            print_duplicates(
                result.traditional_duplicates,
                "Traditional Duplicates (AST/Token/Jaccard)",
                show_source=show_source,
            )
            print_unused(result.potentially_unused)

    if combined_mode:
        has_issues = bool(result.hybrid_duplicates or result.potentially_unused)
    else:
        has_issues = bool(
            result.traditional_duplicates or result.semantic_duplicates or result.potentially_unused
        )
    raise click.exceptions.Exit(1 if has_issues else 0)


@cli.command("search", help="Search for semantically similar code")
@click.argument("path", type=click.Path(path_type=Path))
@click.argument("query")
@click.option(
    "--top-k",
    type=int,
    default=DEFAULT_TOP_K,
    show_default=True,
    callback=_validate_positive_int,
    help="Maximum results",
)
@click.option(
    "--threshold",
    type=float,
    default=DEFAULT_THRESHOLD,
    show_default=True,
    callback=_validate_threshold,
    help="Semantic similarity threshold",
)
@click.option(
    "--semantic-threshold",
    type=float,
    callback=_validate_threshold,
    help="Override semantic threshold",
)
@_add_common_analysis_options
def search_command(
    path: Path,
    query: str,
    top_k: int,
    threshold: float,
    semantic_threshold: float | None,
    no_private: bool,
    min_lines: int,
    model: str,
    instruction_prefix: str | None,
    model_revision: str | None,
    trust_remote_code: bool | None,
    batch_size: int,
    as_json: bool,
    verbose: bool,
    exclude: tuple[str, ...],
    include_stubs: bool,
    output_width: int,
) -> None:
    """Run semantic search over extracted code units."""
    _set_console(output_width)

    if not as_json:
        setup_logging(verbose)

    config = AnalyzerConfig(
        exclude_patterns=list(exclude) or None,
        include_private=not no_private,
        semantic_threshold=_resolve_threshold(threshold, semantic_threshold),
        model_name=model,
        instruction_prefix=instruction_prefix,
        model_revision=model_revision,
        trust_remote_code=trust_remote_code,
        run_traditional=False,
        run_unused=False,
        min_semantic_lines=min_lines,
        batch_size=batch_size,
        include_stubs=include_stubs,
    )

    try:
        analyzer = CodeAnalyzer(config)
        analyzer.analyze(path)
        results = analyzer.search(query, top_k=top_k)
    except Exception as exc:
        console.print(f"[red]Error during search:[/red] {exc}")
        if verbose:
            console.print_exception()
        raise click.exceptions.Exit(1) from exc

    if as_json:
        print_search_json(query, results)
    else:
        console.print(f"[bold cyan]Query:[/bold cyan] {query!r}")
        print_search_results(results)

    raise click.exceptions.Exit(0)


@cli.command("info", help="Print tool and model defaults")
def info_command() -> None:
    """Print version and default settings."""
    click.echo(f"codedupes {__version__}")
    click.echo(f"Default model: {DEFAULT_MODEL}")
    click.echo(f"Default model revision: {DEFAULT_C2LLM_REVISION}")
    click.echo(f"Default semantic threshold: {DEFAULT_THRESHOLD}")
    click.echo(f"Default traditional threshold: {DEFAULT_TRADITIONAL_THRESHOLD}")
    click.echo(f"Default min_lines for semantic: {DEFAULT_MIN_LINES}")
    click.echo(f"Default output width: {DEFAULT_OUTPUT_WIDTH}")
    click.echo("Run with --help for CLI usage")


def main() -> int:
    """Main CLI entrypoint."""
    argv = sys.argv[1:]

    try:
        result = cli.main(args=argv, prog_name="codedupes", standalone_mode=False)
        if isinstance(result, int):
            return result
    except click.exceptions.Exit as exc:
        return int(exc.exit_code)
    except click.ClickException as exc:
        exc.show()
        return exc.exit_code
    except click.Abort:
        click.echo("Aborted!", err=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
