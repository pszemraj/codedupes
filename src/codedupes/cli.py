"""Command-line interface for codedupes."""

from __future__ import annotations

import json
import logging
import platform
import sys
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal, TypeVar, cast

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from codedupes import __version__
from codedupes.analyzer import (
    DEFAULT_SEMANTIC_UNIT_TYPES,
    SEMANTIC_UNIT_TYPE_CHOICES,
    AnalyzerConfig,
    CodeAnalyzer,
)
from codedupes.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHECK_SEMANTIC_TASK,
    DEFAULT_MIN_SEMANTIC_STATEMENTS,
    DEFAULT_MODEL,
    DEFAULT_SEARCH_SEMANTIC_TASK,
    DEFAULT_SEMANTIC_DEVICE,
    DEFAULT_TOP_K,
    DEFAULT_TRADITIONAL_THRESHOLD,
    SEMANTIC_DEVICE_CHOICES,
    SEMANTIC_TASK_CHOICES,
)
from codedupes.devices import (
    configure_mps_environment,
    cpu_bf16_opted_in,
    describe_mps_fallback_env,
    format_mps_memory_snapshot,
    get_device_diagnostics,
)
from codedupes.embedding_cache import EmbeddingCache
from codedupes.extractor import DEFAULT_EXCLUDE_DIR_NAMES, DEFAULT_EXCLUDE_PATTERNS
from codedupes.languages import (
    SUPPORTED_LANGUAGES,
    GrammarUnavailableError,
    get_grammar_statuses,
)
from codedupes.logging_utils import quiet_dependency_loggers
from codedupes.models import (
    AnalysisResult,
    CodeUnit,
    DuplicatePair,
    ExtractionDiagnostic,
    HybridDuplicate,
)
from codedupes.semantic import get_semantic_runtime_versions
from codedupes.semantic_profiles import (
    get_default_search_threshold,
    get_default_semantic_threshold,
    list_supported_models,
    resolve_model_profile,
)

DEFAULT_THRESHOLD = get_default_semantic_threshold(DEFAULT_MODEL)
DEFAULT_MIN_STATEMENTS = DEFAULT_MIN_SEMANTIC_STATEMENTS
DEFAULT_OUTPUT_WIDTH = 160
MIN_OUTPUT_WIDTH = 80
DEFAULT_TABLE_ROWS = 20
DEFAULT_EXCLUDE_HELP_HINT = (
    "Replace default test-file globs with patterns to exclude (repeat for multiple patterns). "
    "Built-in common artifact-directory excludes always apply."
)

console = Console(width=DEFAULT_OUTPUT_WIDTH)
TResult = TypeVar("TResult")


class _CodedupesLogFilter(logging.Filter):
    """Filter log records so non-codedupes INFO chatter is hidden by default."""

    def __init__(self, *, include_external_info: bool) -> None:
        """Create a log filter configured for CLI verbosity.

        :param include_external_info: Allow noisy external INFO logs through.
        """
        super().__init__()
        self.include_external_info = include_external_info

    def filter(self, record: logging.LogRecord) -> bool:
        """Decide whether a log record should be emitted.

        :param record: Candidate log record.
        :return: ``False`` for noisy INFO messages from non-codedupes modules.
        """
        if record.name.startswith("codedupes"):
            return True
        if self.include_external_info:
            return True
        return record.levelno >= logging.WARNING


def _set_console(output_width: int) -> None:
    """Set global console used by all rich output helpers."""
    global console
    console = Console(width=output_width)


def _suppress_logs_for_json() -> tuple[int, list[logging.Handler]]:
    """Prevent log output from contaminating JSON responses.

    :return: Prior root logger ``(level, handlers)`` state for later restoration.
    """
    root_logger = logging.getLogger()
    prior_state = (root_logger.level, list(root_logger.handlers))
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.CRITICAL + 1)
    return prior_state


def _restore_root_logger_state(prior_state: tuple[int, list[logging.Handler]]) -> None:
    """Restore root logger level/handlers after a temporary JSON suppression."""
    prior_level, prior_handlers = prior_state
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
    for handler in prior_handlers:
        root_logger.addHandler(handler)
    root_logger.setLevel(prior_level)


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
    quiet_dependency_loggers(logging.DEBUG if verbose else logging.WARNING)


@contextmanager
def _configured_cli_output(
    *,
    as_json: bool,
    verbose: bool,
    output_width: int,
) -> Iterator[None]:
    """Configure logging/console for a CLI command and restore state on exit.

    :param as_json: Whether JSON mode is enabled.
    :param verbose: Whether verbose logging is enabled.
    :param output_width: Requested rich console width.
    :yield: ``None`` while command-specific work executes.
    :return: Iterator context that restores prior logging state on exit.
    """
    _set_console(output_width)
    logging_state: tuple[int, list[logging.Handler]] | None = None
    if as_json:
        logging_state = _suppress_logs_for_json()
    else:
        setup_logging(verbose)

    try:
        yield
    finally:
        if logging_state is not None:
            _restore_root_logger_state(logging_state)


def _run_cli_action(
    action: Callable[[], TResult],
    *,
    error_label: str,
    verbose: bool,
    catch_file_not_found: bool = False,
) -> TResult:
    """Run a command action and normalize runtime exception handling.

    :param action: Callback to execute.
    :param error_label: Human-readable action label for error messages.
    :param verbose: Whether verbose mode is enabled.
    :param catch_file_not_found: Convert ``FileNotFoundError`` to CLI exit.
    :return: Action return value.
    :raises click.exceptions.Exit: On handled runtime failures.
    """
    try:
        return action()
    except FileNotFoundError as exc:
        if not catch_file_not_found:
            raise
        console.print(f"[red]Error:[/red] {exc}")
        raise click.exceptions.Exit(1) from exc
    except GrammarUnavailableError as exc:
        console.print(f"[red]Parser unavailable:[/red] {exc}")
        console.print("Run `codedupes info` to check Tree-sitter parser package status.")
        raise click.exceptions.Exit(1) from exc
    except Exception as exc:
        console.print(f"[red]Error during {error_label}:[/red] {exc}")
        if verbose:
            console.print_exception()
        raise click.exceptions.Exit(1) from exc


def _validate_positive_int(_ctx: click.Context, _param: click.Parameter, value: int) -> int:
    """Validate a positive integer option that never reaches ``AnalyzerConfig``.

    Numeric options forwarded into ``AnalyzerConfig`` rely on its ``__post_init__``
    range checks instead of CLI callbacks; this callback exists only for options
    (``--top-k``) the library layer never sees.

    :param _ctx: Click callback context (unused).
    :param _param: Click callback parameter metadata (unused).
    :param value: Candidate value.
    :return: Value if it is strictly positive.
    :raises click.BadParameter: When value is ``<= 0``.
    """
    if value <= 0:
        raise click.BadParameter("must be > 0")
    return value


def _validate_output_width(_ctx: click.Context, _param: click.Parameter, value: int) -> int:
    """Validate output width for rich table rendering.

    :param _ctx: Click callback context (unused).
    :param _param: Click callback parameter metadata (unused).
    :param value: Desired output width.
    :return: Value if it meets the minimum width.
    :raises click.BadParameter: When value is below the minimum width.
    """
    if value < MIN_OUTPUT_WIDTH:
        raise click.BadParameter(f"must be >= {MIN_OUTPUT_WIDTH}")
    return value


def _resolve_check_thresholds(
    threshold: float | None,
    semantic_threshold: float | None,
    traditional_threshold: float | None,
    *,
    model_name: str,
) -> tuple[float, float]:
    """Resolve semantic and traditional thresholds using precedence rules.

    :param threshold: Shared threshold override.
    :param semantic_threshold: Explicit semantic threshold override.
    :param traditional_threshold: Explicit traditional threshold override.
    :param model_name: Model name used for default semantic threshold.
    :return: Tuple of ``(semantic_threshold, traditional_threshold)``.
    """
    default_semantic = get_default_semantic_threshold(model_name)
    default_traditional = DEFAULT_TRADITIONAL_THRESHOLD
    return (
        (
            semantic_threshold
            if semantic_threshold is not None
            else threshold
            if threshold is not None
            else default_semantic
        ),
        (
            traditional_threshold
            if traditional_threshold is not None
            else threshold
            if threshold is not None
            else default_traditional
        ),
    )


def _resolve_search_threshold(
    threshold: float | None,
    semantic_threshold: float | None,
) -> float | None:
    """Resolve the explicit semantic threshold override for search mode.

    :param threshold: Shared threshold override.
    :param semantic_threshold: Search-specific semantic threshold override.
    :return: Explicit override, or ``None`` to use the model profile search default.
    """
    if semantic_threshold is not None:
        return semantic_threshold
    return threshold


def _is_cli_explicit(ctx: click.Context, option_name: str) -> bool:
    """Return whether a CLI option was explicitly provided by the user.

    :param ctx: Active Click command context.
    :param option_name: Internal Click option parameter name.
    :return: ``True`` when the option source is command line input.
    """
    return ctx.get_parameter_source(option_name) == click.core.ParameterSource.COMMANDLINE


def _resolve_paired_flags(enabled: bool, disabled: bool, flag: str) -> bool | None:
    """Resolve a ``--<flag>``/``--no-<flag>`` pair and reject contradictory input.

    :param enabled: Whether ``--<flag>`` was provided.
    :param disabled: Whether ``--no-<flag>`` was provided.
    :param flag: Flag name without the leading dashes, e.g. ``mps-fallback``.
    :return: ``True``/``False`` when explicitly set, otherwise ``None``.
    :raises click.UsageError: When both contradictory flags are provided.
    """
    if enabled and disabled:
        raise click.UsageError(f"Cannot combine --{flag} and --no-{flag}.")
    if enabled:
        return True
    if disabled:
        return False
    return None


def _validate_json_output_controls(
    *,
    as_json: bool,
    verbose: bool,
    output_width_explicit: bool,
    show_source: bool = False,
    full_table: bool = False,
) -> None:
    """Reject flags that are incompatible with JSON-only output mode.

    :param as_json: Whether JSON output mode is enabled.
    :param verbose: Whether verbose logging was requested.
    :param output_width_explicit: Whether ``--output-width`` was explicitly set.
    :param show_source: Whether source snippet rendering was requested.
    :param full_table: Whether full table rendering was requested.
    :return: ``None``.
    :raises click.UsageError: When rich/logging controls are combined with JSON mode.
    """
    if not as_json:
        return

    incompatible: list[str] = []
    if verbose:
        incompatible.append("--verbose")
    if output_width_explicit:
        incompatible.append("--output-width")
    if show_source:
        incompatible.append("--show-source")
    if full_table:
        incompatible.append("--full-table")

    if incompatible:
        listed = ", ".join(incompatible)
        raise click.UsageError(f"Cannot use {listed} with --json.")


def format_location(unit: CodeUnit) -> str:
    """Format file:line location for table rendering.

    :param unit: Unit to format.
    :return: ``<filename>:<lineno>`` string.
    """
    return f"{unit.file_path.name}:{unit.lineno}"


def truncate_source(source: str, max_lines: int = 5) -> str:
    """Truncate source code for compact display.

    :param source: Source string to truncate.
    :param max_lines: Maximum lines to keep.
    :return: Truncated source with optional overflow note.
    """
    lines = source.strip().split("\n")
    if len(lines) <= max_lines:
        return source.strip()
    return "\n".join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more lines)"


def print_summary(
    result: AnalysisResult,
    *,
    mode: Literal["combined", "traditional", "semantic"],
) -> None:
    """Print analysis summary.

    :param result: Complete analysis result.
    :param mode: Output mode used for this result.
    :return: ``None``.
    """
    console.print()

    summary = Table(title="Analysis Summary", show_header=False, box=None)
    summary.add_column(style="bold cyan", no_wrap=True)
    summary.add_column(style="white", no_wrap=True)

    summary.add_row("Total code units", str(len(result.units)))
    language_counts = Counter(unit.language for unit in result.units)
    for language, count in sorted(language_counts.items()):
        summary.add_row(f"  {language}", str(count))
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

    if result.extraction_diagnostics:
        summary.add_row("Extraction diagnostics", str(len(result.extraction_diagnostics)))
    if result.unused_excluded_units:
        summary.add_row(
            "Unused-analysis exclusions",
            f"{result.unused_excluded_units} non-Python units",
        )

    console.print(summary)
    if result.extraction_diagnostics:
        console.print("[bold yellow]Extraction diagnostics[/bold yellow]")
        for diagnostic in result.extraction_diagnostics[:10]:
            location = str(diagnostic.file_path)
            if diagnostic.lineno is not None:
                location += f":{diagnostic.lineno}"
            console.print(
                f"  [yellow]{diagnostic.severity}[/yellow] "
                f"[{diagnostic.language}] {location}: {diagnostic.message}"
            )
        remaining = len(result.extraction_diagnostics) - 10
        if remaining > 0:
            console.print(f"  [dim]... and {remaining} more diagnostics[/dim]")
    console.print()


def _language_counts(units: list[CodeUnit]) -> dict[str, int]:
    """Count extracted units by canonical language.

    :param units: Extracted code units.
    :return: Unit counts keyed by language, ordered by language name.
    """
    return dict(sorted(Counter(unit.language for unit in units).items()))


def _diagnostic_to_dict(diagnostic: ExtractionDiagnostic) -> dict[str, Any]:
    """Convert an extraction diagnostic to a JSON-safe mapping.

    :param diagnostic: Diagnostic to convert.
    :return: Dictionary of JSON-serializable diagnostic fields.
    """
    return {
        "file": str(diagnostic.file_path),
        "language": diagnostic.language,
        "severity": diagnostic.severity,
        "code": diagnostic.code,
        "message": diagnostic.message,
        "line": diagnostic.lineno,
        "end_line": diagnostic.end_lineno,
    }


def _unit_to_dict(unit: CodeUnit) -> dict[str, Any]:
    """Convert a code unit to JSON-serializable summary.

    :param unit: Unit to convert.
    :return: Dictionary with public unit fields.
    """
    return {
        "name": unit.name,
        "qualified_name": unit.qualified_name,
        "type": unit.unit_type.name.lower(),
        "language": unit.language,
        "dialect": unit.dialect,
        "native_kind": unit.native_kind,
        "file": str(unit.file_path),
        "line": unit.lineno,
        "end_line": unit.end_lineno,
        "start_byte": unit.start_byte,
        "end_byte": unit.end_byte,
        "start_column": unit.start_column,
        "end_column": unit.end_column,
        "statement_count": unit.statement_count,
        "is_public": unit.is_public,
        "is_exported": unit.is_exported,
    }


def _dup_to_dict(dup: DuplicatePair) -> dict[str, Any]:
    """Convert a duplicate pair to JSON-serializable mapping.

    :param dup: Duplicate pair to serialize.
    :return: Dictionary representation of the pair.
    """
    return {
        "unit_a": _unit_to_dict(dup.unit_a),
        "unit_b": _unit_to_dict(dup.unit_b),
        "similarity": dup.similarity,
        "method": dup.method,
    }


def _hybrid_dup_to_dict(dup: HybridDuplicate) -> dict[str, Any]:
    """Convert a hybrid duplicate pair for JSON output.

    :param dup: Hybrid duplicate to serialize.
    :return: Dictionary representation of the hybrid pair.
    """
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
    """Output combined-mode check results as JSON.

    :param result: Full analysis result to serialize.
    :param show_all: Include raw duplicate lists in addition to hybrid output.
    :return: ``None``.
    """
    output: dict[str, Any] = {
        "analysis_mode": result.analysis_mode,
        "summary": {
            "total_units": len(result.units),
            "units_by_language": _language_counts(result.units),
            "hybrid_duplicates": len(result.hybrid_duplicates),
            "potentially_unused": len(result.potentially_unused),
            "raw_traditional_duplicates": len(result.traditional_duplicates),
            "raw_semantic_duplicates": len(result.semantic_duplicates),
            "filtered_raw_duplicates": result.filtered_raw_duplicates,
            "semantic_fallback": result.semantic_fallback,
            "semantic_fallback_reason": result.semantic_fallback_reason,
            "extraction_diagnostics": len(result.extraction_diagnostics),
            "unused_supported_languages": list(result.unused_supported_languages),
            "unused_excluded_units": result.unused_excluded_units,
        },
        "extraction_diagnostics": [
            _diagnostic_to_dict(diagnostic) for diagnostic in result.extraction_diagnostics
        ],
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
        "analysis_mode": result.analysis_mode,
        "summary": {
            "total_units": len(result.units),
            "units_by_language": _language_counts(result.units),
            "traditional_duplicates": len(result.traditional_duplicates),
            "semantic_duplicates": len(result.semantic_duplicates),
            "potentially_unused": len(result.potentially_unused),
            "semantic_fallback": result.semantic_fallback,
            "semantic_fallback_reason": result.semantic_fallback_reason,
            "extraction_diagnostics": len(result.extraction_diagnostics),
            "unused_supported_languages": list(result.unused_supported_languages),
            "unused_excluded_units": result.unused_excluded_units,
        },
        "extraction_diagnostics": [
            _diagnostic_to_dict(diagnostic) for diagnostic in result.extraction_diagnostics
        ],
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
    """Output search results as JSON.

    :param query: Original search query.
    :param results: Matching units and cosine scores.
    :return: ``None``.
    """
    payload = {
        "query": query,
        "results": [{"score": float(score), **_unit_to_dict(unit)} for unit, score in results],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def _build_duplicates_table(*, hybrid: bool = False) -> Table:
    """Build the duplicate table columns for terminal output.

    :param hybrid: When true, build columns for hybrid duplicate mode.
    :return: Configured rich ``Table`` instance.
    """
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


def _syntax_lexer(unit: CodeUnit) -> str:
    """Return a stable Pygments lexer alias for a code unit.

    :param unit: Unit whose source will be highlighted.
    :return: Pygments lexer alias, or ``"text"`` when the dialect is unknown.
    """
    dialect = unit.dialect or unit.language
    return {
        "python": "python",
        "c": "c",
        "rust": "rust",
        "javascript": "javascript",
        "jsx": "javascript",
        "typescript": "typescript",
        "tsx": "typescript",
    }.get(dialect, "text")


def _print_source_panels(unit_a: CodeUnit, unit_b: CodeUnit) -> None:
    """Print syntax-highlighted source snippets for two units.

    :param unit_a: First code unit.
    :param unit_b: Second code unit.
    :return: ``None``.
    """
    console.print(
        Panel(
            Syntax(truncate_source(unit_a.source), _syntax_lexer(unit_a), theme="monokai"),
            title=f"[cyan]{unit_a.qualified_name}[/cyan]",
            border_style="dim",
        )
    )
    console.print(
        Panel(
            Syntax(truncate_source(unit_b.source), _syntax_lexer(unit_b), theme="monokai"),
            title=f"[cyan]{unit_b.qualified_name}[/cyan]",
            border_style="dim",
        )
    )


def _print_duplicate_table(
    duplicates: list[DuplicatePair] | list[HybridDuplicate],
    *,
    title: str,
    show_source: bool,
    max_items: int | None,
    hybrid: bool,
) -> None:
    """Render duplicate pairs in either raw or hybrid layout.

    :param duplicates: Duplicate pairs to display.
    :param title: Section title.
    :param show_source: Whether to render source snippets.
    :param max_items: Optional row limit.
    :param hybrid: Whether the payload is hybrid duplicates.
    :return: ``None``.
    """
    if not duplicates:
        return

    console.print(f"\n[bold yellow]{title}[/bold yellow] ({len(duplicates)} pairs)")
    table = _build_duplicates_table(hybrid=hybrid)

    visible = duplicates if max_items is None else duplicates[:max_items]
    for duplicate in visible:
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

    if max_items is not None and len(duplicates) > max_items:
        console.print(f"[dim]... and {len(duplicates) - max_items} more[/dim]")


def print_duplicates(
    duplicates: list[DuplicatePair],
    title: str,
    show_source: bool = False,
    max_items: int | None = DEFAULT_TABLE_ROWS,
) -> None:
    """Print duplicate pairs in a table.

    :param duplicates: Duplicate pairs to print.
    :param title: Section title.
    :param show_source: Whether to render source snippets.
    :param max_items: Optional max rows.
    :return: ``None``.
    """
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
    max_items: int | None = DEFAULT_TABLE_ROWS,
) -> None:
    """Print synthesized hybrid duplicate pairs.

    :param duplicates: Hybrid duplicates to print.
    :param show_source: Whether to render source snippets.
    :param max_items: Optional max rows.
    :return: ``None``.
    """
    _print_duplicate_table(
        duplicates,
        title="Hybrid Duplicates",
        show_source=show_source,
        max_items=max_items,
        hybrid=True,
    )


def print_unused(
    unused: list[CodeUnit],
    max_items: int | None = DEFAULT_TABLE_ROWS,
    title: str = "Potentially Unused",
) -> None:
    """Print potentially unused code units.

    :param unused: Units with no detected references.
    :param max_items: Optional max rows.
    :param title: Section title.
    :return: ``None``.
    """
    if not unused:
        return

    console.print(f"\n[bold yellow]{title}[/bold yellow] ({len(unused)} units)")
    console.print("[dim]These have no detected references and don't appear to be public API.[/dim]")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Type", style="dim", no_wrap=True)
    table.add_column("Location", style="dim", no_wrap=True)

    visible = unused if max_items is None else unused[:max_items]
    for unit in visible:
        table.add_row(
            unit.name,
            unit.unit_type.name.lower(),
            format_location(unit),
        )

    console.print(table)

    if max_items is not None and len(unused) > max_items:
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


def _add_common_analysis_options(
    *,
    command_name: Literal["check", "search"],
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Attach shared CLI options to analysis commands.

    :param command_name: Command receiving the shared options.
    :return: Decorator that applies click options to command functions.
    """
    if command_name == "check":
        min_statements_help = (
            "Skip semantic comparison for code units with fewer body statements "
            "(also narrows traditional duplicate scope in combined mode)"
        )
        semantic_unit_help = (
            "Unit type(s) eligible for semantic embedding (repeat option to add more; "
            "also narrows traditional duplicate scope in combined mode)"
        )
    else:
        min_statements_help = "Skip semantic comparison for code units with fewer body statements"
        semantic_unit_help = (
            "Unit type(s) eligible for semantic embedding (repeat option to add more)"
        )

    options = [
        click.option(
            "--language",
            "languages",
            multiple=True,
            type=str,
            metavar="LANGUAGE",
            help=(
                "Limit extraction to a language (repeat for multiple). "
                "Aliases such as py, rs, js, jsx, ts, and tsx are accepted. "
                "Omit to auto-detect all supported languages."
            ),
        ),
        click.option(
            "--no-private",
            is_flag=True,
            help="Exclude private functions/classes",
        ),
        click.option(
            "--min-statements",
            type=int,
            default=DEFAULT_MIN_STATEMENTS,
            show_default=True,
            help=min_statements_help,
        ),
        click.option(
            "--semantic-unit-type",
            "semantic_unit_type",
            multiple=True,
            type=click.Choice(SEMANTIC_UNIT_TYPE_CHOICES),
            default=DEFAULT_SEMANTIC_UNIT_TYPES,
            show_default=True,
            help=semantic_unit_help,
        ),
        click.option(
            "--model",
            default=DEFAULT_MODEL,
            show_default=True,
            help="Embedding model alias, Hugging Face model ID, or complete local model directory",
        ),
        click.option(
            "--instruction-prefix",
            default=None,
            help="Custom instruction prefix prepended to semantic inputs",
        ),
        click.option(
            "--model-revision",
            default=None,
            show_default="auto",
            help=("Model revision/commit. If omitted, uses the model-profile default."),
        ),
        click.option(
            "--trust-remote-code",
            is_flag=True,
            help="Allow execution of model-provided remote code during model loading",
        ),
        click.option(
            "--no-trust-remote-code",
            is_flag=True,
            help="Disallow execution of model-provided remote code during model loading",
        ),
        click.option(
            "--device",
            type=click.Choice(SEMANTIC_DEVICE_CHOICES),
            default=DEFAULT_SEMANTIC_DEVICE,
            show_default=True,
            help="Semantic inference device (auto prefers CUDA, then MPS, then CPU)",
        ),
        click.option(
            "--mps-fallback",
            is_flag=True,
            help="Allow unsupported MPS operators to fall back to CPU",
        ),
        click.option(
            "--no-mps-fallback",
            is_flag=True,
            help="Disallow unsupported MPS operators from falling back to CPU",
        ),
        click.option(
            "--mps-memory-fraction",
            type=float,
            default=None,
            help=(
                "Optional PyTorch MPS allocator limit as a fraction of the recommended "
                "working set, in (0, 2]. Values above 1 increase system memory pressure."
            ),
        ),
        click.option(
            "--batch-size",
            type=int,
            default=DEFAULT_BATCH_SIZE,
            show_default=True,
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
            help=DEFAULT_EXCLUDE_HELP_HINT,
        ),
        click.option(
            "--include-stubs",
            is_flag=True,
            help="Include .pyi files when scanning a directory (single-file targets are analyzed as given)",
        ),
        click.option(
            "--no-cache",
            is_flag=True,
            help="Disable the persistent on-disk embedding cache for this run",
        ),
        click.option(
            "--strict-revision-cache",
            is_flag=True,
            help=(
                "Key an unpinned hub model's cache revision to a resolved commit hash instead of "
                "the requested revision label, disabling caching when a branch/tag can't be "
                "mapped offline (default: key by the requested label; a branch move is detected "
                "whenever a run loads the model, purging that shard so two checkpoints never "
                "mix, while fully warm runs keep serving the pre-move vectors coherently)"
            ),
        ),
        click.option(
            "--output-width",
            type=int,
            default=DEFAULT_OUTPUT_WIDTH,
            show_default=True,
            callback=_validate_output_width,
            help="Width used for rich terminal rendering",
        ),
    ]

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        """Apply shared command options to a click command function.

        :param func: Click command function to decorate.
        :return: Click command function with shared options attached.
        """
        for option in reversed(options):
            func = option(func)
        return func

    return decorator


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    no_args_is_help=False,
    invoke_without_command=True,
)
@click.version_option(__version__, prog_name="codedupes")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Detect duplicate and unused source code using structural and semantic analysis."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        ctx.exit(2)


@cli.command("check", help="Run duplicate + unused analysis")
@click.argument("path", type=click.Path(path_type=Path, exists=True))
@click.option(
    "-t",
    "--threshold",
    type=float,
    default=None,
    show_default=False,
    help="Shared threshold override for semantic and traditional checks",
)
@click.option(
    "--semantic-threshold",
    type=float,
    help="Override semantic similarity threshold",
)
@click.option(
    "--traditional-threshold",
    type=float,
    help="Override traditional (Jaccard) threshold",
)
@click.option(
    "--semantic-task",
    type=click.Choice(SEMANTIC_TASK_CHOICES),
    default=DEFAULT_CHECK_SEMANTIC_TASK,
    show_default=True,
    help="Semantic task mode for duplicate detection embeddings",
)
@click.option("--semantic-only", is_flag=True, help="Only run semantic analysis")
@click.option(
    "--traditional-only",
    is_flag=True,
    help="Only run structural/token analysis",
)
@click.option(
    "--allow-semantic-fallback",
    is_flag=True,
    help=(
        "Allow combined mode to continue with scoped traditional results when semantic "
        "backend loading/inference fails"
    ),
)
@click.option("--no-unused", is_flag=True, help="Skip unused code detection")
@click.option("--strict-unused", is_flag=True, help="Do not skip public functions")
@click.option(
    "--suppress-test-semantic",
    is_flag=True,
    help="Suppress semantic duplicate matches involving test_* functions",
)
@click.option(
    "--no-tiny-filter",
    is_flag=True,
    help="Disable tiny function/method filtering for traditional duplicates",
)
@click.option(
    "--tiny-cutoff",
    type=int,
    default=3,
    show_default=True,
    help="Tiny function/method statement cutoff (exclusive) for traditional filtering",
)
@click.option(
    "--tiny-near-jaccard-min",
    type=float,
    default=0.93,
    show_default=True,
    help="Minimum Jaccard similarity to keep tiny near-duplicate pairs",
)
@click.option(
    "--show-all",
    is_flag=True,
    help="Show raw traditional/semantic duplicate lists alongside hybrid output",
)
@click.option("--show-source", is_flag=True, help="Show source code snippets")
@click.option("--full-table", is_flag=True, help="Show all rows in terminal tables")
@_add_common_analysis_options(command_name="check")
@click.pass_context
def check_command(
    ctx: click.Context,
    path: Path,
    threshold: float | None,
    semantic_threshold: float | None,
    traditional_threshold: float | None,
    semantic_only: bool,
    traditional_only: bool,
    allow_semantic_fallback: bool,
    no_unused: bool,
    strict_unused: bool,
    suppress_test_semantic: bool,
    no_tiny_filter: bool,
    tiny_cutoff: int,
    tiny_near_jaccard_min: float,
    show_all: bool,
    show_source: bool,
    full_table: bool,
    languages: tuple[str, ...],
    no_private: bool,
    min_statements: int,
    semantic_unit_type: tuple[str, ...],
    model: str,
    semantic_task: str,
    instruction_prefix: str | None,
    model_revision: str | None,
    trust_remote_code: bool,
    no_trust_remote_code: bool,
    device: str,
    mps_fallback: bool,
    no_mps_fallback: bool,
    mps_memory_fraction: float | None,
    batch_size: int,
    as_json: bool,
    verbose: bool,
    exclude: tuple[str, ...],
    include_stubs: bool,
    no_cache: bool,
    strict_revision_cache: bool,
    output_width: int,
) -> None:
    """Run duplicate and unused-code analysis.

    :param ctx: Click command context.
    :param path: Source directory or file to analyze.
    :param threshold: Optional shared threshold override.
    :param semantic_threshold: Semantic threshold override.
    :param traditional_threshold: Traditional threshold override.
    :param semantic_only: If true, run semantic analysis only.
    :param traditional_only: If true, run traditional analysis only.
    :param allow_semantic_fallback: Continue with scoped traditional results when
        semantic backend fails in combined mode.
    :param no_unused: If true, skip unused code detection.
    :param strict_unused: If true, do not suppress likely public functions.
    :param suppress_test_semantic: Exclude matches involving test functions.
    :param no_tiny_filter: Disable tiny traditional duplicate filtering.
    :param tiny_cutoff: Tiny function/method statement cutoff for filtering.
    :param tiny_near_jaccard_min: Keep floor for tiny near-duplicate Jaccard pairs.
    :param show_all: Emit raw duplicate lists in combined mode.
    :param show_source: Show source code snippets for duplicate pairs.
    :param full_table: Show all table rows.
    :param languages: Canonical language filters; empty means auto-detect.
    :param no_private: Exclude private symbols.
    :param min_statements: Minimum code body statement lines for semantic candidate code units.
    :param semantic_unit_type: Unit type(s) eligible for semantic embedding.
    :param model: Semantic model alias/identifier.
    :param semantic_task: Semantic task used during duplicate detection.
    :param instruction_prefix: Optional custom embedding prefix.
    :param model_revision: Optional model revision/commit override.
    :param trust_remote_code: Whether remote-code execution was explicitly enabled.
    :param no_trust_remote_code: Whether remote-code execution was explicitly disabled.
    :param device: Semantic inference device.
    :param mps_fallback: Explicitly enable unsupported MPS operator fallback.
    :param no_mps_fallback: Explicitly disable unsupported MPS operator fallback.
    :param mps_memory_fraction: Optional PyTorch MPS allocator fraction.
    :param batch_size: Embedding batch size.
    :param as_json: Output JSON instead of tables.
    :param verbose: Enable debug-level logging.
    :param exclude: Glob patterns to exclude.
    :param include_stubs: Include ``.pyi`` files.
    :param no_cache: Disable the persistent on-disk embedding cache for this run.
    :param strict_revision_cache: Key an unpinned hub revision to a resolved commit hash
        instead of the requested revision label.
    :param output_width: Width used for rich output.
    :return: ``None``.
    """
    if no_unused and strict_unused:
        raise click.UsageError(
            "Cannot combine --no-unused and --strict-unused because unused reporting is disabled."
        )

    if semantic_only and traditional_only:
        raise click.UsageError("Cannot use both --semantic-only and --traditional-only.")

    if allow_semantic_fallback and (semantic_only or traditional_only):
        raise click.UsageError("--allow-semantic-fallback is only valid in default combined mode.")

    if show_all and (semantic_only or traditional_only):
        raise click.UsageError("--show-all is only valid in default combined mode.")

    _validate_json_output_controls(
        as_json=as_json,
        verbose=verbose,
        output_width_explicit=_is_cli_explicit(ctx, "output_width"),
        show_source=show_source,
        full_table=full_table,
    )

    if traditional_only:
        ignored_in_traditional_only = [
            "semantic_threshold",
            "semantic_task",
            "instruction_prefix",
            "model",
            "model_revision",
            "trust_remote_code",
            "no_trust_remote_code",
            "device",
            "mps_fallback",
            "no_mps_fallback",
            "mps_memory_fraction",
            "batch_size",
            "min_statements",
            "semantic_unit_type",
            "suppress_test_semantic",
            "strict_revision_cache",
        ]
        specified_ignored = [
            option_name
            for option_name in ignored_in_traditional_only
            if _is_cli_explicit(ctx, option_name)
        ]
        if specified_ignored:
            listed = ", ".join(f"--{name.replace('_', '-')}" for name in specified_ignored)
            raise click.UsageError(
                f"Cannot use {listed} with --traditional-only; semantic analysis is disabled."
            )

    if semantic_only:
        ignored_in_semantic_only = [
            "traditional_threshold",
            "no_tiny_filter",
            "tiny_cutoff",
            "tiny_near_jaccard_min",
        ]
        specified_ignored = [
            option_name
            for option_name in ignored_in_semantic_only
            if _is_cli_explicit(ctx, option_name)
        ]
        if specified_ignored:
            listed = ", ".join(f"--{name.replace('_', '-')}" for name in specified_ignored)
            raise click.UsageError(
                f"Cannot use {listed} with --semantic-only; traditional duplicate analysis is disabled."
            )

    resolved_trust_remote_code = _resolve_paired_flags(
        trust_remote_code, no_trust_remote_code, "trust-remote-code"
    )
    resolved_mps_fallback = _resolve_paired_flags(mps_fallback, no_mps_fallback, "mps-fallback")

    combined_mode = not semantic_only and not traditional_only
    table_max_items: int | None = None if full_table else DEFAULT_TABLE_ROWS

    semantic_thresh, traditional_thresh = _resolve_check_thresholds(
        threshold,
        semantic_threshold,
        traditional_threshold,
        model_name=model,
    )
    semantic_task_value: str | None = semantic_task
    if semantic_only:
        traditional_thresh = DEFAULT_TRADITIONAL_THRESHOLD
    if traditional_only:
        semantic_thresh = None
        semantic_task_value = None

    try:
        config = AnalyzerConfig(
            exclude_patterns=list(exclude) or None,
            include_private=not no_private,
            languages=languages or None,
            jaccard_threshold=traditional_thresh,
            semantic_threshold=semantic_thresh,
            model_name=model,
            semantic_task=semantic_task_value,
            instruction_prefix=instruction_prefix,
            model_revision=model_revision,
            trust_remote_code=resolved_trust_remote_code,
            device=device,
            mps_fallback=resolved_mps_fallback,
            mps_memory_fraction=mps_memory_fraction,
            run_traditional=not semantic_only,
            run_semantic=not traditional_only,
            allow_semantic_fallback=allow_semantic_fallback,
            run_unused=not no_unused,
            min_semantic_statements=min_statements,
            semantic_unit_types=semantic_unit_type,
            filter_tiny_traditional=not no_tiny_filter,
            tiny_unit_statement_cutoff=tiny_cutoff,
            tiny_near_jaccard_min=tiny_near_jaccard_min,
            strict_unused=strict_unused,
            suppress_test_semantic_matches=suppress_test_semantic,
            batch_size=batch_size,
            include_stubs=include_stubs,
            embedding_cache=not no_cache,
            strict_revision_cache=strict_revision_cache,
        )
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc

    with _configured_cli_output(as_json=as_json, verbose=verbose, output_width=output_width):
        result = _run_cli_action(
            lambda: CodeAnalyzer(config).analyze(path),
            error_label="analysis",
            verbose=verbose,
            catch_file_not_found=True,
        )

        if as_json:
            if combined_mode:
                print_check_json_combined(result, show_all=show_all)
            else:
                print_check_json_raw(result)
        else:
            if combined_mode:
                print_summary(result, mode="combined")
                print_hybrid_duplicates(
                    result.hybrid_duplicates,
                    show_source=show_source,
                    max_items=table_max_items,
                )
                print_unused(
                    result.potentially_unused,
                    title="Likely Dead Code",
                    max_items=table_max_items,
                )

                if show_all:
                    console.print(
                        f"[dim]Filtered out {result.filtered_raw_duplicates} raw duplicate pairs "
                        "from default hybrid output.[/dim]"
                    )
                    print_duplicates(
                        result.traditional_duplicates,
                        "Traditional Duplicates (Raw Structural/Token/Jaccard)",
                        show_source=show_source,
                        max_items=table_max_items,
                    )
                    print_duplicates(
                        result.semantic_duplicates,
                        "Semantic Duplicates (Raw Embedding)",
                        show_source=show_source,
                        max_items=table_max_items,
                    )
            elif semantic_only:
                print_summary(result, mode="semantic")
                print_duplicates(
                    result.semantic_duplicates,
                    "Semantic Duplicates (Embedding)",
                    show_source=show_source,
                    max_items=table_max_items,
                )
                print_unused(result.potentially_unused, max_items=table_max_items)
            else:
                print_summary(result, mode="traditional")
                print_duplicates(
                    result.traditional_duplicates,
                    "Traditional Duplicates (AST/Token/Jaccard)",
                    show_source=show_source,
                    max_items=table_max_items,
                )
                print_unused(result.potentially_unused, max_items=table_max_items)

    if combined_mode:
        has_issues = bool(result.hybrid_duplicates or result.potentially_unused)
    else:
        has_issues = bool(
            result.traditional_duplicates or result.semantic_duplicates or result.potentially_unused
        )
    raise click.exceptions.Exit(1 if has_issues else 0)


@cli.command("search", help="Search for semantically similar code")
@click.argument("path", type=click.Path(path_type=Path, exists=True))
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
    default=None,
    show_default=False,
    help="Shared threshold override for semantic search",
)
@click.option(
    "--semantic-threshold",
    type=float,
    help="Override semantic threshold",
)
@click.option(
    "--semantic-task",
    type=click.Choice(SEMANTIC_TASK_CHOICES),
    default=DEFAULT_SEARCH_SEMANTIC_TASK,
    show_default=True,
    help="Semantic task mode for query/document embeddings",
)
@_add_common_analysis_options(command_name="search")
@click.pass_context
def search_command(
    ctx: click.Context,
    path: Path,
    query: str,
    top_k: int,
    threshold: float | None,
    semantic_threshold: float | None,
    semantic_task: str,
    languages: tuple[str, ...],
    no_private: bool,
    min_statements: int,
    semantic_unit_type: tuple[str, ...],
    model: str,
    instruction_prefix: str | None,
    model_revision: str | None,
    trust_remote_code: bool,
    no_trust_remote_code: bool,
    device: str,
    mps_fallback: bool,
    no_mps_fallback: bool,
    mps_memory_fraction: float | None,
    batch_size: int,
    as_json: bool,
    verbose: bool,
    exclude: tuple[str, ...],
    include_stubs: bool,
    no_cache: bool,
    strict_revision_cache: bool,
    output_width: int,
) -> None:
    """Run semantic search over extracted code units.

    :param ctx: Click command context.
    :param path: Directory or file to analyze for query context.
    :param query: Natural-language search query.
    :param top_k: Maximum results to return.
    :param threshold: Shared threshold override.
    :param semantic_threshold: Semantic threshold override.
    :param semantic_task: Semantic task used for search.
    :param languages: Canonical language filters; empty means auto-detect.
    :param no_private: Exclude private symbols.
    :param min_statements: Minimum code body statement lines for semantic candidate code units.
    :param semantic_unit_type: Unit type(s) eligible for semantic embedding.
    :param model: Semantic model alias/identifier.
    :param instruction_prefix: Optional custom embedding prefix.
    :param model_revision: Optional model revision/commit override.
    :param trust_remote_code: Whether remote-code execution was explicitly enabled.
    :param no_trust_remote_code: Whether remote-code execution was explicitly disabled.
    :param device: Semantic inference device.
    :param mps_fallback: Explicitly enable unsupported MPS operator fallback.
    :param no_mps_fallback: Explicitly disable unsupported MPS operator fallback.
    :param mps_memory_fraction: Optional PyTorch MPS allocator fraction.
    :param batch_size: Embedding batch size.
    :param as_json: Output JSON result instead of table.
    :param verbose: Enable debug-level logging.
    :param exclude: Glob patterns to exclude.
    :param include_stubs: Include ``.pyi`` files.
    :param no_cache: Disable the persistent on-disk embedding cache for this run.
    :param strict_revision_cache: Key an unpinned hub revision to a resolved commit hash
        instead of the requested revision label.
    :param output_width: Width used for rich output.
    :return: ``None``.
    """
    _validate_json_output_controls(
        as_json=as_json,
        verbose=verbose,
        output_width_explicit=_is_cli_explicit(ctx, "output_width"),
    )
    resolved_trust_remote_code = _resolve_paired_flags(
        trust_remote_code, no_trust_remote_code, "trust-remote-code"
    )
    resolved_mps_fallback = _resolve_paired_flags(mps_fallback, no_mps_fallback, "mps-fallback")

    try:
        config = AnalyzerConfig(
            exclude_patterns=list(exclude) or None,
            include_private=not no_private,
            languages=languages or None,
            semantic_threshold=_resolve_search_threshold(threshold, semantic_threshold),
            model_name=model,
            semantic_task=semantic_task,
            instruction_prefix=instruction_prefix,
            model_revision=model_revision,
            trust_remote_code=resolved_trust_remote_code,
            device=device,
            mps_fallback=resolved_mps_fallback,
            mps_memory_fraction=mps_memory_fraction,
            run_traditional=False,
            run_unused=False,
            min_semantic_statements=min_statements,
            semantic_unit_types=semantic_unit_type,
            batch_size=batch_size,
            include_stubs=include_stubs,
            embedding_cache=not no_cache,
            strict_revision_cache=strict_revision_cache,
        )
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc

    with _configured_cli_output(as_json=as_json, verbose=verbose, output_width=output_width):
        analyzer = CodeAnalyzer(config)
        _run_cli_action(
            lambda: analyzer.index(path),
            error_label="search",
            verbose=verbose,
        )
        results = _run_cli_action(
            lambda: analyzer.search(query, top_k=top_k),
            error_label="search",
            verbose=verbose,
        )

        if as_json:
            print_search_json(query, results)
        else:
            console.print(f"[bold cyan]Query:[/bold cyan] {query!r}")
            print_search_results(results)

    raise click.exceptions.Exit(0)


@cli.command("info", help="Print tool and model defaults")
def info_command() -> None:
    """Print version and default settings."""
    default_profile = resolve_model_profile(DEFAULT_MODEL)
    click.echo(f"codedupes {__version__}")
    runtime_versions = get_semantic_runtime_versions()
    click.echo(f"Python: {runtime_versions['python']}")
    click.echo(f"Platform: {platform.platform()}")
    click.echo(f"PyTorch: {runtime_versions['torch']}")
    click.echo(f"Transformers: {runtime_versions['transformers']}")
    click.echo(f"Sentence Transformers: {runtime_versions['sentence-transformers']}")
    # Diagnostics import torch, which reads PYTORCH_ENABLE_MPS_FALLBACK exactly once, so the
    # environment must be settled first or a later semantic run in this process cannot enable it.
    configure_mps_environment(DEFAULT_SEMANTIC_DEVICE, fallback=None)
    diagnostics = get_device_diagnostics(DEFAULT_SEMANTIC_DEVICE)
    click.echo(f"Default semantic device request: {DEFAULT_SEMANTIC_DEVICE}")
    click.echo(f"Resolved semantic device: {diagnostics.resolved or 'unavailable'}")
    click.echo(f"CUDA available: {diagnostics.cuda_available}")
    click.echo(f"MPS built/available: {diagnostics.mps_built}/{diagnostics.mps_available}")
    click.echo(
        f"PYTORCH_ENABLE_MPS_FALLBACK: {diagnostics.mps_fallback_env} "
        f"(torch reads this as: {describe_mps_fallback_env(diagnostics.mps_fallback_env)})"
    )
    click.echo(
        f"MLX loaded in process: {diagnostics.mlx_loaded} "
        "(MLX allocator is not managed by codedupes)"
    )
    click.echo(
        f"CPU: {diagnostics.cpu_name or 'unknown'} ({diagnostics.cpu_architecture or 'unknown'})"
    )
    click.echo(
        f"CPU bfloat16 GEMM capable: {diagnostics.cpu_bf16_native} "
        f"(native bf16 ISA={diagnostics.cpu_bf16_isa}, mkldnn available={diagnostics.cpu_mkldnn_available})"
    )
    if cpu_bf16_opted_in():
        cpu_bf16_policy = (
            "enabled (experimental)"
            if diagnostics.cpu_bf16_native
            else "disabled (CODEDUPES_CPU_BF16=1 set, but the capability gate failed)"
        )
    else:
        cpu_bf16_policy = (
            "disabled (experimental; set CODEDUPES_CPU_BF16=1 on gate-capable hardware)"
        )
    click.echo(f"CPU bfloat16 inference: {cpu_bf16_policy}")
    if diagnostics.mps_memory_bytes:
        click.echo(f"MPS memory: {format_mps_memory_snapshot(diagnostics.mps_memory_bytes)}")
    if diagnostics.error is not None:
        click.echo(f"Device diagnostic error: {diagnostics.error}")
    click.echo(f"Default model: {DEFAULT_MODEL}")
    click.echo(f"Default model revision: {default_profile.default_revision or 'auto'}")
    click.echo(f"Default semantic threshold ({DEFAULT_MODEL}): {DEFAULT_THRESHOLD}")
    click.echo(f"Default traditional threshold: {DEFAULT_TRADITIONAL_THRESHOLD}")
    click.echo(f"Default semantic task for check: {DEFAULT_CHECK_SEMANTIC_TASK}")
    click.echo(f"Default semantic task for search: {DEFAULT_SEARCH_SEMANTIC_TASK}")
    click.echo(f"Default min_statements for semantic: {DEFAULT_MIN_STATEMENTS}")
    click.echo(f"Default output width: {DEFAULT_OUTPUT_WIDTH}")
    click.echo("Default combined semantic fallback: disabled")
    click.echo(f"Supported languages: {', '.join(SUPPORTED_LANGUAGES)}")
    click.echo("Unused-code analysis languages: python")
    click.echo("Tree-sitter grammar packages:")
    for status in get_grammar_statuses():
        installed = status.installed_version or "not installed"
        state = "ready" if status.available else "unavailable"
        click.echo(
            f"  - {status.dialect}: {status.package}=={status.pinned_version} "
            f"(installed={installed}, {state})"
        )
        if status.error:
            click.echo(f"      {status.error}")
    click.echo("Default built-in exclude globs:")
    for pattern in DEFAULT_EXCLUDE_PATTERNS:
        click.echo(f"  - {pattern}")
    click.echo(f"Default excluded directory names ({len(DEFAULT_EXCLUDE_DIR_NAMES)} total):")
    click.echo(f"  {', '.join(sorted(DEFAULT_EXCLUDE_DIR_NAMES))}")
    click.echo("Built-in semantic model aliases:")
    for profile in list_supported_models():
        aliases = ", ".join(profile.all_aliases())
        threshold = get_default_semantic_threshold(profile.key)
        search_threshold = get_default_search_threshold(profile.key)
        click.echo(f"  - {profile.key} -> {profile.canonical_name}")
        click.echo(
            f"      family={profile.family} semantic_threshold={threshold}"
            f" search_threshold={search_threshold}"
        )
        click.echo(f"      aliases: {aliases}")
        if profile.default_revision is not None:
            click.echo(f"      default_revision: {profile.default_revision}")
        click.echo(f"      default_trust_remote_code: {profile.default_trust_remote_code}")
    click.echo("Embedding cache:")
    try:
        _echo_cache_summary(EmbeddingCache().stats())
    except Exception as exc:  # noqa: BLE001 - info is diagnostics; report and keep printing
        click.echo(f"  unavailable: {exc}")
    click.echo("Run with --help for CLI usage")


@cli.group("cache", help="Inspect or clear the persistent embedding cache")
def cache_group() -> None:
    """Group namespace for embedding-cache management subcommands."""


def _echo_cache_summary(stats: dict[str, Any]) -> None:
    """Print the embedding-cache summary lines shared by ``info`` and ``cache info``.

    :param stats: Mapping returned by ``EmbeddingCache.stats()``.
    :return: ``None``.
    """
    click.echo(f"Cache path: {stats['path']}")
    click.echo(f"Disabled via CODEDUPES_NO_CACHE: {stats['disabled']}")
    click.echo(f"Entries: {stats['entries']}")
    click.echo(f"Size on disk: {stats['size_bytes']} bytes")


@cache_group.command("info", help="Show embedding cache location, size, and breakdown")
def cache_info_command() -> None:
    """Print embedding cache path, entry counts, size, and per-model/per-repo breakdown."""
    try:
        stats = EmbeddingCache().stats()
    except Exception as exc:
        click.echo(f"Cache unavailable: {exc}")
        raise click.exceptions.Exit(1) from exc
    _echo_cache_summary(stats)
    if stats["models"]:
        click.echo("Per-model entry counts:")
        for model_name, count in sorted(stats["models"].items()):
            click.echo(f"  - {model_name}: {count}")
    if stats["repos"]:
        click.echo("Per-repo breakdown:")
        for repo in stats["repos"]:
            click.echo(
                f"  - {repo['repo']}: {repo['shards']} shard(s), {repo['entries']} entries, "
                f"{repo['size_bytes']} bytes"
            )


@cache_group.command("clear", help="Clear cached embeddings")
@click.option(
    "--model",
    default=None,
    help="Only clear entries for this model alias or canonical HuggingFace ID",
)
def cache_clear_command(model: str | None) -> None:
    """Clear cached embeddings, optionally scoped to a single model.

    :param model: Optional model alias or canonical name filter.
    :return: ``None``.
    """
    canonical_model = resolve_model_profile(model).canonical_name if model else None
    try:
        cleared = EmbeddingCache().clear(model=canonical_model)
    except Exception as exc:
        click.echo(f"Cache clear failed: {exc}")
        raise click.exceptions.Exit(1) from exc
    if model:
        click.echo(
            f"Cleared {cleared} cached embedding(s) for model '{model}' ({canonical_model})."
        )
    else:
        click.echo(f"Cleared {cleared} cached embedding(s).")


def main() -> int:
    """CLI program entrypoint.

    :return: Process exit code from click dispatch.
    """
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
