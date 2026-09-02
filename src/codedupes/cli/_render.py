"""Rich terminal rendering for CLI results."""

from __future__ import annotations

import os
from collections import Counter
from typing import Literal, cast

from rich.markup import escape
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from codedupes.models import (
    AnalysisResult,
    CodeUnit,
    DuplicatePair,
    ExtractionDiagnostic,
    HybridDuplicate,
)
from codedupes.semantic import EmbeddingRunStats

from . import _output

DEFAULT_TABLE_ROWS = 20


def _format_embedding_stats(stats: EmbeddingRunStats) -> str:
    """Format one embedding run as a compact terminal summary.

    :param stats: Embedding telemetry to format.
    :return: One-line terminal summary.
    """
    parts = [
        f"{stats.cache_hit_rows:,} rows from cache",
        f"{stats.encoded_inputs:,} inputs encoded",
        f"{stats.duplicate_rows_reused:,} duplicate rows reused",
    ]
    if stats.moved_units_reused:
        parts.append(f"{stats.moved_units_reused:,} moved units remapped")
    if stats.orphan_rows_retained:
        parts.append(f"{stats.orphan_rows_retained:,} orphan rows retained")
    if stats.orphan_rows_collected:
        parts.append(f"{stats.orphan_rows_collected:,} orphan rows collected")
    context: list[str] = []
    if stats.manifest_generation is not None:
        context.append(f"gen {stats.manifest_generation}")
    context.append(stats.execution_device if stats.model_loaded else "model not loaded")
    return f"{', '.join(parts)} ({', '.join(context)})"


def format_location(unit: CodeUnit) -> str:
    """Format a compact, markup-safe file:line location for table rendering.

    Bare file names collide across directories, which renders a cross-directory
    duplicate pair as two identical cells. Prefer the shorter of the relative
    and absolute spellings so deeply nested working directories retain the
    filename within narrow tables.

    :param unit: Unit to format.
    :return: Markup-escaped ``<path>:<lineno>`` string.
    """
    absolute = str(unit.file_path)
    try:
        relative = os.path.relpath(unit.file_path)
    except ValueError:
        # Windows: no relative path exists across drives.
        location = absolute
    else:
        location = min(relative, absolute, key=len)
    return escape(f"{location}:{unit.lineno}")


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


def _print_diagnostics(title: str, diagnostics: list[ExtractionDiagnostic]) -> None:
    """Print one diagnostic section, capped at the first ten entries.

    :param title: Section heading.
    :param diagnostics: Diagnostics to render; nothing prints when empty.
    :return: ``None``.
    """
    if not diagnostics:
        return
    _output.console.print(f"[bold yellow]{title}[/bold yellow]")
    for diagnostic in diagnostics[:10]:
        location = str(diagnostic.file_path)
        if diagnostic.lineno is not None:
            location += f":{diagnostic.lineno}"
        _output.console.print(
            f"  [yellow]{escape(diagnostic.severity)}[/yellow] "
            f"{escape(f'[{diagnostic.language}]')} {escape(location)}: "
            f"{escape(diagnostic.message)}"
        )
    remaining = len(diagnostics) - 10
    if remaining > 0:
        _output.console.print(f"  [dim]... and {remaining} more diagnostics[/dim]")


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
    _output.console.print()

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
    elif mode == "traditional":
        summary.add_row("Traditional duplicates", str(len(result.traditional_duplicates)))
        summary.add_row("Potentially unused", str(len(result.potentially_unused)))
    else:
        summary.add_row("Semantic duplicates", str(len(result.semantic_duplicates)))
        summary.add_row("Potentially unused", str(len(result.potentially_unused)))

    if result.extraction_diagnostics:
        summary.add_row("Extraction diagnostics", str(len(result.extraction_diagnostics)))
    if result.semantic_diagnostics:
        summary.add_row("Semantic diagnostics", str(len(result.semantic_diagnostics)))
    if result.unused_excluded_units:
        summary.add_row(
            "Unused-analysis exclusions",
            f"{result.unused_excluded_units} non-Python units",
        )
    if result.embedding_stats is not None:
        summary.add_row("Embeddings", _format_embedding_stats(result.embedding_stats))

    _output.console.print(summary)
    _print_diagnostics("Extraction diagnostics", result.extraction_diagnostics)
    _print_diagnostics("Semantic diagnostics", result.semantic_diagnostics)
    _output.console.print()


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
    _output.console.print(
        Panel(
            Syntax(truncate_source(unit_a.source), _syntax_lexer(unit_a), theme="monokai"),
            title=f"[cyan]{unit_a.qualified_name}[/cyan]",
            border_style="dim",
        )
    )
    _output.console.print(
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

    _output.console.print(f"\n[bold yellow]{title}[/bold yellow] ({len(duplicates)} pairs)")
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
            _output.console.print(table)
            _print_source_panels(unit_a, unit_b)
            table = _build_duplicates_table(hybrid=hybrid)

    if not show_source:
        _output.console.print(table)

    if max_items is not None and len(duplicates) > max_items:
        _output.console.print(f"[dim]... and {len(duplicates) - max_items} more[/dim]")


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

    _output.console.print(f"\n[bold yellow]{title}[/bold yellow] ({len(unused)} units)")
    _output.console.print(
        "[dim]These have no detected references and don't appear to be public API.[/dim]"
    )

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

    _output.console.print(table)

    if max_items is not None and len(unused) > max_items:
        _output.console.print(f"[dim]... and {len(unused) - max_items} more[/dim]")


def print_search_results(results: list[tuple[CodeUnit, float]]) -> None:
    """Print search results in a simple rank table."""
    if not results:
        _output.console.print("[yellow]No matches found.[/yellow]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("Rank", justify="right", no_wrap=True)
    table.add_column("Score", style="green", width=10, no_wrap=True)
    table.add_column("Name", no_wrap=True)
    table.add_column("Location", style="dim", no_wrap=True)

    for idx, (unit, score) in enumerate(results, start=1):
        table.add_row(str(idx), f"{score:.2%}", unit.name, format_location(unit))

    _output.console.print(table)
