"""Rich terminal rendering for CLI results."""

from __future__ import annotations

import difflib
import os
import textwrap
from collections import Counter
from collections.abc import Iterable
from typing import cast

from rich import box
from rich.markup import escape
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from codedupes.models import (
    CodeUnit,
    DuplicatePair,
    ExtractionDiagnostic,
    HybridDuplicate,
)
from codedupes.report.selection import (
    ExactFamily,
    FailOnPolicy,
    FileSearchResult,
    ReportSelection,
    hidden_only_failure,
)
from codedupes.semantic import EmbeddingRunStats

from . import _output
from ._output import DEFAULT_TABLE_ROWS

_RAW_DUPLICATE_TITLES = {
    "traditional": "Near Duplicates (Jaccard)",
    "semantic": "Semantic Duplicates (Embedding)",
    "none": "Duplicates",
}


def _settings_panel(title: str, rows: Iterable[tuple[str, object]]) -> Panel:
    """Build a fitted diagnostic panel with literal labels and values.

    :param title: Section heading.
    :param rows: Label/value pairs to display.
    :return: Rich panel with naturally sized columns and wrapped values.
    """
    table = Table.grid(padding=(0, 2))
    table.add_column(style="cyan", max_width=24, overflow="fold")
    table.add_column(overflow="fold")
    for label, value in rows:
        table.add_row(Text(label), Text(str(value)))
    return Panel(
        table,
        title=Text(title, style="bold cyan"),
        title_align="left",
        border_style="dim",
        expand=False,
    )


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
    if stats.deleted_units:
        parts.append(f"{stats.deleted_units:,} units deleted")
    if stats.orphan_rows_retained:
        parts.append(f"{stats.orphan_rows_retained:,} orphan rows retained")
    if stats.orphan_rows_collected:
        parts.append(f"{stats.orphan_rows_collected:,} orphan rows collected")
    context: list[str] = []
    if stats.manifest_generation is not None:
        context.append(f"gen {stats.manifest_generation}")
    context.append(stats.execution_device if stats.model_loaded else "model not loaded")
    return f"{', '.join(parts)} ({', '.join(context)})"


def _display_path(path: os.PathLike[str] | str) -> str:
    """Return a compact display path, unescaped.

    Bare file names collide across directories, which renders a cross-directory
    duplicate pair as two identical cells. Prefer the shorter of the relative
    and absolute spellings so deeply nested working directories retain the
    filename within narrow tables.

    :param path: Path to format.
    :return: Compact path, not markup-escaped.
    """
    absolute = os.fspath(path)
    try:
        relative = os.path.relpath(path)
    except ValueError:
        # Windows: no relative path exists across drives.
        return absolute
    return min(relative, absolute, key=len)


def format_path(path: os.PathLike[str] | str) -> str:
    """Format a compact, markup-safe path for table rendering.

    :param path: Path to format.
    :return: Markup-escaped path.
    """
    return escape(_display_path(path))


def format_location(unit: CodeUnit) -> str:
    """Format a compact, markup-safe file:line location for table rendering.

    :param unit: Unit to format.
    :return: Markup-escaped ``<path>:<lineno>`` string.
    """
    return f"{format_path(unit.file_path)}:{unit.lineno}"


def _count(count: int, noun: str) -> str:
    """Pluralize a simple count/noun pair for terminal output.

    :param count: Item count.
    :param noun: Singular noun, regular plural (append ``s``).
    :return: ``"N noun"`` for one item, ``"N nouns"`` otherwise.
    """
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


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
        _output.console.print(f"  [dim]... and {_count(remaining, 'more diagnostic')}[/dim]")


def _family_noun(count: int) -> str:
    """Return ``family`` or ``families`` for a count.

    :param count: Number of families.
    :return: Singular or plural noun.
    """
    return "family" if count == 1 else "families"


def print_summary(
    selection: ReportSelection,
    *,
    fail_on: FailOnPolicy,
    exit_code: int,
    strict_unused: bool = False,
) -> None:
    """Print analysis summary.

    :param selection: Findings selected for this report, with the complete result.
    :param fail_on: Finding policy selected for this run.
    :param exit_code: Exit code computed from the selected policy.
    :param strict_unused: Whether unused findings count under the failure policy.
    :return: ``None``.
    """
    result = selection.result
    withheld = len(selection.omitted_review)
    truncated = selection.truncated_findings
    # Name the cut tiers: review pairs rank last, so under --include-review a
    # cap can drop every one of them while the withheld row stays absent.
    cut_tiers = ", ".join(
        f"{count} exact {_family_noun(count)}" if tier == "exact" else f"{count} {tier}"
        for tier, count in selection.truncated_by_tier.items()
        if count
    )
    truncation_note = (
        f"{truncated} ({cut_tiers + '; ' if cut_tiers else ''}use --max-duplicates all)"
    )
    truncated_unused = len(selection.truncated_unused)
    unused_note = f"{truncated_unused} (use --max-unused all)"
    families = len(selection.all_exact_families)
    family_note = f"{families} {_family_noun(families)} ({selection.exact_family_members} units)"
    _output.console.print()

    summary = Table(title="Analysis Summary", show_header=False, box=None)
    summary.add_column(style="bold cyan", overflow="fold")
    summary.add_column(style="white", overflow="fold")

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

    if selection.mode == "combined":
        summary.add_row("Hybrid duplicates", str(selection.total_findings))
        for tier, count in selection.duplicates_by_tier.items():
            summary.add_row(f"  {tier}", family_note if tier == "exact" and count else str(count))
        summary.add_row(
            "Actionable duplicates",
            f"{selection.actionable_findings} ({selection.reported_actionable_findings} reported)",
        )
        summary.add_row("Reported duplicates", str(selection.reported_findings))
        if withheld:
            summary.add_row("Withheld review candidates", f"{withheld} (use --include-review)")
        if truncated:
            summary.add_row("Truncated duplicates", truncation_note)
    else:
        if selection.mode == "traditional":
            summary.add_row("Traditional duplicates", str(len(result.traditional_duplicates)))
        elif selection.mode == "semantic":
            summary.add_row("Semantic duplicates", str(len(result.semantic_duplicates)))
        else:
            summary.add_row(
                "Duplicates",
                str(len(result.traditional_duplicates) + len(result.semantic_duplicates)),
            )
        if families:
            summary.add_row("Exact duplicate families", family_note)
        if truncated:
            summary.add_row("Reported duplicates", str(selection.reported_findings))
            summary.add_row("Truncated duplicates", truncation_note)

    summary.add_row("Potentially unused", str(len(result.potentially_unused)))
    if truncated_unused:
        summary.add_row("Truncated unused", unused_note)
    summary.add_row("Unused policy", "strict" if strict_unused else "default")

    if selection.mode == "combined":
        summary.add_row("", "")
        summary.add_row("Raw traditional duplicates", str(len(result.traditional_duplicates)))
        summary.add_row("Raw semantic duplicates", str(len(result.semantic_duplicates)))

    if result.extraction_diagnostics:
        summary.add_row("Extraction diagnostics", str(len(result.extraction_diagnostics)))
    if result.semantic_diagnostics:
        summary.add_row("Semantic diagnostics", str(len(result.semantic_diagnostics)))
    if result.unused_diagnostics:
        summary.add_row("Unused diagnostics", str(len(result.unused_diagnostics)))
    if result.unused_excluded_units:
        summary.add_row(
            "Unused-analysis exclusions",
            _count(result.unused_excluded_units, "non-Python unit"),
        )
    if result.embedding_stats is not None:
        summary.add_row("Embeddings", _format_embedding_stats(result.embedding_stats))
    summary.add_row("Failure policy", fail_on)
    status = f"{'fail' if exit_code else 'pass'} (exit {exit_code})"
    if exit_code and hidden_only_failure(selection, policy=fail_on, strict_unused=strict_unused):
        # Only withheld review pairs can fail without an emitted finding failing
        # too: the primary list ranks actionable tiers first, so the cap never
        # hides every failing pair.
        status = (
            f"fail (exit {exit_code}; only withheld semantic_review candidates fail "
            f"--fail-on {fail_on}, use --include-review to list them in the primary report)"
        )
    summary.add_row("Finding status", status)

    _output.console.print(summary)
    _print_diagnostics("Extraction diagnostics", result.extraction_diagnostics)
    _print_diagnostics("Semantic diagnostics", result.semantic_diagnostics)
    _print_diagnostics("Unused diagnostics", result.unused_diagnostics)
    _output.console.print()


def _build_duplicates_table(*, hybrid: bool = False, compact: bool = False) -> Table:
    """Build the duplicate table columns for terminal output.

    :param hybrid: When true, build columns for hybrid duplicate mode.
    :param compact: Whether to stack metrics and code units for a narrow terminal.
    :return: Configured rich ``Table`` instance.
    """
    table = Table(header_style="bold", box=box.ROUNDED, border_style="dim", show_lines=True)
    if compact:
        table.add_column("Evidence", width=26, min_width=18, overflow="fold")
        table.add_column("Code units", style="cyan", overflow="fold")
    elif hybrid:
        table.add_column("Score", style="green", width=10, min_width=10, no_wrap=True)
        table.add_column("Tier", style="magenta", overflow="fold")
        table.add_column("Semantic", style="green", width=8, min_width=8, no_wrap=True)
        table.add_column("Jaccard", style="green", width=7, min_width=7, no_wrap=True)
        table.add_column("Unit A", style="cyan", overflow="fold")
        table.add_column("Unit B", style="cyan", overflow="fold")
    else:
        table.add_column("Similarity", style="green", width=10, min_width=10, no_wrap=True)
        table.add_column("Unit A", style="cyan", overflow="fold")
        table.add_column("Unit B", style="cyan", overflow="fold")
        table.add_column("Method", style="dim", overflow="fold")
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


def _print_source_panels(*units: CodeUnit, source_lines: int | None) -> None:
    """Print a syntax-highlighted source snippet per unit, bounded by a line budget.

    :param units: Code units to render, in order.
    :param source_lines: Maximum lines to keep per unit, or ``None`` for no bound.
    :return: ``None``.
    """
    for unit in units:
        lines, omitted = unit.source_lines(source_lines)
        text = "\n".join(lines)
        if omitted:
            text += f"\n... ({_count(omitted, 'more line')})"
        _output.console.print(
            Panel(
                Syntax(text, _syntax_lexer(unit), theme="monokai"),
                title=f"[cyan]{escape(unit.qualified_name)}[/cyan]",
                border_style="dim",
            )
        )


def _diff_lines(unit: CodeUnit) -> list[str]:
    """Split a unit's source into diff lines, dedenting its body only.

    A method's first line already carries the signature at its own
    indentation, but ``textwrap.dedent`` on the remaining lines keeps a
    function-vs-method pair from diffing on indentation alone.

    :param unit: Unit whose source is being diffed.
    :return: Source lines, with every line after the first dedented as a block.
    """
    lines = unit.source.split("\n")
    if len(lines) <= 1:
        return lines
    return [lines[0], *textwrap.dedent("\n".join(lines[1:])).split("\n")]


def _print_diff_panel(unit_a: CodeUnit, unit_b: CodeUnit, *, source_lines: int | None) -> None:
    """Print a unified diff panel between two units' source, when they differ.

    :param unit_a: First unit; the diff's "from" side.
    :param unit_b: Second unit; the diff's "to" side.
    :param source_lines: Maximum diff lines to keep, or ``None`` for no bound.
    :return: ``None``.
    """
    diff = list(
        difflib.unified_diff(
            _diff_lines(unit_a),
            _diff_lines(unit_b),
            fromfile=f"{_display_path(unit_a.file_path)}:{unit_a.lineno} {unit_a.qualified_name}",
            tofile=f"{_display_path(unit_b.file_path)}:{unit_b.lineno} {unit_b.qualified_name}",
            n=2,
            lineterm="",
        )
    )
    if not diff:
        return
    omitted = 0
    if source_lines is not None and len(diff) > source_lines:
        omitted = len(diff) - source_lines
        diff = diff[:source_lines]
    text = "\n".join(diff)
    if omitted:
        text += f"\n... ({_count(omitted, 'more diff line')})"
    _output.console.print(
        Panel(
            Syntax(text, "diff", theme="monokai"),
            title=f"[cyan]{escape(unit_a.qualified_name)} vs {escape(unit_b.qualified_name)}[/cyan]",
            border_style="dim",
        )
    )


def _print_duplicate_table(
    duplicates: list[DuplicatePair] | list[HybridDuplicate],
    *,
    title: str,
    show_source: bool,
    source_lines: int | None = None,
    show_diff: bool = False,
    max_items: int | None,
    hybrid: bool,
    withheld: int = 0,
    truncated: int = 0,
) -> None:
    """Render duplicate pairs in either raw or hybrid layout.

    :param duplicates: Duplicate pairs to display.
    :param title: Section title.
    :param show_source: Whether to render source snippets.
    :param source_lines: Maximum source/diff lines per unit when ``show_source``/``show_diff`` is set.
    :param show_diff: Whether to render a unified diff per pair.
    :param max_items: Optional row limit for the raw diagnostic tables; the primary list is already bounded by the report cap and passes ``None``.
    :param hybrid: Whether the payload is hybrid duplicates.
    :param withheld: Review pairs the report policy withheld from this table.
    :param truncated: Pairs the ``--max-duplicates`` cap cut from this table.
    :return: ``None``.
    """
    if not duplicates:
        if withheld:
            _output.console.print(
                f"\n[dim]{title}: no reported pairs; {withheld} semantic_review "
                "candidates withheld (use --include-review to list them).[/dim]"
            )
        return

    counts = _count(len(duplicates), "pair")
    if withheld:
        counts += f", {withheld} review withheld"
    if truncated:
        counts += f", {truncated} truncated"
    _output.console.print(f"\n[bold yellow]{title}[/bold yellow] ({counts})")
    compact = _output.console.width < 120
    table = _build_duplicates_table(hybrid=hybrid, compact=compact)

    visible = duplicates if max_items is None else duplicates[:max_items]
    pending_rows = False
    for duplicate in visible:
        if hybrid:
            pair = cast(HybridDuplicate, duplicate)
            semantic = (
                f"{pair.semantic_similarity:.2%}" if pair.semantic_similarity is not None else "-"
            )
            jaccard = (
                f"{pair.jaccard_similarity:.2%}" if pair.jaccard_similarity is not None else "-"
            )
            cells = (
                f"{pair.score:.2%}",
                pair.tier,
                semantic,
                jaccard,
                f"{escape(pair.unit_a.qualified_name)}\n[dim]{format_location(pair.unit_a)}[/dim]",
                f"{escape(pair.unit_b.qualified_name)}\n[dim]{format_location(pair.unit_b)}[/dim]",
            )
            unit_a = pair.unit_a
            unit_b = pair.unit_b
        else:
            pair = cast(DuplicatePair, duplicate)
            cells = (
                f"{pair.similarity:.2%}",
                f"{escape(pair.unit_a.qualified_name)}\n[dim]{format_location(pair.unit_a)}[/dim]",
                f"{escape(pair.unit_b.qualified_name)}\n[dim]{format_location(pair.unit_b)}[/dim]",
                pair.method,
            )
            unit_a = pair.unit_a
            unit_b = pair.unit_b

        if compact:
            evidence = (
                f"Score: {cells[0]}\nTier: {cells[1]}\nSemantic: {cells[2]}\nJaccard: {cells[3]}"
                if hybrid
                else f"Similarity: {cells[0]}\nMethod: {cells[3]}"
            )
            table.add_row(
                evidence,
                f"A: {escape(unit_a.qualified_name)}\n[dim]{format_location(unit_a)}[/dim]\n"
                f"B: {escape(unit_b.qualified_name)}\n[dim]{format_location(unit_b)}[/dim]",
            )
        else:
            table.add_row(*cells)
        pending_rows = True

        if show_source or show_diff:
            _output.console.print(table)
            if show_source:
                _print_source_panels(unit_a, unit_b, source_lines=source_lines)
            if show_diff:
                _print_diff_panel(unit_a, unit_b, source_lines=source_lines)
            table = _build_duplicates_table(hybrid=hybrid, compact=compact)
            pending_rows = False

    if pending_rows:
        _output.console.print(table)

    if max_items is not None and len(duplicates) > max_items:
        _output.console.print(
            f"[dim]... and {len(duplicates) - max_items} more "
            "(use --full-table to list all rows)[/dim]"
        )


def _build_families_table(*, compact: bool) -> Table:
    """Build the exact-family table columns for terminal output.

    :param compact: Whether to stack the counts for a narrow terminal.
    :return: Configured rich ``Table`` instance.
    """
    table = Table(header_style="bold", box=box.ROUNDED, border_style="dim", show_lines=True)
    if compact:
        table.add_column("Family", width=26, min_width=18, overflow="fold")
        table.add_column("Code units", style="cyan", overflow="fold")
    else:
        table.add_column("Members", style="green", width=7, min_width=7, justify="right")
        table.add_column("Lines", style="green", width=5, min_width=5, justify="right")
        table.add_column("Method", style="magenta", width=15, min_width=15, no_wrap=True)
        table.add_column("First member", style="cyan", overflow="fold")
        table.add_column("Others", style="cyan", overflow="fold")
    return table


def print_exact_families(
    families: list[ExactFamily],
    *,
    truncated: int = 0,
    show_source: bool = False,
    source_lines: int | None = None,
    show_diff: bool = False,
) -> None:
    """Print every selected exact family; the report cap is the only bound.

    Diffs only make sense for ``structural_hash`` families (each member
    against the first); a ``token_hash`` family is token-identical, so
    ``--show-diff`` prints nothing extra for it.

    :param families: Families to print, in report order.
    :param truncated: Families the ``--max-duplicates`` cap cut from the report.
    :param show_source: Whether to render a source snippet per member.
    :param source_lines: Maximum source/diff lines per unit when ``show_source``/``show_diff`` is set.
    :param show_diff: Whether to render a unified diff per non-first member.
    :return: ``None``.
    """
    if not families:
        return

    counts = f"{len(families)} {_family_noun(len(families))}"
    if truncated:
        counts += f", {truncated} truncated"
    _output.console.print(f"\n[bold yellow]Exact Duplicate Families[/bold yellow] ({counts})")
    _output.console.print(
        "[dim]Each row is one set of mutually identical units; token_hash members are "
        "token-for-token copies, structural_hash members differ only in names or "
        "string literals.[/dim]"
    )
    compact = _output.console.width < 120
    table = _build_families_table(compact=compact)

    pending_rows = False
    for family in families:
        first, *others = family.members
        shown = others[:3]
        overflow = len(others) - len(shown)
        other_cells = [format_location(unit) for unit in shown]
        if overflow:
            other_cells.append(f"+{overflow} more")
        if compact:
            table.add_row(
                f"Members: {len(family.members)}\nLines: {family.lines}\nMethod: {family.method}",
                f"{escape(first.qualified_name)}\n[dim]{format_location(first)}[/dim]\n"
                + "\n".join(f"[dim]{cell}[/dim]" for cell in other_cells),
            )
        else:
            table.add_row(
                str(len(family.members)),
                str(family.lines),
                family.method,
                f"{escape(first.qualified_name)}\n[dim]{format_location(first)}[/dim]",
                "\n".join(other_cells),
            )
        pending_rows = True

        diff_members = others if show_diff and family.method == "structural_hash" else ()
        if show_source or diff_members:
            _output.console.print(table)
            if show_source:
                _print_source_panels(*family.members, source_lines=source_lines)
            for other in diff_members:
                _print_diff_panel(first, other, source_lines=source_lines)
            table = _build_families_table(compact=compact)
            pending_rows = False

    if pending_rows:
        _output.console.print(table)


def print_duplicates(
    duplicates: list[DuplicatePair],
    title: str,
    show_source: bool = False,
    source_lines: int | None = None,
    show_diff: bool = False,
    max_items: int | None = DEFAULT_TABLE_ROWS,
    truncated: int = 0,
) -> None:
    """Print duplicate pairs in a table.

    :param duplicates: Duplicate pairs to print.
    :param title: Section title.
    :param show_source: Whether to render source snippets.
    :param source_lines: Maximum source/diff lines per unit when ``show_source``/``show_diff`` is set.
    :param show_diff: Whether to render a unified diff per pair.
    :param max_items: Optional max rows.
    :param truncated: Pairs the ``--max-duplicates`` cap cut from the table.
    :return: ``None``.
    """
    _print_duplicate_table(
        duplicates,
        title=title,
        show_source=show_source,
        source_lines=source_lines,
        show_diff=show_diff,
        max_items=max_items,
        hybrid=False,
        truncated=truncated,
    )


def print_hybrid_duplicates(
    duplicates: list[HybridDuplicate],
    show_source: bool = False,
    source_lines: int | None = None,
    show_diff: bool = False,
    withheld: int = 0,
    truncated: int = 0,
) -> None:
    """Print every selected hybrid duplicate pair; the report cap is the only bound.

    :param duplicates: Hybrid duplicates to print.
    :param show_source: Whether to render source snippets.
    :param source_lines: Maximum source/diff lines per unit when ``show_source``/``show_diff`` is set.
    :param show_diff: Whether to render a unified diff per pair.
    :param withheld: Review pairs the report policy withheld from the table.
    :param truncated: Pairs the ``--max-duplicates`` cap cut from the table.
    :return: ``None``.
    """
    _print_duplicate_table(
        duplicates,
        title="Hybrid Duplicates",
        show_source=show_source,
        source_lines=source_lines,
        show_diff=show_diff,
        max_items=None,
        hybrid=True,
        withheld=withheld,
        truncated=truncated,
    )


def print_unused(
    unused: list[CodeUnit],
    *,
    strict: bool,
    truncated: int = 0,
) -> None:
    """Print every selected unused unit, largest first; the report cap is the only bound.

    :param unused: Units with no detected references, in report order.
    :param strict: Whether public functions and methods are also reported.
    :param truncated: Units the ``--max-unused`` cap cut from the report.
    :return: ``None``.
    """
    if not unused:
        return

    counts = _count(len(unused), "unit")
    if truncated:
        counts += f", {truncated} truncated"
    _output.console.print(f"\n[bold yellow]Potentially Unused[/bold yellow] ({counts})")
    blurb = (
        "No detected references, including public functions and methods; largest first."
        if strict
        else "No detected references; public functions and methods are excluded "
        "(use --strict-unused to include them); largest first."
    )
    _output.console.print(f"[dim]{blurb}[/dim]")

    table = Table(header_style="bold", box=box.ROUNDED, border_style="dim", show_lines=True)
    table.add_column("Name", style="cyan", overflow="fold")
    table.add_column("Type", style="dim", width=8, min_width=8, no_wrap=True)
    table.add_column("Lines", style="green", width=5, min_width=5, justify="right")
    table.add_column("Location", style="dim", overflow="fold")

    for unit in unused:
        table.add_row(
            escape(unit.qualified_name),
            unit.unit_type.name.lower(),
            str(unit.end_lineno - unit.lineno + 1),
            format_location(unit),
        )

    _output.console.print(table)


def print_findings(
    selection: ReportSelection,
    *,
    show_source: bool,
    source_lines: int | None = None,
    show_diff: bool = False,
    max_items: int | None,
    strict_unused: bool,
) -> None:
    """Print every finding panel selected for one check report.

    The primary duplicate and unused lists are already bounded by the report
    caps, so they render in full; ``max_items`` only abbreviates the raw
    ``--show-all`` lists.

    :param selection: Findings selected for this report.
    :param show_source: Whether to render source snippets.
    :param source_lines: Maximum source/diff lines per unit when ``show_source``/``show_diff`` is set.
    :param show_diff: Whether to render a unified diff per pair.
    :param max_items: Optional row limit for the raw diagnostic tables.
    :param strict_unused: Whether public functions and methods are also reported.
    :return: ``None``.
    """
    print_exact_families(
        selection.exact_families,
        truncated=len(selection.truncated_exact_families),
        show_source=show_source,
        source_lines=source_lines,
        show_diff=show_diff,
    )
    if selection.mode == "combined":
        print_hybrid_duplicates(
            cast(list[HybridDuplicate], selection.duplicates),
            show_source=show_source,
            source_lines=source_lines,
            show_diff=show_diff,
            withheld=len(selection.omitted_review),
            truncated=len(selection.truncated),
        )
        print_unused(
            selection.potentially_unused,
            strict=strict_unused,
            truncated=len(selection.truncated_unused),
        )
        if selection.traditional_duplicates is not None:
            print_duplicates(
                selection.traditional_duplicates,
                "Traditional Duplicates (Raw Structural/Token/Jaccard)",
                show_source=show_source,
                source_lines=source_lines,
                show_diff=show_diff,
                max_items=max_items,
            )
        if selection.semantic_duplicates is not None:
            print_duplicates(
                selection.semantic_duplicates,
                "Semantic Duplicates (Raw Embedding)",
                show_source=show_source,
                source_lines=source_lines,
                show_diff=show_diff,
                max_items=max_items,
            )
        return

    print_duplicates(
        cast(list[DuplicatePair], selection.duplicates),
        _RAW_DUPLICATE_TITLES[selection.mode],
        show_source=show_source,
        source_lines=source_lines,
        show_diff=show_diff,
        max_items=None,
        truncated=len(selection.truncated),
    )
    print_unused(
        selection.potentially_unused,
        strict=strict_unused,
        truncated=len(selection.truncated_unused),
    )


def _ranked_table() -> Table:
    """Build the Rank/Score scaffold every search result table starts from.

    :return: Table with the two leading columns; callers append their own.
    """
    table = Table(header_style="bold", box=box.ROUNDED, border_style="dim", show_lines=True)
    table.add_column("Rank", justify="right", width=4, min_width=4, no_wrap=True)
    table.add_column("Score", style="green", width=7, min_width=7, no_wrap=True)
    return table


def print_search_results(results: list[tuple[CodeUnit, float]]) -> None:
    """Print search results in a simple rank table."""
    if not results:
        _output.console.print("[yellow]No matches found.[/yellow]")
        return

    table = _ranked_table()
    table.add_column("Name", overflow="fold")
    table.add_column("Location", style="dim", overflow="fold")

    for idx, (unit, score) in enumerate(results, start=1):
        table.add_row(str(idx), f"{score:.2%}", escape(unit.qualified_name), format_location(unit))

    _output.console.print(table)


def print_file_search_results(results: list[FileSearchResult]) -> None:
    """Print ranked files with brief evidence from their matching code units.

    :param results: Ranked files with up to three contributing units each.
    :return: ``None``.
    """
    if not results:
        _output.console.print("[yellow]No matches found.[/yellow]")
        return

    table = _ranked_table()
    table.add_column("File", style="dim", overflow="fold")
    table.add_column("Matching code units", overflow="fold")

    for rank, result in enumerate(results, start=1):
        location = format_path(result.file_path)
        evidence = [
            f"{escape(unit.qualified_name)}:{unit.lineno} ({score:.2%})"
            for unit, score in result.matches
        ]
        remaining = result.matching_units - len(result.matches)
        if remaining:
            evidence.append(f"+{_count(remaining, 'more matching unit')}")
        table.add_row(str(rank), f"{result.score:.2%}", location, "\n".join(evidence))

    _output.console.print(table)
