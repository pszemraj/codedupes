"""Command-line interface for codedupes."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax

from .analyzer import DEFAULT_MODEL, AnalyzerConfig, CodeAnalyzer
from .models import AnalysisResult, CodeUnit, CodeUnitType, DuplicatePair

console = Console()


def setup_logging(verbose: bool = False) -> None:
    """Configure logging with rich handler."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_time=False, show_path=False)],
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


def print_summary(result: AnalysisResult) -> None:
    """Print analysis summary."""
    console.print()

    summary = Table(title="Analysis Summary", show_header=False, box=None)
    summary.add_column(style="bold cyan")
    summary.add_column(style="white")

    summary.add_row("Total code units", str(len(result.units)))
    summary.add_row(
        "  Functions",
        str(sum(1 for u in result.units if u.unit_type == CodeUnitType.FUNCTION)),
    )
    summary.add_row(
        "  Methods",
        str(sum(1 for u in result.units if u.unit_type == CodeUnitType.METHOD)),
    )
    summary.add_row(
        "  Classes",
        str(sum(1 for u in result.units if u.unit_type == CodeUnitType.CLASS)),
    )
    summary.add_row("", "")
    summary.add_row("Exact duplicates", str(len(result.exact_duplicates)))
    summary.add_row("Semantic duplicates", str(len(result.semantic_duplicates)))
    summary.add_row("Potentially unused", str(len(result.potentially_unused)))

    console.print(summary)
    console.print()


def print_duplicates(
    duplicates: list[DuplicatePair],
    title: str,
    show_source: bool = False,
    max_items: int = 20,
) -> None:
    """Print duplicate pairs in a table."""
    if not duplicates:
        return

    console.print(f"\n[bold yellow]{title}[/bold yellow] ({len(duplicates)} pairs)")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Similarity", style="green", width=10)
    table.add_column("Unit A", style="cyan")
    table.add_column("Unit B", style="cyan")
    table.add_column("Method", style="dim")

    for dup in duplicates[:max_items]:
        table.add_row(
            f"{dup.similarity:.2%}",
            f"{dup.unit_a.name}\n[dim]{format_location(dup.unit_a)}[/dim]",
            f"{dup.unit_b.name}\n[dim]{format_location(dup.unit_b)}[/dim]",
            dup.method,
        )

        if show_source:
            console.print(table)
            console.print(
                Panel(
                    Syntax(truncate_source(dup.unit_a.source), "python", theme="monokai"),
                    title=f"[cyan]{dup.unit_a.qualified_name}[/cyan]",
                    border_style="dim",
                )
            )
            console.print(
                Panel(
                    Syntax(truncate_source(dup.unit_b.source), "python", theme="monokai"),
                    title=f"[cyan]{dup.unit_b.qualified_name}[/cyan]",
                    border_style="dim",
                )
            )
            table = Table(show_header=True, header_style="bold")
            table.add_column("Similarity", style="green", width=10)
            table.add_column("Unit A", style="cyan")
            table.add_column("Unit B", style="cyan")
            table.add_column("Method", style="dim")

    if not show_source:
        console.print(table)

    if len(duplicates) > max_items:
        console.print(f"[dim]... and {len(duplicates) - max_items} more[/dim]")


def print_unused(unused: list[CodeUnit], max_items: int = 20) -> None:
    """Print potentially unused code units."""
    if not unused:
        return

    console.print(f"\n[bold yellow]Potentially Unused[/bold yellow] ({len(unused)} units)")
    console.print("[dim]These have no detected references and don't appear to be public API.[/dim]")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="dim")
    table.add_column("Location", style="dim")

    for unit in unused[:max_items]:
        table.add_row(
            unit.name,
            unit.unit_type.name.lower(),
            format_location(unit),
        )

    console.print(table)

    if len(unused) > max_items:
        console.print(f"[dim]... and {len(unused) - max_items} more[/dim]")


def print_json(result: AnalysisResult) -> None:
    """Output results as JSON."""
    import json

    def unit_to_dict(u: CodeUnit) -> dict:
        return {
            "name": u.name,
            "qualified_name": u.qualified_name,
            "type": u.unit_type.name.lower(),
            "file": str(u.file_path),
            "line": u.lineno,
            "end_line": u.end_lineno,
            "is_public": u.is_public,
            "is_exported": u.is_exported,
        }

    def dup_to_dict(d: DuplicatePair) -> dict:
        return {
            "unit_a": unit_to_dict(d.unit_a),
            "unit_b": unit_to_dict(d.unit_b),
            "similarity": d.similarity,
            "method": d.method,
        }

    output = {
        "summary": {
            "total_units": len(result.units),
            "exact_duplicates": len(result.exact_duplicates),
            "semantic_duplicates": len(result.semantic_duplicates),
            "potentially_unused": len(result.potentially_unused),
        },
        "exact_duplicates": [dup_to_dict(d) for d in result.exact_duplicates],
        "semantic_duplicates": [dup_to_dict(d) for d in result.semantic_duplicates],
        "potentially_unused": [unit_to_dict(u) for u in result.potentially_unused],
    }

    print(json.dumps(output, indent=2, sort_keys=True))


def _validate_threshold(value: float, label: str) -> bool:
    if not 0.0 <= value <= 1.0:
        console.print(f"[red]Error:[/red] {label} must be in [0.0, 1.0], got {value}")
        return False
    return True


def main() -> int:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Detect duplicate and unused Python code using AST analysis and semantic embeddings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  codedupes ./src                    # Analyze ./src directory
  codedupes ./src --semantic-only    # Skip traditional methods
  codedupes ./src --threshold 0.9    # Higher similarity threshold
  codedupes ./src --json             # Output JSON for tooling
        """,
    )

    parser.add_argument("path", type=Path, help="Directory or file to analyze")

    # Thresholds
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.82,
        help="Similarity threshold for both methods (default: 0.82)",
    )
    parser.add_argument(
        "--semantic-threshold",
        type=float,
        help="Override semantic similarity threshold",
    )
    parser.add_argument(
        "--traditional-threshold",
        type=float,
        help="Override traditional (Jaccard) threshold",
    )

    # What to run
    parser.add_argument(
        "--semantic-only",
        action="store_true",
        help="Only run semantic (embedding) analysis",
    )
    parser.add_argument(
        "--traditional-only",
        action="store_true",
        help="Only run traditional (AST/token) analysis",
    )
    parser.add_argument(
        "--no-unused",
        action="store_true",
        help="Skip unused code detection",
    )

    # Model
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"HuggingFace embedding model (default: {DEFAULT_MODEL})",
    )

    # Output
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON instead of rich tables",
    )
    parser.add_argument(
        "--show-source",
        action="store_true",
        help="Show source code snippets for duplicates",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )

    # Filtering
    parser.add_argument(
        "--exclude",
        nargs="+",
        help="Glob patterns to exclude (e.g., '**/test_*')",
    )
    parser.add_argument(
        "--no-private",
        action="store_true",
        help="Exclude private (_prefixed) functions",
    )

    args = parser.parse_args()

    if not args.json:
        setup_logging(args.verbose)

    if not _validate_threshold(args.threshold, "--threshold"):
        return 1
    if args.semantic_threshold is not None and not _validate_threshold(
        args.semantic_threshold, "--semantic-threshold"
    ):
        return 1
    if args.traditional_threshold is not None and not _validate_threshold(
        args.traditional_threshold, "--traditional-threshold"
    ):
        return 1

    # Build config
    semantic_thresh = args.semantic_threshold or args.threshold
    trad_thresh = args.traditional_threshold or args.threshold

    config = AnalyzerConfig(
        exclude_patterns=args.exclude,
        include_private=not args.no_private,
        jaccard_threshold=trad_thresh,
        semantic_threshold=semantic_thresh,
        model_name=args.model,
        run_traditional=not args.semantic_only,
        run_semantic=not args.traditional_only,
        run_unused=not args.no_unused,
    )

    # Run analysis
    try:
        analyzer = CodeAnalyzer(config)
        result = analyzer.analyze(args.path)
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        return 1
    except Exception as e:
        console.print(f"[red]Error during analysis:[/red] {e}")
        if args.verbose:
            console.print_exception()
        return 1

    # Output
    if args.json:
        print_json(result)
    else:
        print_summary(result)
        print_duplicates(
            result.exact_duplicates,
            "Exact Duplicates (AST/Token)",
            show_source=args.show_source,
        )
        print_duplicates(
            result.semantic_duplicates,
            "Semantic Duplicates (Embedding)",
            show_source=args.show_source,
        )
        if not args.no_unused:
            print_unused(result.potentially_unused)

    # Exit code: 0 if clean, 1 if issues found
    has_issues = result.exact_duplicates or result.semantic_duplicates or result.potentially_unused
    return 1 if has_issues else 0


if __name__ == "__main__":
    sys.exit(main())
