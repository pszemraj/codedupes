"""Command-line interface for codedupes."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from codedupes import __version__
from codedupes.analyzer import DEFAULT_MODEL, AnalyzerConfig, CodeAnalyzer
from codedupes.models import CodeUnit, DuplicatePair

console = Console()


def setup_logging(verbose: bool = False) -> None:
    """Configure logging with rich handler."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_time=False, show_path=False)],
    )


def _validate_threshold(value: float, label: str) -> bool:
    if not 0.0 <= value <= 1.0:
        console.print(f"[red]Error:[/red] {label} must be in [0.0, 1.0], got {value}")
        return False
    return True


def format_location(unit: CodeUnit) -> str:
    """Format file:line location."""
    return f"{unit.file_path.name}:{unit.lineno}"


def truncate_source(source: str, max_lines: int = 5) -> str:
    """Truncate source code for display."""
    lines = source.strip().split("\n")
    if len(lines) <= max_lines:
        return source.strip()
    return "\n".join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more lines)"


def print_summary(result: dict) -> None:
    """Print analysis summary."""
    console.print()

    summary = Table(title="Analysis Summary", show_header=False, box=None)
    summary.add_column(style="bold cyan")
    summary.add_column(style="white")

    summary.add_row("Total code units", str(len(result["units"])))
    summary.add_row("  Functions", str(sum(1 for u in result["units"] if u["type"] == "function")))
    summary.add_row("  Methods", str(sum(1 for u in result["units"] if u["type"] == "method")))
    summary.add_row("  Classes", str(sum(1 for u in result["units"] if u["type"] == "class")))
    summary.add_row("", "")
    summary.add_row("Exact duplicates", str(len(result["exact_duplicates"])))
    summary.add_row("Semantic duplicates", str(len(result["semantic_duplicates"])))
    summary.add_row("Potentially unused", str(len(result["potentially_unused"])))

    console.print(summary)
    console.print()


def _unit_to_dict(unit: CodeUnit) -> dict:
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


def _dup_to_dict(dup: DuplicatePair) -> dict:
    return {
        "unit_a": _unit_to_dict(dup.unit_a),
        "unit_b": _unit_to_dict(dup.unit_b),
        "similarity": dup.similarity,
        "method": dup.method,
    }


def print_check_json(result) -> None:
    """Output results as JSON."""
    output = {
        "summary": {
            "total_units": len(result.units),
            "exact_duplicates": len(result.exact_duplicates),
            "semantic_duplicates": len(result.semantic_duplicates),
            "potentially_unused": len(result.potentially_unused),
        },
        "exact_duplicates": [_dup_to_dict(d) for d in result.exact_duplicates],
        "semantic_duplicates": [_dup_to_dict(d) for d in result.semantic_duplicates],
        "potentially_unused": [_unit_to_dict(u) for u in result.potentially_unused],
    }
    print(json.dumps(output, indent=2, sort_keys=True))


def print_search_json(query: str, results: list[tuple[CodeUnit, float]]) -> None:
    """Output search output as JSON."""
    payload = {
        "query": query,
        "results": [{"score": float(score), **_unit_to_dict(unit)} for unit, score in results],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


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


def print_search_results(results: list[tuple[CodeUnit, float]]) -> None:
    """Print search results in a simple rank table."""
    if not results:
        console.print("[yellow]No matches found.[/yellow]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("Rank", justify="right")
    table.add_column("Score", style="green", width=10)
    table.add_column("Name")
    table.add_column("Location", style="dim")

    for idx, (unit, score) in enumerate(results, start=1):
        table.add_row(str(idx), f"{score:.2%}", unit.name, format_location(unit))

    console.print(table)


def _add_common_analysis_args(parser: argparse.ArgumentParser) -> None:
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
    parser.add_argument("--semantic-only", action="store_true", help="Only run semantic analysis")
    parser.add_argument(
        "--traditional-only",
        action="store_true",
        help="Only run traditional (AST/token) analysis",
    )
    parser.add_argument("--no-unused", action="store_true", help="Skip unused code detection")
    parser.add_argument("--strict-unused", action="store_true", help="Do not skip public functions")
    parser.add_argument(
        "--min-lines",
        type=int,
        default=3,
        help="Skip semantic comparison for functions with fewer body statements",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"HuggingFace embedding model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--no-private", action="store_true", help="Exclude private functions/classes"
    )
    parser.add_argument("--json", action="store_true", help="Output JSON instead of rich tables")
    parser.add_argument("--show-source", action="store_true", help="Show source code snippets")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    parser.add_argument("--exclude", nargs="+", help="Glob patterns to exclude")
    parser.add_argument("--include-stubs", action="store_true", help="Include .pyi files")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for embeddings")


def _run_check(args: argparse.Namespace) -> int:
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

    if args.batch_size <= 0:
        console.print("[red]Error:[/red] --batch-size must be > 0")
        return 1
    if args.min_lines < 0:
        console.print("[red]Error:[/red] --min-lines must be >= 0")
        return 1

    if not args.json:
        setup_logging(args.verbose)

    semantic_thresh = (
        args.semantic_threshold if args.semantic_threshold is not None else args.threshold
    )
    trad_thresh = (
        args.traditional_threshold if args.traditional_threshold is not None else args.threshold
    )

    config = AnalyzerConfig(
        exclude_patterns=args.exclude,
        include_private=not args.no_private,
        jaccard_threshold=trad_thresh,
        semantic_threshold=semantic_thresh,
        model_name=args.model,
        run_traditional=not args.semantic_only,
        run_semantic=not args.traditional_only,
        run_unused=not args.no_unused,
        min_semantic_lines=args.min_lines,
        strict_unused=args.strict_unused,
        batch_size=args.batch_size,
        include_stubs=args.include_stubs,
    )

    try:
        analyzer = CodeAnalyzer(config)
        result = analyzer.analyze(args.path)
    except FileNotFoundError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        return 1
    except Exception as exc:
        console.print(f"[red]Error during analysis:[/red] {exc}")
        if args.verbose:
            console.print_exception()
        return 1

    if args.json:
        print_check_json(result)
    else:
        result_dict = {
            "units": [_unit_to_dict(u) for u in result.units],
            "exact_duplicates": [_dup_to_dict(d) for d in result.exact_duplicates],
            "semantic_duplicates": [_dup_to_dict(d) for d in result.semantic_duplicates],
            "potentially_unused": [_unit_to_dict(u) for u in result.potentially_unused],
        }
        print_summary(result_dict)
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
        print_unused(result.potentially_unused)

    has_issues = bool(
        result.exact_duplicates or result.semantic_duplicates or result.potentially_unused
    )
    return 1 if has_issues else 0


def _run_search(args: argparse.Namespace) -> int:
    if args.top_k <= 0:
        console.print("[red]Error:[/red] --top-k must be > 0")
        return 1

    if not _validate_threshold(args.threshold, "--threshold"):
        return 1
    if args.semantic_threshold is not None and not _validate_threshold(
        args.semantic_threshold, "--semantic-threshold"
    ):
        return 1

    if args.semantic_threshold is not None:
        args.threshold = args.semantic_threshold

    if args.batch_size <= 0:
        console.print("[red]Error:[/red] --batch-size must be > 0")
        return 1

    if not args.json:
        setup_logging(args.verbose)

    config = AnalyzerConfig(
        exclude_patterns=args.exclude,
        include_private=not args.no_private,
        semantic_threshold=args.threshold,
        model_name=args.model,
        run_traditional=False,
        run_unused=False,
        min_semantic_lines=args.min_lines,
        batch_size=args.batch_size,
        include_stubs=args.include_stubs,
    )

    try:
        analyzer = CodeAnalyzer(config)
        analyzer.analyze(args.path)
        results = analyzer.search(args.query, top_k=args.top_k)
    except Exception as exc:
        console.print(f"[red]Error during search:[/red] {exc}")
        if args.verbose:
            console.print_exception()
        return 1

    if args.json:
        print_search_json(args.query, results)
    else:
        console.print(f"[bold cyan]Query:[/bold cyan] {args.query!r}")
        print_search_results(results)
    return 0


def _run_info(_: argparse.Namespace) -> int:
    print(f"codedupes {__version__}")
    print(f"Default model: {DEFAULT_MODEL}")
    print("Default semantic threshold: 0.82")
    print("Default traditional threshold: 0.85")
    print("Default min_lines for semantic: 3")
    print("Run with --help for CLI usage")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect duplicate and unused Python code using AST and semantic analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  codedupes check ./src --json
  codedupes search ./src \"sum numbers\" --top-k 5
  codedupes check ./src --semantic-only --threshold 0.8
        """,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command")

    check_parser = subparsers.add_parser("check", help="Run duplicate + unused analysis")
    check_parser.add_argument("path", type=Path, help="Directory or file to analyze")
    _add_common_analysis_args(check_parser)
    check_parser.set_defaults(func=_run_check)

    search_parser = subparsers.add_parser("search", help="Search for semantically similar code")
    search_parser.add_argument("path", type=Path, help="Directory or file to analyze")
    search_parser.add_argument("query", help="Query text")
    search_parser.add_argument("--top-k", type=int, default=10, help="Maximum results")
    search_parser.add_argument("--model", default=DEFAULT_MODEL, help="Embedding model")
    search_parser.add_argument("--threshold", type=float, default=0.82)
    search_parser.add_argument(
        "--semantic-threshold",
        type=float,
        help="Override semantic threshold",
    )
    search_parser.add_argument(
        "--no-private", action="store_true", help="Exclude private functions/classes"
    )
    search_parser.add_argument("--json", action="store_true", help="Output JSON")
    search_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    search_parser.add_argument("--exclude", nargs="+", help="Glob patterns to exclude")
    search_parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for embeddings"
    )
    search_parser.add_argument(
        "--min-lines", type=int, default=3, help="Minimum body statements for semantic candidates"
    )
    search_parser.add_argument("--include-stubs", action="store_true", help="Include .pyi files")
    search_parser.set_defaults(func=_run_search)

    info_parser = subparsers.add_parser("info", help="Print tool and model defaults")
    info_parser.set_defaults(func=_run_info)

    return parser


def main() -> int:
    """Main CLI entrypoint."""
    parser = _build_parser()

    argv = sys.argv[1:]

    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        if isinstance(exc.code, int):
            return exc.code
        return 1

    if not hasattr(args, "func"):
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
