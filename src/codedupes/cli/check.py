"""Implementation of the ``codedupes check`` command."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import rich_click as click

import codedupes.cli as cli_module
from codedupes.constants import DEFAULT_CHECK_SEMANTIC_TASK, SEMANTIC_TASK_CHOICES

from ._json import print_check_json_combined, print_check_json_raw
from ._options import CheckOptions, Panel, option_panels, semantic_options
from ._output import _configured_cli_output, _run_cli_action
from ._render import print_duplicates, print_hybrid_duplicates, print_summary, print_unused


@cli_module.cli.command("check", help="Run duplicate + unused analysis")
@click.argument("path", type=click.Path(path_type=Path, exists=True), panel=Panel.SCOPE)
@click.option(
    "-t",
    "--threshold",
    type=float,
    default=None,
    show_default=False,
    panel=Panel.DETECTION,
    show_envvar=True,
    help="Shared threshold override for semantic and traditional checks",
)
@click.option(
    "--semantic-threshold",
    type=float,
    panel=Panel.SEMANTIC,
    show_envvar=True,
    help=(
        "Flat semantic similarity gate for every language "
        "(default: the model profile's calibrated per-language gates)"
    ),
)
@click.option(
    "--traditional-threshold",
    type=float,
    panel=Panel.DETECTION,
    show_envvar=True,
    help="Override traditional (Jaccard) threshold",
)
@click.option(
    "--cross-language",
    is_flag=True,
    panel=Panel.SEMANTIC,
    show_envvar=True,
    help=(
        "Also report semantic duplicate pairs across languages "
        "(uncalibrated; a mixed pair uses the looser of its two language gates)"
    ),
)
@click.option(
    "--semantic-task",
    type=click.Choice(SEMANTIC_TASK_CHOICES),
    default=DEFAULT_CHECK_SEMANTIC_TASK,
    show_default=True,
    panel=Panel.SEMANTIC,
    show_envvar=True,
    help="Semantic task mode for duplicate detection embeddings",
)
@click.option(
    "--semantic-only",
    is_flag=True,
    panel=Panel.DETECTION,
    show_envvar=True,
    help="Only run semantic analysis",
)
@click.option(
    "--traditional-only",
    is_flag=True,
    panel=Panel.DETECTION,
    show_envvar=True,
    help="Only run structural/token analysis",
)
@click.option(
    "--allow-semantic-fallback",
    is_flag=True,
    panel=Panel.SEMANTIC,
    show_envvar=True,
    help=(
        "Allow combined mode to continue with scoped traditional results when semantic "
        "backend loading/inference fails"
    ),
)
@click.option(
    "--no-unused",
    is_flag=True,
    panel=Panel.DETECTION,
    show_envvar=True,
    help="Skip unused code detection",
)
@click.option(
    "--strict-unused",
    is_flag=True,
    panel=Panel.DETECTION,
    show_envvar=True,
    help="Do not skip public functions",
)
@click.option(
    "--suppress-test-semantic",
    is_flag=True,
    panel=Panel.SEMANTIC,
    show_envvar=True,
    help="Suppress semantic duplicate matches involving test_* functions",
)
@click.option(
    "--no-tiny-filter",
    is_flag=True,
    panel=Panel.DETECTION,
    show_envvar=True,
    help="Disable tiny function/method filtering for traditional duplicates",
)
@click.option(
    "--tiny-cutoff",
    type=int,
    default=cli_module.DEFAULT_TINY_UNIT_STATEMENT_CUTOFF,
    show_default=True,
    panel=Panel.DETECTION,
    show_envvar=True,
    help="Tiny function/method statement cutoff (exclusive) for traditional filtering",
)
@click.option(
    "--tiny-near-jaccard-min",
    type=float,
    default=cli_module.DEFAULT_TINY_NEAR_JACCARD_MIN,
    show_default=True,
    panel=Panel.DETECTION,
    show_envvar=True,
    help="Minimum Jaccard similarity to keep tiny near-duplicate pairs",
)
@click.option(
    "--show-all",
    is_flag=True,
    panel=Panel.OUTPUT,
    show_envvar=True,
    help="Show raw traditional/semantic duplicate lists alongside hybrid output",
)
@click.option(
    "--show-source",
    is_flag=True,
    panel=Panel.OUTPUT,
    show_envvar=True,
    help="Show source code snippets",
)
@click.option(
    "--full-table",
    is_flag=True,
    panel=Panel.OUTPUT,
    show_envvar=True,
    help="Show all rows in terminal tables",
)
@semantic_options("check")
@option_panels
@click.pass_context
def check_command(ctx: click.Context, path: Path, **params: Any) -> None:
    """Run duplicate and unused-code analysis."""
    opts = CheckOptions.from_params(ctx, params)
    try:
        config = opts.to_analysis_config()
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc

    with _configured_cli_output(
        as_json=opts.as_json,
        verbose=opts.verbose,
        output_width=opts.output_width,
    ):
        result = _run_cli_action(
            lambda: cli_module.CodeAnalyzer(config).analyze(path),
            error_label="analysis",
            verbose=opts.verbose,
            catch_file_not_found=True,
        )

        if opts.as_json:
            if opts.combined_mode:
                print_check_json_combined(result, show_all=opts.show_all)
            else:
                print_check_json_raw(result)
        elif opts.combined_mode:
            print_summary(result, mode="combined")
            print_hybrid_duplicates(
                result.hybrid_duplicates,
                show_source=opts.show_source,
                max_items=opts.table_max_items,
            )
            print_unused(
                result.potentially_unused,
                title="Likely Dead Code",
                max_items=opts.table_max_items,
            )
            if opts.show_all:
                print_duplicates(
                    result.traditional_duplicates,
                    "Traditional Duplicates (Raw Structural/Token/Jaccard)",
                    show_source=opts.show_source,
                    max_items=opts.table_max_items,
                )
                print_duplicates(
                    result.semantic_duplicates,
                    "Semantic Duplicates (Raw Embedding)",
                    show_source=opts.show_source,
                    max_items=opts.table_max_items,
                )
        elif opts.semantic_only:
            print_summary(result, mode="semantic")
            print_duplicates(
                result.semantic_duplicates,
                "Semantic Duplicates (Embedding)",
                show_source=opts.show_source,
                max_items=opts.table_max_items,
            )
            print_unused(result.potentially_unused, max_items=opts.table_max_items)
        else:
            print_summary(result, mode="traditional")
            print_duplicates(
                result.traditional_duplicates,
                "Traditional Duplicates (Structural/Token/Jaccard)",
                show_source=opts.show_source,
                max_items=opts.table_max_items,
            )
            print_unused(result.potentially_unused, max_items=opts.table_max_items)

    if opts.combined_mode:
        has_issues = bool(result.hybrid_duplicates or result.potentially_unused)
    else:
        has_issues = bool(
            result.traditional_duplicates or result.semantic_duplicates or result.potentially_unused
        )
    raise click.exceptions.Exit(1 if has_issues else 0)
