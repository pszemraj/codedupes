"""Implementation of the ``codedupes check`` command."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import rich_click as click

import codedupes.cli as cli_module
from codedupes.constants import DEFAULT_CHECK_SEMANTIC_TASK, SEMANTIC_TASK_CHOICES
from codedupes.models import AnalysisResult, HybridTier

from ._json import print_check_json_combined, print_check_json_raw
from ._options import CheckOptions, Panel, option_panels, semantic_options
from ._output import _configured_cli_output, _run_cli_action
from ._render import print_duplicates, print_hybrid_duplicates, print_summary, print_unused

FailOnPolicy = Literal["actionable", "all", "none"]
ACTIONABLE_TIERS: frozenset[HybridTier] = frozenset(
    {"exact", "traditional_near", "hybrid_confirmed"}
)


def run_should_fail(
    result: AnalysisResult,
    *,
    policy: FailOnPolicy,
    combined_mode: bool,
    strict_unused: bool,
) -> bool:
    """Return whether reported findings should make ``check`` exit one.

    :param result: Completed analysis result.
    :param policy: Selected finding policy.
    :param combined_mode: Whether hybrid output is active.
    :param strict_unused: Whether unused findings are strict rather than heuristic.
    :return: Whether findings require exit code one.
    """
    if policy == "none":
        return False
    if combined_mode:
        duplicates = result.hybrid_duplicates
        if policy == "actionable":
            duplicates = [
                duplicate for duplicate in duplicates if duplicate.tier in ACTIONABLE_TIERS
            ]
    else:
        duplicates = result.traditional_duplicates + result.semantic_duplicates
    unused = result.potentially_unused
    if policy == "actionable" and not strict_unused:
        unused = []
    return bool(duplicates or unused)


@cli_module.cli.command(
    "check",
    help="Run duplicate + unused analysis",
    context_settings={"auto_envvar_prefix": "CODEDUPES"},
)
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
        "Allow combined mode to continue with full-scope traditional results when semantic "
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
    help="Disable tiny code-unit filtering for traditional duplicates",
)
@click.option(
    "--tiny-cutoff",
    type=int,
    default=cli_module.DEFAULT_TINY_UNIT_STATEMENT_CUTOFF,
    show_default=True,
    panel=Panel.DETECTION,
    show_envvar=True,
    help="Tiny code-unit statement cutoff (exclusive) for traditional filtering",
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
@click.option(
    "--fail-on",
    type=click.Choice(["actionable", "all", "none"]),
    default="actionable",
    show_default=True,
    panel=Panel.OUTPUT,
    show_envvar=True,
    help="Which findings make the exit code 1",
)
@semantic_options()
@option_panels
@click.pass_context
def check_command(ctx: click.Context, path: Path, **params: Any) -> None:
    """Run duplicate and unused-code analysis.

    :param ctx: Active Click context.
    :param path: File or directory to analyze.
    :param params: Parsed command options.
    :return: ``None``.
    """
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
        exit_code = int(
            run_should_fail(
                result,
                policy=opts.fail_on,
                combined_mode=opts.combined_mode,
                strict_unused=opts.strict_unused,
            )
        )

        if opts.as_json:
            if opts.combined_mode:
                print_check_json_combined(
                    result,
                    show_all=opts.show_all,
                    fail_on=opts.fail_on,
                    exit_code=exit_code,
                )
            else:
                print_check_json_raw(result, fail_on=opts.fail_on, exit_code=exit_code)
        elif opts.combined_mode:
            print_summary(
                result,
                mode="combined",
                fail_on=opts.fail_on,
                exit_code=exit_code,
            )
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
            print_summary(
                result,
                mode="semantic",
                fail_on=opts.fail_on,
                exit_code=exit_code,
            )
            print_duplicates(
                result.semantic_duplicates,
                "Semantic Duplicates (Embedding)",
                show_source=opts.show_source,
                max_items=opts.table_max_items,
            )
            print_unused(result.potentially_unused, max_items=opts.table_max_items)
        else:
            print_summary(
                result,
                mode="traditional",
                fail_on=opts.fail_on,
                exit_code=exit_code,
            )
            print_duplicates(
                result.traditional_duplicates,
                "Traditional Duplicates (Structural/Token/Jaccard)",
                show_source=opts.show_source,
                max_items=opts.table_max_items,
            )
            print_unused(result.potentially_unused, max_items=opts.table_max_items)

    raise click.exceptions.Exit(exit_code)
