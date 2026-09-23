"""Implementation of the ``codedupes check`` command."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import rich_click as click

import codedupes.cli as cli_module
from codedupes.constants import DEFAULT_CHECK_SEMANTIC_TASK, SEMANTIC_TASK_CHOICES
from codedupes.report.json import check_result_to_json, to_json_text
from codedupes.report.selection import (
    DEFAULT_MAX_DUPLICATES,
    DEFAULT_MAX_UNUSED,
    focus_result,
    run_should_fail,
    select_findings,
)

from . import _output
from ._options import (
    REPORT_CAP,
    CheckOptions,
    Panel,
    option_panels,
    resolve_focus_paths,
    semantic_options,
)
from ._output import DEFAULT_SOURCE_LINES, _configured_cli_output, _run_cli_action
from ._render import print_findings, print_run, print_summary


@cli_module.cli.command(
    "check",
    help="Run duplicate + unused analysis",
)
@click.argument("path", type=click.Path(path_type=Path, exists=True), panel=Panel.SCOPE)
@click.option(
    "--focus",
    multiple=True,
    type=click.Path(path_type=Path, exists=True),
    panel=Panel.SCOPE,
    help=(
        "Scope the report and exit code to findings touching this file or directory "
        "(repeat for multiple); an exact-duplicate family is kept whole when any member "
        "is in focus, a duplicate pair when either endpoint is, and unused units by file. "
        "Requires a directory target"
    ),
)
@click.option(
    "-t",
    "--threshold",
    type=float,
    default=None,
    show_default=False,
    panel=Panel.DETECTION,
    help="Shared threshold override for semantic and traditional checks",
)
@click.option(
    "--semantic-threshold",
    type=float,
    panel=Panel.SEMANTIC,
    help=(
        "Flat semantic similarity gate for every language "
        "(default: the model profile's calibrated per-language gates)"
    ),
)
@click.option(
    "--traditional-threshold",
    type=float,
    panel=Panel.DETECTION,
    help="Override traditional (Jaccard) threshold",
)
@click.option(
    "--cross-language",
    is_flag=True,
    panel=Panel.SEMANTIC,
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
    help="Semantic task mode for duplicate detection embeddings",
)
@click.option(
    "--semantic-only",
    is_flag=True,
    panel=Panel.DETECTION,
    help="Only run semantic analysis",
)
@click.option(
    "--traditional-only",
    is_flag=True,
    panel=Panel.DETECTION,
    help="Only run structural/token analysis",
)
@click.option(
    "--unused-only",
    is_flag=True,
    panel=Panel.DETECTION,
    help="Only run unused-code detection; no model is loaded",
)
@click.option(
    "--allow-semantic-fallback",
    is_flag=True,
    panel=Panel.SEMANTIC,
    help=(
        "Allow combined mode to continue with full-scope traditional results when semantic "
        "backend loading/inference fails"
    ),
)
@click.option(
    "--no-unused",
    is_flag=True,
    panel=Panel.DETECTION,
    help="Skip unused code detection",
)
@click.option(
    "--strict-unused",
    is_flag=True,
    panel=Panel.DETECTION,
    help="Report unreferenced public functions and public methods too",
)
@click.option(
    "--suppress-test-semantic",
    is_flag=True,
    panel=Panel.SEMANTIC,
    help="Suppress semantic duplicate matches involving test_* functions",
)
@click.option(
    "--no-tiny-filter",
    is_flag=True,
    panel=Panel.DETECTION,
    help="Disable tiny code-unit filtering for traditional duplicates",
)
@click.option(
    "--tiny-cutoff",
    type=int,
    default=cli_module.DEFAULT_TINY_UNIT_STATEMENT_CUTOFF,
    show_default=True,
    panel=Panel.DETECTION,
    help="Tiny code-unit statement cutoff (exclusive) for traditional filtering",
)
@click.option(
    "--include-review",
    is_flag=True,
    panel=Panel.OUTPUT,
    help=(
        "Also list semantic_review pairs (semantic match without corroboration) and "
        "lift the default --max-duplicates cap; implied by --show-all"
    ),
)
@click.option(
    "--show-all",
    is_flag=True,
    panel=Panel.OUTPUT,
    help=(
        "Show every hybrid tier plus the raw traditional/semantic duplicate lists and "
        "lift the default --max-duplicates cap (implies --include-review)"
    ),
)
@click.option(
    "--max-duplicates",
    type=REPORT_CAP,
    metavar="N|all",
    default=DEFAULT_MAX_DUPLICATES,
    show_default=True,
    panel=Panel.OUTPUT,
    help=(
        "Cap the primary duplicate list at N findings in JSON and terminal output: exact "
        "families first (each counts once), then actionable pairs; 'all' removes the cap, "
        "as do --include-review, --show-all, and --full-table unless N is given. Raw "
        "--show-all lists stay complete; the exit code counts every finding"
    ),
)
@click.option(
    "--max-unused",
    type=REPORT_CAP,
    metavar="N|all",
    default=DEFAULT_MAX_UNUSED,
    show_default=True,
    panel=Panel.OUTPUT,
    help=(
        "Cap the potentially-unused list at N units, largest line span first, in JSON and "
        "terminal output; 'all' removes the cap, as do --include-review, --show-all, and "
        "--full-table unless N is given. The exit code counts every finding"
    ),
)
@click.option(
    "--show-source",
    is_flag=True,
    panel=Panel.OUTPUT,
    help=(
        "Show a source snippet per reported unit (terminal panels; JSON adds "
        "`source` to every unit record)"
    ),
)
@click.option(
    "--source-lines",
    type=REPORT_CAP,
    metavar="N|all",
    default=DEFAULT_SOURCE_LINES,
    show_default=True,
    panel=Panel.OUTPUT,
    help="Max source lines per reported unit; implies --show-source ('all' removes the cap)",
)
@click.option(
    "--show-diff",
    is_flag=True,
    panel=Panel.OUTPUT,
    help=(
        "Show a unified source diff per duplicate pair; exact families diff each member "
        "against the first when their source differs"
    ),
)
@click.option(
    "--full-table",
    is_flag=True,
    panel=Panel.OUTPUT,
    help=(
        "Show all rows in the raw --show-all duplicate tables and lift the default "
        "--max-duplicates and --max-unused caps"
    ),
)
@click.option(
    "--fail-on",
    type=click.Choice(["actionable", "all", "none"]),
    default="actionable",
    show_default=True,
    panel=Panel.OUTPUT,
    help="Which findings make the exit code 1",
)
@click.option(
    "--fail-on-incomplete",
    is_flag=True,
    panel=Panel.OUTPUT,
    help=(
        "Also exit 1 when the analysis did not complete (empty extraction, a file-level "
        "extraction failure, semantic fallback, or a file skipped by unused analysis), "
        "independent of --fail-on, including --fail-on none"
    ),
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
    with _configured_cli_output(
        as_json=opts.as_json,
        verbose=opts.verbose,
        output_width=opts.output_width,
    ):
        try:
            config = opts.to_analysis_config(path)
        except ValueError as exc:
            raise click.UsageError(str(exc)) from exc
        focus_paths = resolve_focus_paths(opts.focus, path)

        result = _run_cli_action(
            lambda: cli_module.CodeAnalyzer(config).analyze(path),
            error_label="analysis",
            verbose=opts.verbose,
            catch_file_not_found=True,
        )
        if focus_paths:
            result = focus_result(result, focus_paths)
        # Exit status is decided on the complete result before any report
        # selection so visibility flags can never change CI outcomes.
        exit_code = int(
            run_should_fail(
                result,
                policy=opts.fail_on,
                strict_unused=opts.strict_unused,
                fail_on_incomplete=opts.fail_on_incomplete,
            )
        )
        selection = select_findings(result, opts.report_policy)

        if opts.as_json:
            report_text = to_json_text(
                check_result_to_json(
                    selection,
                    fail_on=opts.fail_on,
                    exit_code=exit_code,
                    strict_unused=opts.strict_unused,
                    fail_on_incomplete=opts.fail_on_incomplete,
                    include_source=opts.show_source,
                    source_lines=opts.source_lines,
                )
            )
        else:
            if result.checks.extraction.status == "empty":
                _output.error_console.print(
                    "[yellow]Warning:[/yellow] extraction produced no code units; ensure "
                    "the path contains supported source code and that extraction filters "
                    "permit it."
                )
            print_run(result.run, result.checks)
            print_summary(
                selection,
                fail_on=opts.fail_on,
                exit_code=exit_code,
                strict_unused=opts.strict_unused,
                fail_on_incomplete=opts.fail_on_incomplete,
            )
            print_findings(
                selection,
                show_source=opts.show_source,
                source_lines=opts.source_lines,
                show_diff=opts.show_diff,
                max_items=opts.table_max_items,
                strict_unused=opts.strict_unused,
            )

    if opts.as_json:
        print(report_text)
    raise click.exceptions.Exit(exit_code)
