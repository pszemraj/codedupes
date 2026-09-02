"""Implementation of the ``codedupes search`` command."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import rich_click as click

import codedupes.cli as cli_module
from codedupes.constants import (
    DEFAULT_SEARCH_SEMANTIC_TASK,
    DEFAULT_TOP_K,
    SEMANTIC_TASK_CHOICES,
)

from . import _output
from ._json import print_search_json
from ._options import Panel, SearchOptions, option_panels, semantic_options
from ._output import _configured_cli_output, _run_cli_action, _validate_positive_int
from ._render import _print_diagnostics, print_search_results


@cli_module.cli.command(
    "search",
    help="Search for semantically similar code",
    context_settings={"auto_envvar_prefix": "CODEDUPES"},
)
@click.argument("path", type=click.Path(path_type=Path, exists=True), panel=Panel.SCOPE)
@click.argument("query", panel=Panel.SCOPE)
@click.option(
    "--top-k",
    type=int,
    default=DEFAULT_TOP_K,
    show_default=True,
    callback=_validate_positive_int,
    panel=Panel.DETECTION,
    show_envvar=True,
    help="Maximum results",
)
@click.option(
    "--threshold",
    type=float,
    default=None,
    show_default=False,
    panel=Panel.DETECTION,
    show_envvar=True,
    help="Shared threshold override for semantic search",
)
@click.option(
    "--semantic-threshold",
    type=float,
    panel=Panel.DETECTION,
    show_envvar=True,
    help="Override semantic threshold",
)
@click.option(
    "--semantic-task",
    type=click.Choice(SEMANTIC_TASK_CHOICES),
    default=DEFAULT_SEARCH_SEMANTIC_TASK,
    show_default=True,
    panel=Panel.SEMANTIC,
    show_envvar=True,
    help="Semantic task mode for query/document embeddings",
)
@click.option(
    "--search-document",
    type=click.Choice(["source", "contextual"]),
    default="source",
    show_default=True,
    panel=Panel.SEMANTIC,
    show_envvar=True,
    help="Text representation embedded for each search-index unit",
)
@semantic_options("search")
@option_panels
@click.pass_context
def search_command(ctx: click.Context, path: Path, query: str, **params: Any) -> None:
    """Run semantic search over extracted code units.

    :param ctx: Active Click context.
    :param path: File or directory to index.
    :param query: Natural-language search query.
    :param params: Parsed command options.
    :return: ``None``.
    """
    opts = SearchOptions.from_params(ctx, params)
    try:
        config = opts.to_analysis_config()
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc

    with _configured_cli_output(
        as_json=opts.as_json,
        verbose=opts.verbose,
        output_width=opts.output_width,
    ):
        analyzer = cli_module.CodeAnalyzer(config)
        indexed_units = _run_cli_action(
            lambda: analyzer.index(path),
            error_label="search",
            verbose=opts.verbose,
            catch_file_not_found=True,
        )
        results = _run_cli_action(
            lambda: analyzer.search(query, top_k=opts.top_k),
            error_label="search",
            verbose=opts.verbose,
        )

        if opts.as_json:
            print_search_json(
                query,
                results,
                analyzer.semantic_diagnostics,
                indexed_units,
                analyzer.embedding_stats,
            )
        else:
            _output.console.print(f"[bold cyan]Query:[/bold cyan] {query!r}")
            if indexed_units == 0:
                if analyzer.extracted_unit_count == 0:
                    reason = (
                        "extraction produced no code units; ensure the path contains "
                        "supported source code and that extraction filters permit it"
                    )
                elif analyzer.semantic_diagnostics:
                    reason = (
                        "no semantic candidates survived indexing; inspect the semantic "
                        "diagnostics below"
                    )
                else:
                    reason = (
                        f"semantic eligibility filtering removed all "
                        f"{analyzer.extracted_unit_count} extracted unit(s); adjust "
                        "--min-statements or --semantic-unit-type"
                    )
                _output.error_console.print(
                    "[yellow]Warning:[/yellow] the search index is empty, so no query can "
                    f"match: {reason}."
                )
            _print_diagnostics("Semantic diagnostics", analyzer.semantic_diagnostics)
            print_search_results(results)

    raise click.exceptions.Exit(0)
