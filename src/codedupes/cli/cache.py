"""Embedding-cache CLI subcommands."""

from __future__ import annotations

from typing import Any

import rich_click as click
from rich.panel import Panel
from rich.text import Text

import codedupes.cli as cli_module
from codedupes.semantic_profiles import (
    is_explicit_local_model_path,
    resolve_local_model_path,
    resolve_model_profile,
)

from . import _output
from ._options import output_width_option
from ._output import _configured_cli_output
from ._render import _settings_panel


@cli_module.cli.group("cache", help="Inspect or clear the persistent embedding cache")
def cache_group() -> None:
    """Group namespace for embedding-cache management subcommands."""


def _cache_summary_panel(stats: dict[str, Any]) -> Panel:
    """Build the cache summary shared by ``info`` and ``cache info``.

    :param stats: Embedding cache statistics.
    :return: Rich summary panel.
    """
    return _settings_panel(
        "Embedding cache",
        [
            ("Cache path", stats["path"]),
            ("Disabled via CODEDUPES_NO_CACHE", stats["disabled"]),
            ("Entries", stats["entries"]),
            ("Size on disk", f"{stats['size_bytes']} bytes"),
        ],
    )


@cache_group.command("info", help="Show embedding cache location, size, and breakdown")
@output_width_option
def cache_info_command(output_width: int) -> None:
    """Display cache summary and per-model/per-repo breakdown in Rich panels.

    :param output_width: Width used for Rich terminal rendering.
    """
    with _configured_cli_output(as_json=False, verbose=False, output_width=output_width):
        console = _output.console
        try:
            stats = cli_module.EmbeddingCache().stats()
        except Exception as exc:
            _output.error_console.print(
                f"Cache unavailable: {exc}", style="red", markup=False, highlight=False
            )
            raise click.exceptions.Exit(1) from exc
        console.print(_cache_summary_panel(stats))
        if stats["models"]:
            console.print(
                _settings_panel("Per-model entry counts", sorted(stats["models"].items()))
            )
        if stats["repos"]:
            console.print(
                _settings_panel(
                    "Per-repo breakdown",
                    [
                        (
                            repo["repo"],
                            (
                                f"{repo['shards']} shard(s), {repo['entries']} entries, "
                                f"{repo['size_bytes']} bytes, {repo['orphan_rows']} orphan rows, "
                                f"last complete generation {repo['last_complete_generation']}"
                            ),
                        )
                        for repo in stats["repos"]
                    ],
                )
            )


@cache_group.command("clear", help="Clear cached embeddings")
@click.option(
    "--model",
    default=None,
    help="Only clear entries for this model alias or canonical HuggingFace ID",
)
@output_width_option
def cache_clear_command(model: str | None, output_width: int) -> None:
    """Clear cached embeddings, optionally scoped to a single model.

    :param model: Optional model alias or canonical identifier to clear.
    :param output_width: Width used for Rich terminal rendering.
    """
    with _configured_cli_output(as_json=False, verbose=False, output_width=output_width):
        console = _output.console
        error_console = _output.error_console
        if model is not None and not model.strip():
            raise click.BadParameter("must not be empty", param_hint="--model")
        if (
            model
            and is_explicit_local_model_path(model)
            and resolve_local_model_path(model) is None
        ):
            error_console.print(
                Text(
                    f"Local model directory '{model}' does not exist, so its cache identity "
                    "cannot be resolved; run `codedupes cache clear` without --model to drop "
                    "its entries.",
                    style="yellow",
                )
            )
        canonical_model = resolve_model_profile(model).canonical_name if model else None
        try:
            clear_result = cli_module.EmbeddingCache().clear(model=canonical_model)
        except Exception as exc:
            error_console.print(Text(f"Cache clear failed: {exc}", style="red"))
            raise click.exceptions.Exit(1) from exc
        if clear_result.failed_deletions:
            error_console.print(
                Text(
                    f"Cache clear incomplete: removed {clear_result.removed_entries} cached "
                    f"embedding(s), but {clear_result.failed_deletions} deletion operation(s) failed.",
                    style="red",
                )
            )
            raise click.exceptions.Exit(1)
        if model:
            console.print(
                Text(
                    f"Cleared {clear_result.removed_entries} cached embedding(s) for model '{model}' "
                    f"({canonical_model}).",
                    style="green",
                )
            )
        else:
            console.print(
                Text(f"Cleared {clear_result.removed_entries} cached embedding(s).", style="green")
            )
