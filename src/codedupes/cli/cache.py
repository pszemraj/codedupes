"""Embedding-cache CLI subcommands."""

from __future__ import annotations

from typing import Any

import rich_click as click
from rich.console import Console
from rich.table import Table

import codedupes.cli as cli_module
from codedupes.semantic_profiles import (
    is_explicit_local_model_path,
    resolve_local_model_path,
    resolve_model_profile,
)

from ._options import output_width_option
from ._render import _settings_table


@cli_module.cli.group("cache", help="Inspect or clear the persistent embedding cache")
def cache_group() -> None:
    """Group namespace for embedding-cache management subcommands."""


def _cache_summary_table(stats: dict[str, Any]) -> Table:
    """Build the cache summary shared by ``info`` and ``cache info``.

    :param stats: Embedding cache statistics.
    :return: Rich summary table.
    """
    return _settings_table(
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
    """Display cache summary and per-model/per-repo breakdown in Rich tables.

    :param output_width: Width used for Rich terminal rendering.
    """
    console = Console(width=output_width)
    try:
        stats = cli_module.EmbeddingCache().stats()
    except Exception as exc:
        Console(stderr=True, width=output_width).print(
            f"Cache unavailable: {exc}", style="red", markup=False, highlight=False
        )
        raise click.exceptions.Exit(1) from exc
    console.print(_cache_summary_table(stats))
    if stats["models"]:
        console.print(_settings_table("Per-model entry counts", sorted(stats["models"].items())))
    if stats["repos"]:
        console.print(
            _settings_table(
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
    console = Console(width=output_width, markup=False, highlight=False, style="green")
    error_console = Console(
        stderr=True, width=output_width, markup=False, highlight=False, style="red"
    )
    if model is not None and not model.strip():
        raise click.BadParameter("must not be empty", param_hint="--model")
    if model and is_explicit_local_model_path(model) and resolve_local_model_path(model) is None:
        error_console.print(
            f"Local model directory '{model}' does not exist, so its cache identity "
            "cannot be resolved; run `codedupes cache clear` without --model to drop "
            "its entries.",
        )
    canonical_model = resolve_model_profile(model).canonical_name if model else None
    try:
        clear_result = cli_module.EmbeddingCache().clear(model=canonical_model)
    except Exception as exc:
        error_console.print(f"Cache clear failed: {exc}")
        raise click.exceptions.Exit(1) from exc
    if clear_result.failed_deletions:
        error_console.print(
            f"Cache clear incomplete: removed {clear_result.removed_entries} cached "
            f"embedding(s), but {clear_result.failed_deletions} deletion operation(s) failed.",
        )
        raise click.exceptions.Exit(1)
    if model:
        console.print(
            f"Cleared {clear_result.removed_entries} cached embedding(s) for model '{model}' "
            f"({canonical_model})."
        )
    else:
        console.print(f"Cleared {clear_result.removed_entries} cached embedding(s).")
