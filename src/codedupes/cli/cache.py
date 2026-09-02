"""Embedding-cache CLI subcommands."""

from __future__ import annotations

from typing import Any

import rich_click as click

import codedupes.cli as cli_module
from codedupes.semantic_profiles import (
    is_explicit_local_model_path,
    resolve_local_model_path,
    resolve_model_profile,
)


@cli_module.cli.group("cache", help="Inspect or clear the persistent embedding cache")
def cache_group() -> None:
    """Group namespace for embedding-cache management subcommands."""


def _echo_cache_summary(stats: dict[str, Any]) -> None:
    """Print cache summary lines shared by ``info`` and ``cache info``."""
    click.echo(f"Cache path: {stats['path']}")
    click.echo(f"Disabled via CODEDUPES_NO_CACHE: {stats['disabled']}")
    click.echo(f"Entries: {stats['entries']}")
    click.echo(f"Size on disk: {stats['size_bytes']} bytes")


@cache_group.command("info", help="Show embedding cache location, size, and breakdown")
def cache_info_command() -> None:
    """Print cache path, entry counts, size, and per-model/per-repo breakdown."""
    try:
        stats = cli_module.EmbeddingCache().stats()
    except Exception as exc:
        click.echo(f"Cache unavailable: {exc}", err=True)
        raise click.exceptions.Exit(1) from exc
    _echo_cache_summary(stats)
    if stats["models"]:
        click.echo("Per-model entry counts:")
        for model_name, count in sorted(stats["models"].items()):
            click.echo(f"  - {model_name}: {count}")
    if stats["repos"]:
        click.echo("Per-repo breakdown:")
        for repo in stats["repos"]:
            click.echo(
                f"  - {repo['repo']}: {repo['shards']} shard(s), {repo['entries']} entries, "
                f"{repo['size_bytes']} bytes"
            )


@cache_group.command("clear", help="Clear cached embeddings")
@click.option(
    "--model",
    default=None,
    help="Only clear entries for this model alias or canonical HuggingFace ID",
)
def cache_clear_command(model: str | None) -> None:
    """Clear cached embeddings, optionally scoped to a single model."""
    if model and is_explicit_local_model_path(model) and resolve_local_model_path(model) is None:
        click.echo(
            f"Local model directory '{model}' does not exist, so its cache identity "
            "cannot be resolved; run `codedupes cache clear` without --model to drop "
            "its entries.",
            err=True,
        )
    canonical_model = resolve_model_profile(model).canonical_name if model else None
    try:
        clear_result = cli_module.EmbeddingCache().clear(model=canonical_model)
    except Exception as exc:
        click.echo(f"Cache clear failed: {exc}", err=True)
        raise click.exceptions.Exit(1) from exc
    if clear_result.failed_deletions:
        click.echo(
            f"Cache clear incomplete: removed {clear_result.removed_entries} cached "
            f"embedding(s), but {clear_result.failed_deletions} deletion operation(s) failed.",
            err=True,
        )
        raise click.exceptions.Exit(1)
    if model:
        click.echo(
            f"Cleared {clear_result.removed_entries} cached embedding(s) for model '{model}' "
            f"({canonical_model})."
        )
    else:
        click.echo(f"Cleared {clear_result.removed_entries} cached embedding(s).")
