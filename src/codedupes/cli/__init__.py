"""Command-line interface for codedupes."""

from __future__ import annotations

import sys
from typing import Any

import rich_click as click

from codedupes import __version__
from codedupes.analyzer import (
    DEFAULT_TINY_UNIT_STATEMENT_CUTOFF,
    AnalyzerConfig,
    CodeAnalyzer,
)
from codedupes.constants import (
    DEFAULT_MODEL,
    DEFAULT_SEMANTIC_DEVICE,
    DEFAULT_TRADITIONAL_THRESHOLD,
)
from codedupes.devices import configure_mps_environment, get_device_diagnostics
from codedupes.embedding_cache import EmbeddingCache
from codedupes.semantic_profiles import resolve_model_profile

from . import _output
from ._json import _unit_to_dict as _unit_to_dict
from ._options import Panel, options_in_panels
from ._output import DEFAULT_OUTPUT_WIDTH, setup_logging
from ._render import _syntax_lexer as _syntax_lexer
from ._render import format_location


@click.group(
    context_settings={
        "help_option_names": ["-h", "--help"],
        "auto_envvar_prefix": "CODEDUPES",
    },
    no_args_is_help=False,
    invoke_without_command=True,
)
@click.rich_config(
    {
        "options_table_column_types": ["opt_long", "opt_short", "help"],
        "options_table_help_sections": ["metavar", "help", "default", "envvar"],
    }
)
@click.version_option(__version__, prog_name="codedupes")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Detect duplicate and unused source code using structural and semantic analysis."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        ctx.exit(2)


# Importing the command modules registers them on the group above. Public names
# remain re-exported here because callers and tests historically import them from
# ``codedupes.cli``.
from .cache import cache_clear_command, cache_group, cache_info_command
from .check import check_command, run_should_fail
from .info import info_command
from .search import search_command


def __getattr__(name: str) -> Any:
    """Expose the currently configured Rich consoles without stale aliases."""
    if name in {"console", "error_console"}:
        return getattr(_output, name)
    raise AttributeError(name)


def main() -> int:
    """Run the installed ``codedupes`` command and return its process exit code.

    ``pyproject.toml`` registers this callable as the supported console entry point.
    """
    argv = sys.argv[1:]
    try:
        result = cli.main(args=argv, prog_name="codedupes", standalone_mode=False)
        if isinstance(result, int):
            return result
    except click.exceptions.Exit as exc:
        return int(exc.exit_code)
    except click.ClickException as exc:
        exc.show()
        return exc.exit_code
    except click.Abort:
        click.echo("Aborted!", err=True)
        return 1
    return 0


__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_OUTPUT_WIDTH",
    "DEFAULT_SEMANTIC_DEVICE",
    "DEFAULT_TINY_UNIT_STATEMENT_CUTOFF",
    "DEFAULT_TRADITIONAL_THRESHOLD",
    "AnalyzerConfig",
    "CodeAnalyzer",
    "EmbeddingCache",
    "Panel",
    "cache_clear_command",
    "cache_group",
    "cache_info_command",
    "check_command",
    "cli",
    "configure_mps_environment",
    "format_location",
    "get_device_diagnostics",
    "info_command",
    "main",
    "options_in_panels",
    "resolve_model_profile",
    "run_should_fail",
    "search_command",
    "setup_logging",
]
