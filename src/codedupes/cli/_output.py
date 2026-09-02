"""CLI logging, console configuration, and runtime error handling."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TypeVar

import rich_click as click
from rich.console import Console
from rich.logging import RichHandler

from codedupes.languages import GrammarUnavailableError
from codedupes.logging_utils import quiet_dependency_loggers

DEFAULT_OUTPUT_WIDTH = 160
MIN_OUTPUT_WIDTH = 80

console = Console(width=DEFAULT_OUTPUT_WIDTH)
# Errors and warnings never share stdout: `--json` promises machine-parseable
# JSON only on stdout, so diagnostics go to stderr.
error_console = Console(stderr=True, width=DEFAULT_OUTPUT_WIDTH)
TResult = TypeVar("TResult")


class _CodedupesLogFilter(logging.Filter):
    """Filter log records so non-codedupes INFO chatter is hidden by default."""

    def __init__(self, *, include_external_info: bool) -> None:
        """Create a log filter configured for CLI verbosity."""
        super().__init__()
        self.include_external_info = include_external_info

    def filter(self, record: logging.LogRecord) -> bool:
        """Return whether a log record should be emitted."""
        if record.name.startswith("codedupes"):
            return True
        if self.include_external_info:
            return True
        return record.levelno >= logging.WARNING


def _set_console(output_width: int) -> None:
    """Set global stdout/stderr consoles used by all rich output helpers."""
    global console, error_console
    console = Console(width=output_width)
    error_console = Console(stderr=True, width=output_width)


def _suppress_logs_for_json() -> tuple[int, list[logging.Handler]]:
    """Prevent log output from contaminating JSON responses."""
    root_logger = logging.getLogger()
    prior_state = (root_logger.level, list(root_logger.handlers))
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.CRITICAL + 1)
    return prior_state


def _restore_root_logger_state(prior_state: tuple[int, list[logging.Handler]]) -> None:
    """Restore root logger level/handlers after temporary JSON suppression."""
    prior_level, prior_handlers = prior_state
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
    for handler in prior_handlers:
        root_logger.addHandler(handler)
    root_logger.setLevel(prior_level)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging with a Rich stderr handler."""
    level = logging.DEBUG if verbose else logging.INFO
    handler = RichHandler(console=error_console, show_time=False, show_path=False)
    handler.addFilter(_CodedupesLogFilter(include_external_info=verbose))
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[handler],
        force=True,
    )
    quiet_dependency_loggers(logging.DEBUG if verbose else logging.WARNING)


@contextmanager
def _configured_cli_output(
    *,
    as_json: bool,
    verbose: bool,
    output_width: int,
) -> Iterator[None]:
    """Configure logging/console for a CLI command and restore it on exit."""
    _set_console(output_width)
    logging_state: tuple[int, list[logging.Handler]] | None = None
    if as_json:
        logging_state = _suppress_logs_for_json()
    else:
        setup_logging(verbose)

    try:
        yield
    finally:
        if logging_state is not None:
            _restore_root_logger_state(logging_state)


def _run_cli_action(
    action: Callable[[], TResult],
    *,
    error_label: str,
    verbose: bool,
    catch_file_not_found: bool = False,
) -> TResult:
    """Run a command action and normalize runtime exception handling."""
    try:
        return action()
    except FileNotFoundError as exc:
        if not catch_file_not_found:
            raise
        error_console.print(f"[red]Error:[/red] {exc}")
        raise click.exceptions.Exit(1) from exc
    except GrammarUnavailableError as exc:
        error_console.print(f"[red]Parser unavailable:[/red] {exc}")
        error_console.print("Run `codedupes info` to check Tree-sitter parser package status.")
        raise click.exceptions.Exit(1) from exc
    except Exception as exc:
        error_console.print(f"[red]Error during {error_label}:[/red] {exc}")
        if verbose:
            error_console.print_exception()
        raise click.exceptions.Exit(1) from exc


def _validate_positive_int(_ctx: click.Context, _param: click.Parameter, value: int) -> int:
    """Validate a positive integer option that never reaches ``AnalyzerConfig``."""
    if value <= 0:
        raise click.BadParameter("must be > 0")
    return value


def _validate_output_width(_ctx: click.Context, _param: click.Parameter, value: int) -> int:
    """Validate output width for Rich table rendering."""
    if value < MIN_OUTPUT_WIDTH:
        raise click.BadParameter(f"must be >= {MIN_OUTPUT_WIDTH}")
    return value


def _is_cli_explicit(ctx: click.Context, option_name: str) -> bool:
    """Return whether a CLI option was explicitly provided on the command line."""
    return ctx.get_parameter_source(option_name) == click.core.ParameterSource.COMMANDLINE


def _validate_json_output_controls(
    *,
    as_json: bool,
    verbose: bool,
    output_width_explicit: bool,
    show_source: bool = False,
    full_table: bool = False,
) -> None:
    """Reject flags that are incompatible with JSON-only output mode."""
    if not as_json:
        return

    incompatible: list[str] = []
    if verbose:
        incompatible.append("--verbose")
    if output_width_explicit:
        incompatible.append("--output-width")
    if show_source:
        incompatible.append("--show-source")
    if full_table:
        incompatible.append("--full-table")

    if incompatible:
        listed = ", ".join(incompatible)
        raise click.UsageError(f"Cannot use {listed} with --json.")
