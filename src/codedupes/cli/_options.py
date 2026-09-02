"""CLI option bundles and decorators."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal, TypeVar

import rich_click as click

from codedupes.analyzer import (
    DEFAULT_SEMANTIC_UNIT_TYPES,
    SEMANTIC_UNIT_TYPE_CHOICES,
)
from codedupes.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MIN_SEMANTIC_STATEMENTS,
    DEFAULT_MODEL,
    DEFAULT_SEMANTIC_DEVICE,
    DEFAULT_TRADITIONAL_THRESHOLD,
    SEMANTIC_DEVICE_CHOICES,
)
from codedupes.semantic import ProgressMode

from ._output import (
    DEFAULT_OUTPUT_WIDTH,
    _is_cli_explicit,
    _validate_json_output_controls,
    _validate_output_width,
)

F = TypeVar("F", bound=Callable[..., Any])
DEFAULT_EXCLUDE_HELP_HINT = (
    "Replace default test-file globs with patterns to exclude (repeat for multiple patterns). "
    "Built-in common artifact-directory excludes always apply."
)


class Panel(StrEnum):
    """Help-panel names shared by option definitions and validation."""

    SCOPE = "Scope"
    DETECTION = "Detection"
    SEMANTIC = "Semantic model"
    DEVICE = "Device"
    CACHE = "Cache"
    OUTPUT = "Output"


SEMANTIC_ONLY_PANELS = frozenset({Panel.SEMANTIC, Panel.DEVICE})


def options_in_panels(command: click.Command, panels: frozenset[Panel]) -> list[str]:
    """Return parameter names assigned to any requested help panel."""
    return [
        parameter.name
        for parameter in command.params
        if parameter.name is not None and getattr(parameter, "panel", None) in panels
    ]


def option_panels(func: F) -> F:
    """Declare the common option-panel order on a command."""
    for panel in reversed(tuple(Panel)):
        func = click.option_panel(panel.value)(func)
    return func


def _resolve_check_thresholds(
    threshold: float | None,
    semantic_threshold: float | None,
    traditional_threshold: float | None,
) -> tuple[float | None, float]:
    """Resolve semantic and traditional thresholds using CLI precedence."""
    return (
        semantic_threshold if semantic_threshold is not None else threshold,
        (
            traditional_threshold
            if traditional_threshold is not None
            else threshold
            if threshold is not None
            else DEFAULT_TRADITIONAL_THRESHOLD
        ),
    )


def _resolve_search_threshold(
    threshold: float | None,
    semantic_threshold: float | None,
) -> float | None:
    """Resolve the explicit semantic threshold override for search mode."""
    return semantic_threshold if semantic_threshold is not None else threshold


def _resolve_native_pair(values: tuple[bool, ...], flag: str) -> bool | None:
    """Resolve repeated values from one native Click boolean flag pair."""
    if True in values and False in values:
        raise click.UsageError(f"Cannot combine --{flag} and --no-{flag}.")
    return values[-1] if values else None


def _display_option(name: str, params: dict[str, Any]) -> str:
    """Return the spelling used for a possibly negated boolean option."""
    if name in {"trust_remote_code", "mps_fallback"} and params[name] == (False,):
        return f"--no-{name.replace('_', '-')}"
    return f"--{name.replace('_', '-')}"


@dataclass(frozen=True)
class SemanticOptions:
    """Shared semantic-analysis command options."""

    model: str
    semantic_task: str
    instruction_prefix: str | None
    model_revision: str | None
    trust_remote_code: bool | None
    device: str
    mps_fallback: bool | None
    mps_memory_fraction: float | None
    batch_size: int
    min_statements: int
    semantic_unit_type: tuple[str, ...]
    no_cache: bool
    strict_revision_cache: bool
    progress: ProgressMode

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> SemanticOptions:
        """Build shared semantic options from Click parameters."""
        return cls(
            model=params["model"],
            semantic_task=params["semantic_task"],
            instruction_prefix=params["instruction_prefix"],
            model_revision=params["model_revision"],
            trust_remote_code=_resolve_native_pair(
                params["trust_remote_code"], "trust-remote-code"
            ),
            device=params["device"],
            mps_fallback=_resolve_native_pair(params["mps_fallback"], "mps-fallback"),
            mps_memory_fraction=params["mps_memory_fraction"],
            batch_size=params["batch_size"],
            min_statements=params["min_statements"],
            semantic_unit_type=params["semantic_unit_type"],
            no_cache=params["no_cache"],
            strict_revision_cache=params["strict_revision_cache"],
            progress="never" if params["as_json"] else "auto",
        )

    def analysis_kwargs(self) -> dict[str, Any]:
        """Return analyzer keyword arguments shared by check and search."""
        return {
            "model_name": self.model,
            "semantic_task": self.semantic_task,
            "instruction_prefix": self.instruction_prefix,
            "model_revision": self.model_revision,
            "trust_remote_code": self.trust_remote_code,
            "device": self.device,
            "mps_fallback": self.mps_fallback,
            "mps_memory_fraction": self.mps_memory_fraction,
            "batch_size": self.batch_size,
            "min_semantic_statements": self.min_statements,
            "semantic_unit_types": self.semantic_unit_type,
            "embedding_cache": not self.no_cache,
            "strict_revision_cache": self.strict_revision_cache,
            "progress": self.progress,
        }


@dataclass(frozen=True)
class CheckOptions:
    """Validated options for the ``check`` command."""

    semantic: SemanticOptions
    languages: tuple[str, ...]
    no_private: bool
    exclude: tuple[str, ...]
    include_stubs: bool
    as_json: bool
    verbose: bool
    output_width: int
    threshold: float | None
    semantic_threshold: float | None
    traditional_threshold: float | None
    cross_language: bool
    semantic_only: bool
    traditional_only: bool
    allow_semantic_fallback: bool
    no_unused: bool
    strict_unused: bool
    suppress_test_semantic: bool
    no_tiny_filter: bool
    tiny_cutoff: int
    tiny_near_jaccard_min: float
    show_all: bool
    show_source: bool
    full_table: bool
    fail_on: Literal["actionable", "all", "none"]

    @classmethod
    def from_params(cls, ctx: click.Context, params: dict[str, Any]) -> CheckOptions:
        """Validate and build one ``check`` option bundle."""
        if params["no_unused"] and params["strict_unused"]:
            raise click.UsageError(
                "Cannot combine --no-unused and --strict-unused because unused reporting is disabled."
            )
        if params["semantic_only"] and params["traditional_only"]:
            raise click.UsageError("Cannot use both --semantic-only and --traditional-only.")
        if params["allow_semantic_fallback"] and (
            params["semantic_only"] or params["traditional_only"]
        ):
            raise click.UsageError(
                "--allow-semantic-fallback is only valid in default combined mode."
            )
        if params["show_all"] and (params["semantic_only"] or params["traditional_only"]):
            raise click.UsageError("--show-all is only valid in default combined mode.")

        _validate_json_output_controls(
            as_json=params["as_json"],
            verbose=params["verbose"],
            output_width_explicit=_is_cli_explicit(ctx, "output_width"),
            show_source=params["show_source"],
            full_table=params["full_table"],
        )

        if params["traditional_only"]:
            specified = [
                name
                for name in options_in_panels(ctx.command, SEMANTIC_ONLY_PANELS)
                if _is_cli_explicit(ctx, name)
            ]
            if specified:
                listed = ", ".join(_display_option(name, params) for name in specified)
                raise click.UsageError(
                    f"Cannot use {listed} with --traditional-only; semantic analysis is disabled."
                )

        if params["semantic_only"]:
            specified = [
                name
                for name in (
                    "traditional_threshold",
                    "no_tiny_filter",
                    "tiny_cutoff",
                    "tiny_near_jaccard_min",
                )
                if _is_cli_explicit(ctx, name)
            ]
            if specified:
                listed = ", ".join(f"--{name.replace('_', '-')}" for name in specified)
                raise click.UsageError(
                    f"Cannot use {listed} with --semantic-only; traditional duplicate analysis is disabled."
                )

        return cls(
            semantic=SemanticOptions.from_params(params),
            **{name: params[name] for name in cls.__dataclass_fields__ if name != "semantic"},
        )

    @property
    def combined_mode(self) -> bool:
        """Return whether both duplicate-detection methods are enabled."""
        return not self.semantic_only and not self.traditional_only

    @property
    def table_max_items(self) -> int | None:
        """Return the terminal table row cap."""
        return None if self.full_table else 20

    def to_analysis_config(self) -> Any:
        """Build the analyzer config represented by this option bundle."""
        import codedupes.cli as cli_module

        semantic_threshold, traditional_threshold = _resolve_check_thresholds(
            self.threshold,
            self.semantic_threshold,
            self.traditional_threshold,
        )
        semantic_kwargs = self.semantic.analysis_kwargs()
        if self.semantic_only:
            traditional_threshold = DEFAULT_TRADITIONAL_THRESHOLD
        if self.traditional_only:
            semantic_threshold = None
            semantic_kwargs["semantic_task"] = None

        return cli_module.AnalyzerConfig(
            exclude_patterns=list(self.exclude) or None,
            include_private=not self.no_private,
            languages=self.languages or None,
            jaccard_threshold=traditional_threshold,
            semantic_threshold=semantic_threshold,
            cross_language=self.cross_language,
            run_traditional=not self.semantic_only,
            run_semantic=not self.traditional_only,
            allow_semantic_fallback=self.allow_semantic_fallback,
            run_unused=not self.no_unused,
            filter_tiny_traditional=not self.no_tiny_filter,
            tiny_unit_statement_cutoff=self.tiny_cutoff,
            tiny_near_jaccard_min=self.tiny_near_jaccard_min,
            strict_unused=self.strict_unused,
            suppress_test_semantic_matches=self.suppress_test_semantic,
            include_stubs=self.include_stubs,
            **semantic_kwargs,
        )


@dataclass(frozen=True)
class SearchOptions:
    """Validated options for the ``search`` command."""

    semantic: SemanticOptions
    languages: tuple[str, ...]
    no_private: bool
    exclude: tuple[str, ...]
    include_stubs: bool
    as_json: bool
    verbose: bool
    output_width: int
    top_k: int
    threshold: float | None
    semantic_threshold: float | None
    search_document: Literal["source", "contextual"]

    @classmethod
    def from_params(cls, ctx: click.Context, params: dict[str, Any]) -> SearchOptions:
        """Validate and build one ``search`` option bundle."""
        _validate_json_output_controls(
            as_json=params["as_json"],
            verbose=params["verbose"],
            output_width_explicit=_is_cli_explicit(ctx, "output_width"),
        )
        return cls(
            semantic=SemanticOptions.from_params(params),
            **{name: params[name] for name in cls.__dataclass_fields__ if name != "semantic"},
        )

    def to_analysis_config(self) -> Any:
        """Build the analyzer config represented by this option bundle."""
        import codedupes.cli as cli_module

        return cli_module.AnalyzerConfig(
            mode="search",
            exclude_patterns=list(self.exclude) or None,
            include_private=not self.no_private,
            languages=self.languages or None,
            semantic_threshold=_resolve_search_threshold(
                self.threshold,
                self.semantic_threshold,
            ),
            run_traditional=False,
            run_unused=False,
            include_stubs=self.include_stubs,
            search_document=self.search_document,
            **self.semantic.analysis_kwargs(),
        )


def semantic_options(command: Literal["check", "search"]) -> Callable[[F], F]:
    """Attach shared scope, semantic, device, cache, and output options."""
    scope_suffix = (
        " (also narrows traditional duplicate scope in combined mode)" if command == "check" else ""
    )
    options = [
        click.option(
            "--language",
            "languages",
            multiple=True,
            type=str,
            metavar="LANGUAGE",
            panel=Panel.SCOPE,
            show_envvar=True,
            help=(
                "Limit extraction to a language (repeat for multiple). Aliases such as py, rs, "
                "js, jsx, ts, and tsx are accepted. Omit to auto-detect all supported languages."
            ),
        ),
        click.option(
            "--no-private",
            is_flag=True,
            panel=Panel.SCOPE,
            show_envvar=True,
            help="Exclude private functions/classes",
        ),
        click.option(
            "--exclude",
            multiple=True,
            panel=Panel.SCOPE,
            show_envvar=True,
            help=DEFAULT_EXCLUDE_HELP_HINT,
        ),
        click.option(
            "--include-stubs",
            is_flag=True,
            panel=Panel.SCOPE,
            show_envvar=True,
            help=(
                "Include .pyi files when scanning a directory "
                "(single-file targets are analyzed as given)"
            ),
        ),
        click.option(
            "--min-statements",
            type=int,
            default=DEFAULT_MIN_SEMANTIC_STATEMENTS,
            show_default=True,
            panel=Panel.SEMANTIC,
            show_envvar=True,
            help=f"Skip semantic comparison for code units with fewer body statements{scope_suffix}",
        ),
        click.option(
            "--semantic-unit-type",
            multiple=True,
            type=click.Choice(SEMANTIC_UNIT_TYPE_CHOICES),
            default=DEFAULT_SEMANTIC_UNIT_TYPES,
            show_default=True,
            panel=Panel.SEMANTIC,
            show_envvar=True,
            help=(
                "Unit type(s) eligible for semantic embedding "
                f"(repeat option to add more){scope_suffix}"
            ),
        ),
        click.option(
            "--model",
            default=DEFAULT_MODEL,
            show_default=True,
            panel=Panel.SEMANTIC,
            show_envvar=True,
            help="Embedding model alias, Hugging Face model ID, or complete local model directory",
        ),
        click.option(
            "--instruction-prefix",
            default=None,
            panel=Panel.SEMANTIC,
            show_envvar=True,
            help="Custom instruction prefix prepended to semantic inputs",
        ),
        click.option(
            "--model-revision",
            default=None,
            show_default="auto",
            panel=Panel.SEMANTIC,
            show_envvar=True,
            help="Model revision/commit. If omitted, uses the model-profile default.",
        ),
        click.option(
            "--trust-remote-code/--no-trust-remote-code",
            default=None,
            multiple=True,
            panel=Panel.SEMANTIC,
            show_envvar=True,
            help="Override the model profile's remote-code trust setting",
        ),
        click.option(
            "--strict-revision-cache",
            is_flag=True,
            panel=Panel.SEMANTIC,
            show_envvar=True,
            help=(
                "Key an unpinned hub model's cache revision to a resolved commit hash instead of "
                "the requested revision label"
            ),
        ),
        click.option(
            "--device",
            type=click.Choice(SEMANTIC_DEVICE_CHOICES),
            default=DEFAULT_SEMANTIC_DEVICE,
            show_default=True,
            panel=Panel.DEVICE,
            show_envvar=True,
            help="Semantic inference device (auto prefers CUDA, then MPS, then CPU)",
        ),
        click.option(
            "--mps-fallback/--no-mps-fallback",
            default=None,
            multiple=True,
            panel=Panel.DEVICE,
            show_envvar=True,
            help="Override MPS unsupported-op CPU fallback",
        ),
        click.option(
            "--mps-memory-fraction",
            type=float,
            default=None,
            panel=Panel.DEVICE,
            show_envvar=True,
            help=(
                "Optional PyTorch MPS allocator limit as a fraction of the recommended working "
                "set, in (0, 2]. Values above 1 increase system memory pressure."
            ),
        ),
        click.option(
            "--batch-size",
            type=int,
            default=DEFAULT_BATCH_SIZE,
            show_default=True,
            panel=Panel.DEVICE,
            show_envvar=True,
            help="Batch size for embeddings",
        ),
        click.option(
            "--no-cache",
            is_flag=True,
            panel=Panel.CACHE,
            show_envvar=True,
            help="Disable the persistent on-disk embedding cache for this run",
        ),
        click.option(
            "--json",
            "as_json",
            is_flag=True,
            panel=Panel.OUTPUT,
            show_envvar=True,
            help="Output JSON instead of rich tables",
        ),
        click.option(
            "--verbose",
            "-v",
            is_flag=True,
            panel=Panel.OUTPUT,
            show_envvar=True,
            help="Verbose logging",
        ),
        click.option(
            "--output-width",
            type=int,
            default=DEFAULT_OUTPUT_WIDTH,
            show_default=True,
            callback=_validate_output_width,
            panel=Panel.OUTPUT,
            show_envvar=True,
            help="Width used for rich terminal rendering",
        ),
    ]

    def decorator(func: F) -> F:
        for option in reversed(options):
            func = option(func)
        return func

    return decorator
