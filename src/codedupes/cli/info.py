"""Implementation of the ``codedupes info`` command."""

from __future__ import annotations

import platform

import rich_click as click

import codedupes.cli as cli_module
from codedupes import __version__
from codedupes.constants import (
    DEFAULT_CHECK_SEMANTIC_TASK,
    DEFAULT_EXCLUDE_DIR_NAMES,
    DEFAULT_MIN_SEMANTIC_STATEMENTS,
    DEFAULT_MODEL,
    DEFAULT_SEARCH_SEMANTIC_TASK,
    DEFAULT_SEMANTIC_DEVICE,
    DEFAULT_TRADITIONAL_THRESHOLD,
)
from codedupes.devices import (
    cpu_bf16_opted_in,
    describe_mps_fallback_env,
    format_mps_memory_snapshot,
)
from codedupes.extractor import DEFAULT_EXCLUDE_PATTERNS
from codedupes.languages import SUPPORTED_LANGUAGES, get_grammar_statuses
from codedupes.semantic import get_semantic_runtime_versions
from codedupes.semantic_profiles import (
    SemanticModelProfile,
    get_default_search_threshold,
    list_supported_models,
    resolve_model_profile,
)

from . import _output
from ._options import Panel, output_width_option
from ._output import DEFAULT_OUTPUT_WIDTH, _configured_cli_output
from ._render import _settings_panel
from .cache import _cache_summary_panel


def _format_language_gates(profile: SemanticModelProfile) -> str:
    """Format one model profile's per-language semantic duplicate gates.

    :param profile: Semantic model profile.
    :return: Compact language-to-threshold text.
    """
    gates = ", ".join(
        f"{language}={gate}" for language, gate in profile.language_semantic_thresholds.items()
    )
    fallback = f"fallback={profile.default_semantic_threshold}"
    return f"{gates} ({fallback})" if gates else fallback


@cli_module.cli.command("info", help="Show a compact runtime, device, and model overview")
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    panel=Panel.OUTPUT,
    help="Include full runtime/device diagnostics, parser status, defaults, models, and cache",
)
@output_width_option
def info_command(output_width: int, verbose: bool) -> None:
    """Display a compact overview or full diagnostics in Rich panels.

    :param output_width: Width used for Rich terminal rendering.
    :param verbose: Whether to include detailed diagnostics and defaults.
    """
    with _configured_cli_output(as_json=False, verbose=False, output_width=output_width):
        console = _output.console
        runtime_versions = get_semantic_runtime_versions()
        cli_module.configure_mps_environment(DEFAULT_SEMANTIC_DEVICE, fallback=None)
        diagnostics = cli_module.get_device_diagnostics(DEFAULT_SEMANTIC_DEVICE)
        if not verbose:
            rows = [
                ("Python", runtime_versions["python"]),
                ("PyTorch", runtime_versions["torch"]),
                ("Device", diagnostics.resolved or "unavailable"),
                ("Default model", DEFAULT_MODEL),
                ("Supported languages", ", ".join(SUPPORTED_LANGUAGES)),
            ]
            if diagnostics.error is not None:
                rows.append(("Device diagnostic error", diagnostics.error))
            console.print(_settings_panel(f"codedupes {__version__}", rows))
            console.print(
                "Run codedupes info --verbose for full diagnostics and defaults.", style="dim"
            )
            return

        default_profile = resolve_model_profile(DEFAULT_MODEL)
        console.print(
            _settings_panel(
                f"codedupes {__version__}",
                [
                    ("Python", runtime_versions["python"]),
                    ("Platform", platform.platform()),
                    ("PyTorch", runtime_versions["torch"]),
                    ("Transformers", runtime_versions["transformers"]),
                    ("Sentence Transformers", runtime_versions["sentence-transformers"]),
                ],
            )
        )
        if cpu_bf16_opted_in():
            cpu_bf16_policy = (
                "enabled (experimental)"
                if diagnostics.cpu_bf16_native
                else "disabled (CODEDUPES_CPU_BF16=1 set, but the capability gate failed)"
            )
        else:
            cpu_bf16_policy = (
                "disabled (experimental; set CODEDUPES_CPU_BF16=1 on gate-capable hardware)"
            )
        device_rows = [
            ("Default semantic device request", DEFAULT_SEMANTIC_DEVICE),
            ("Resolved semantic device", diagnostics.resolved or "unavailable"),
            ("CUDA available", str(diagnostics.cuda_available)),
            ("MPS built/available", f"{diagnostics.mps_built}/{diagnostics.mps_available}"),
            (
                "MPS fallback environment",
                (
                    f"PYTORCH_ENABLE_MPS_FALLBACK={diagnostics.mps_fallback_env} "
                    f"(torch reads this as: {describe_mps_fallback_env(diagnostics.mps_fallback_env)})"
                ),
            ),
            (
                "MLX loaded in process",
                f"{diagnostics.mlx_loaded} (MLX allocator is not managed by codedupes)",
            ),
            (
                "CPU",
                f"{diagnostics.cpu_name or 'unknown'} ({diagnostics.cpu_architecture or 'unknown'})",
            ),
            (
                "CPU bfloat16 GEMM capable",
                (
                    f"{diagnostics.cpu_bf16_native} "
                    f"(native bf16 ISA={diagnostics.cpu_bf16_isa}, "
                    f"mkldnn available={diagnostics.cpu_mkldnn_available})"
                ),
            ),
            ("CPU bfloat16 inference", cpu_bf16_policy),
        ]
        if diagnostics.mps_memory_bytes:
            device_rows.append(
                ("MPS memory", format_mps_memory_snapshot(diagnostics.mps_memory_bytes))
            )
        if diagnostics.error is not None:
            device_rows.append(("Device diagnostic error", diagnostics.error))
        console.print(_settings_panel("Device", device_rows))
        console.print(
            _settings_panel(
                "Analysis defaults",
                [
                    ("Default model", DEFAULT_MODEL),
                    ("Default model revision", default_profile.default_revision or "auto"),
                    (
                        f"Semantic duplicate gates ({DEFAULT_MODEL})",
                        _format_language_gates(default_profile),
                    ),
                    ("Default traditional threshold", DEFAULT_TRADITIONAL_THRESHOLD),
                    ("Default semantic task for check", DEFAULT_CHECK_SEMANTIC_TASK),
                    ("Default semantic task for search", DEFAULT_SEARCH_SEMANTIC_TASK),
                    ("Default min_statements for semantic", DEFAULT_MIN_SEMANTIC_STATEMENTS),
                    ("Default output width", DEFAULT_OUTPUT_WIDTH),
                    ("Default combined semantic fallback", "disabled"),
                    ("Supported languages", ", ".join(SUPPORTED_LANGUAGES)),
                    ("Unused-code analysis languages", "python"),
                ],
            )
        )
        grammar_rows = []
        for status in get_grammar_statuses():
            installed = status.installed_version or "not installed"
            state = "ready" if status.available else "unavailable"
            detail = f"{status.package}=={status.pinned_version} (installed={installed}, {state})"
            if status.error:
                detail += f"\n{status.error}"
            grammar_rows.append((status.dialect, detail))
        console.print(_settings_panel("Tree-sitter grammar packages", grammar_rows))
        console.print(
            _settings_panel(
                "Default exclusions",
                [
                    ("Default built-in exclude globs", "\n".join(DEFAULT_EXCLUDE_PATTERNS)),
                    (
                        f"Default excluded directory names ({len(DEFAULT_EXCLUDE_DIR_NAMES)} total)",
                        ", ".join(sorted(DEFAULT_EXCLUDE_DIR_NAMES)),
                    ),
                ],
            )
        )
        console.print("Built-in semantic model aliases", style="bold cyan")
        for profile in list_supported_models():
            model_rows = [
                ("Model", profile.canonical_name),
                ("Family", profile.family),
                ("Search threshold", str(get_default_search_threshold(profile.key))),
                ("Semantic duplicate gates", _format_language_gates(profile)),
                ("Aliases", ", ".join(profile.all_aliases())),
            ]
            if profile.default_revision is not None:
                model_rows.append(("Default revision", profile.default_revision))
            model_rows.append(("Default trust remote code", str(profile.default_trust_remote_code)))
            console.print(_settings_panel(profile.key, model_rows))
        try:
            console.print(_cache_summary_panel(cli_module.EmbeddingCache().stats()))
        except Exception as exc:  # noqa: BLE001 - info is diagnostics; report and keep printing
            console.print(_settings_panel("Embedding cache", [("Unavailable", str(exc))]))
        console.print("Run with --help for CLI usage", style="dim")
