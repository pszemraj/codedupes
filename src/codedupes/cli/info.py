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

from ._output import DEFAULT_OUTPUT_WIDTH
from .cache import _echo_cache_summary


def _format_language_gates(profile: SemanticModelProfile) -> str:
    """Format one model profile's per-language semantic duplicate gates."""
    gates = ", ".join(
        f"{language}={gate}" for language, gate in profile.language_semantic_thresholds.items()
    )
    fallback = f"fallback={profile.default_semantic_threshold}"
    return f"{gates} ({fallback})" if gates else fallback


@cli_module.cli.command("info", help="Print tool and model defaults")
def info_command() -> None:
    """Print version and default settings."""
    default_profile = resolve_model_profile(DEFAULT_MODEL)
    click.echo(f"codedupes {__version__}")
    runtime_versions = get_semantic_runtime_versions()
    click.echo(f"Python: {runtime_versions['python']}")
    click.echo(f"Platform: {platform.platform()}")
    click.echo(f"PyTorch: {runtime_versions['torch']}")
    click.echo(f"Transformers: {runtime_versions['transformers']}")
    click.echo(f"Sentence Transformers: {runtime_versions['sentence-transformers']}")
    cli_module.configure_mps_environment(DEFAULT_SEMANTIC_DEVICE, fallback=None)
    diagnostics = cli_module.get_device_diagnostics(DEFAULT_SEMANTIC_DEVICE)
    click.echo(f"Default semantic device request: {DEFAULT_SEMANTIC_DEVICE}")
    click.echo(f"Resolved semantic device: {diagnostics.resolved or 'unavailable'}")
    click.echo(f"CUDA available: {diagnostics.cuda_available}")
    click.echo(f"MPS built/available: {diagnostics.mps_built}/{diagnostics.mps_available}")
    click.echo(
        f"PYTORCH_ENABLE_MPS_FALLBACK: {diagnostics.mps_fallback_env} "
        f"(torch reads this as: {describe_mps_fallback_env(diagnostics.mps_fallback_env)})"
    )
    click.echo(
        f"MLX loaded in process: {diagnostics.mlx_loaded} "
        "(MLX allocator is not managed by codedupes)"
    )
    click.echo(
        f"CPU: {diagnostics.cpu_name or 'unknown'} ({diagnostics.cpu_architecture or 'unknown'})"
    )
    click.echo(
        f"CPU bfloat16 GEMM capable: {diagnostics.cpu_bf16_native} "
        f"(native bf16 ISA={diagnostics.cpu_bf16_isa}, "
        f"mkldnn available={diagnostics.cpu_mkldnn_available})"
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
    click.echo(f"CPU bfloat16 inference: {cpu_bf16_policy}")
    if diagnostics.mps_memory_bytes:
        click.echo(f"MPS memory: {format_mps_memory_snapshot(diagnostics.mps_memory_bytes)}")
    if diagnostics.error is not None:
        click.echo(f"Device diagnostic error: {diagnostics.error}")
    click.echo(f"Default model: {DEFAULT_MODEL}")
    click.echo(f"Default model revision: {default_profile.default_revision or 'auto'}")
    click.echo(
        f"Semantic duplicate gates ({DEFAULT_MODEL}): {_format_language_gates(default_profile)}"
    )
    click.echo(f"Default traditional threshold: {DEFAULT_TRADITIONAL_THRESHOLD}")
    click.echo(f"Default semantic task for check: {DEFAULT_CHECK_SEMANTIC_TASK}")
    click.echo(f"Default semantic task for search: {DEFAULT_SEARCH_SEMANTIC_TASK}")
    click.echo(f"Default min_statements for semantic: {DEFAULT_MIN_SEMANTIC_STATEMENTS}")
    click.echo(f"Default output width: {DEFAULT_OUTPUT_WIDTH}")
    click.echo("Default combined semantic fallback: disabled")
    click.echo(f"Supported languages: {', '.join(SUPPORTED_LANGUAGES)}")
    click.echo("Unused-code analysis languages: python")
    click.echo("Tree-sitter grammar packages:")
    for status in get_grammar_statuses():
        installed = status.installed_version or "not installed"
        state = "ready" if status.available else "unavailable"
        click.echo(
            f"  - {status.dialect}: {status.package}=={status.pinned_version} "
            f"(installed={installed}, {state})"
        )
        if status.error:
            click.echo(f"      {status.error}")
    click.echo("Default built-in exclude globs:")
    for pattern in DEFAULT_EXCLUDE_PATTERNS:
        click.echo(f"  - {pattern}")
    click.echo(f"Default excluded directory names ({len(DEFAULT_EXCLUDE_DIR_NAMES)} total):")
    click.echo(f"  {', '.join(sorted(DEFAULT_EXCLUDE_DIR_NAMES))}")
    click.echo("Built-in semantic model aliases:")
    for profile in list_supported_models():
        aliases = ", ".join(profile.all_aliases())
        search_threshold = get_default_search_threshold(profile.key)
        click.echo(f"  - {profile.key} -> {profile.canonical_name}")
        click.echo(f"      family={profile.family} search_threshold={search_threshold}")
        click.echo(f"      semantic duplicate gates: {_format_language_gates(profile)}")
        click.echo(f"      aliases: {aliases}")
        if profile.default_revision is not None:
            click.echo(f"      default_revision: {profile.default_revision}")
        click.echo(f"      default_trust_remote_code: {profile.default_trust_remote_code}")
    click.echo("Embedding cache:")
    try:
        _echo_cache_summary(cli_module.EmbeddingCache().stats())
    except Exception as exc:  # noqa: BLE001 - info is diagnostics; report and keep printing
        click.echo(f"  unavailable: {exc}")
    click.echo("Run with --help for CLI usage")
