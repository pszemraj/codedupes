"""``AnalyzerConfig`` validation, normalization, and mode dependencies."""

from __future__ import annotations

from pathlib import Path

import pytest

import codedupes.semantic as semantic_module
from codedupes import analyzer as analyzer_module
from codedupes.analyzer import AnalyzerConfig, CodeAnalyzer, analyze_directory
from codedupes.semantic import SemanticBackendError
from tests.analyzer_helpers import make_semantic_runner
from tests.conftest import create_project


def test_analyze_directory_uses_auto_revision_for_custom_model(tmp_path: Path, monkeypatch) -> None:
    source = "def add_one(x):\n    return x + 1\n"
    project = create_project(tmp_path, source)
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        analyzer_module,
        "run_semantic_analysis",
        make_semantic_runner(capture=captured),
    )

    analyze_directory(
        project,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        min_semantic_statements=0,
        run_unused=False,
    )

    assert captured["model_name"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert captured["revision"] is None


def test_analyzer_rejects_invalid_or_disabled_threshold_profile() -> None:
    with pytest.raises(ValueError, match="threshold_profile must be one of"):
        AnalyzerConfig(threshold_profile="invalid")
    with pytest.raises(ValueError, match="threshold_profile"):
        AnalyzerConfig(run_semantic=False, threshold_profile="generic")


def test_allow_semantic_fallback_requires_combined_mode() -> None:
    with pytest.raises(
        ValueError,
        match="allow_semantic_fallback requires run_semantic=True and run_traditional=True",
    ):
        AnalyzerConfig(
            run_traditional=False,
            run_semantic=True,
            allow_semantic_fallback=True,
        )


@pytest.mark.parametrize(
    ("config_overrides", "threshold_profile"),
    [
        ({"semantic_task": "classification"}, "auto"),
        ({"instruction_prefix": "CUSTOM: "}, "auto"),
        ({"model_revision": "f" * 40}, "auto"),
        ({"trust_remote_code": True}, "auto"),
        # Selecting another threshold profile cannot bypass the context guard.
        ({"instruction_prefix": "CUSTOM: "}, "generic"),
        ({"model_revision": "f" * 40}, "embeddinggemma-300m"),
    ],
)
def test_uncalibrated_duplicate_context_rejected_at_construction(
    config_overrides: dict[str, str],
    threshold_profile: str,
) -> None:
    with pytest.raises(ValueError, match="provide semantic_threshold explicitly"):
        AnalyzerConfig(
            run_traditional=False,
            run_semantic=True,
            run_unused=False,
            min_semantic_statements=0,
            model_name="embeddinggemma-300m",
            threshold_profile=threshold_profile,
            **config_overrides,
        )


@pytest.mark.parametrize(
    "config_overrides",
    [
        {"semantic_task": "classification"},
        {"instruction_prefix": "CUSTOM: "},
        {"model_revision": "f" * 40},
        {"trust_remote_code": True},
    ],
)
def test_search_mode_defers_uncalibrated_context_to_query_time(
    config_overrides: dict[str, str],
) -> None:
    config = AnalyzerConfig(
        mode="search",
        run_traditional=False,
        run_semantic=True,
        run_unused=False,
        min_semantic_statements=0,
        model_name="embeddinggemma-300m",
        **config_overrides,
    )
    assert config.mode == "search"


def test_analyze_rejects_search_mode_config(tmp_path: Path) -> None:
    project = create_project(tmp_path, "def entry(x):\n    return x + 1\n")
    analyzer = CodeAnalyzer(AnalyzerConfig(mode="search", run_traditional=False, run_unused=False))

    with pytest.raises(ValueError, match="mode='check'"):
        analyzer.analyze(project)


def test_invalid_mode_rejected() -> None:
    with pytest.raises(ValueError, match="mode must be"):
        AnalyzerConfig(mode="banana")


def test_search_mode_requires_semantic() -> None:
    with pytest.raises(ValueError, match="requires run_semantic=True"):
        AnalyzerConfig(mode="search", run_semantic=False)


def test_search_mode_accepts_negative_floor_but_check_mode_keeps_duplicate_bounds() -> None:
    search_config = AnalyzerConfig(
        mode="search",
        run_traditional=False,
        run_unused=False,
        semantic_threshold=-0.5,
    )

    assert search_config.semantic_threshold == -0.5
    with pytest.raises(ValueError, match=r"semantic_threshold must be in \[0.0, 1.0\]"):
        AnalyzerConfig(semantic_threshold=-0.5)
    with pytest.raises(ValueError, match="semantic_threshold must be finite"):
        AnalyzerConfig(
            mode="search",
            run_traditional=False,
            run_unused=False,
            semantic_threshold=float("nan"),
        )


def test_configuration_validation() -> None:
    for progress in ("auto", "always", "never"):
        assert AnalyzerConfig(progress=progress).progress == progress

    with pytest.raises(ValueError, match="progress must be 'auto', 'always', or 'never'"):
        AnalyzerConfig(progress="nevre")

    with pytest.raises(ValueError, match="jaccard_threshold"):
        AnalyzerConfig(jaccard_threshold=1.5)

    with pytest.raises(ValueError, match="semantic_threshold"):
        AnalyzerConfig(semantic_threshold=-0.1)

    with pytest.raises(ValueError, match="semantic_unit_types"):
        AnalyzerConfig(semantic_unit_types=())

    with pytest.raises(ValueError, match="Invalid semantic_unit_types"):
        AnalyzerConfig(semantic_unit_types=("invalid",))

    with pytest.raises(ValueError, match="Invalid semantic_task"):
        AnalyzerConfig(semantic_task="not-a-task")

    with pytest.raises(ValueError, match="tiny_unit_statement_cutoff"):
        AnalyzerConfig(tiny_unit_statement_cutoff=-1)


def test_analyzer_config_rejects_every_check_disabled() -> None:
    with pytest.raises(
        ValueError,
        match="At least one of run_traditional, run_semantic, or run_unused must be True",
    ):
        AnalyzerConfig(run_traditional=False, run_semantic=False, run_unused=False)


def test_analyzer_config_rejects_semantic_candidate_controls_without_semantic_mode() -> None:
    with pytest.raises(ValueError, match="model_name.*require run_semantic=True"):
        AnalyzerConfig(run_semantic=False, model_name="embeddinggemma-300m")

    with pytest.raises(ValueError, match="min_semantic_statements.*require run_semantic=True"):
        AnalyzerConfig(run_semantic=False, min_semantic_statements=5)

    with pytest.raises(ValueError, match="semantic_unit_types.*require run_semantic=True"):
        AnalyzerConfig(run_semantic=False, semantic_unit_types=("class",))


def test_invalid_mode_dependency_raises() -> None:
    with pytest.raises(ValueError, match="strict_unused requires run_unused=True"):
        AnalyzerConfig(run_unused=False, strict_unused=True)

    with pytest.raises(ValueError, match="require run_semantic=True"):
        AnalyzerConfig(run_semantic=False, semantic_task="classification")

    with pytest.raises(ValueError, match="require run_semantic=True"):
        AnalyzerConfig(run_semantic=False, model_revision="abc123")

    config = AnalyzerConfig(run_semantic=False, embedding_cache=False)
    assert config.embedding_cache is False

    with pytest.raises(ValueError, match="require run_traditional=True"):
        AnalyzerConfig(run_traditional=False, tiny_unit_statement_cutoff=5)


def test_empty_extraction_still_validates_explicit_device(tmp_path: Path, monkeypatch) -> None:
    """An empty corpus must not turn an unavailable explicit device into a success.

    Empty analysis resolves semantic cache identity so it can publish a final
    manifest, but it must still enforce device policy without loading a model.
    """

    def _raise_unavailable(*_args, **_kwargs):
        raise SemanticBackendError("mps is not available in this environment")

    def _fail_if_called(*_args, **_kwargs):
        raise AssertionError("an empty corpus must not load a model")

    monkeypatch.setattr(semantic_module, "_resolve_semantic_device_request", _raise_unavailable)
    monkeypatch.setattr(semantic_module, "get_model", _fail_if_called)
    empty_project = tmp_path / "empty"
    empty_project.mkdir()

    with pytest.raises(RuntimeError, match="Semantic analysis failed in combined mode"):
        CodeAnalyzer(AnalyzerConfig(device="mps")).analyze(empty_project)
    with pytest.raises(SemanticBackendError, match="mps is not available"):
        CodeAnalyzer(
            AnalyzerConfig(
                device="mps",
                run_traditional=False,
                run_unused=False,
            )
        ).analyze(empty_project)

    # Opting into combined-mode semantic fallback degrades instead of raising.
    fallback_result = CodeAnalyzer(
        AnalyzerConfig(device="mps", allow_semantic_fallback=True)
    ).analyze(empty_project)
    assert fallback_result.units == []

    # A device that always has a CPU path resolves the empty manifest without
    # selecting a runtime device or loading the model.
    monkeypatch.setattr(semantic_module, "_resolve_semantic_device_request", _fail_if_called)
    cpu_result = CodeAnalyzer(AnalyzerConfig(device="cpu")).analyze(empty_project)
    assert cpu_result.units == []
    assert cpu_result.embedding_stats is not None
    assert cpu_result.embedding_stats.model_loaded is False


def test_analyzer_config_normalizes_semantic_device_options() -> None:
    config = AnalyzerConfig(
        device=" MPS ",
        mps_fallback=False,
        mps_memory_fraction=0.8,
    )

    assert config.device == "mps"
    assert config.mps_fallback is False
    assert config.mps_memory_fraction == 0.8


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_analyzer_config_rejects_mps_memory_fraction_for_non_mps_devices(
    device: str,
) -> None:
    with pytest.raises(ValueError, match="requires device='mps' or device='auto'"):
        AnalyzerConfig(device=device, mps_memory_fraction=0.8)


@pytest.mark.parametrize("fraction", [0.0, -0.1, 2.1])
def test_analyzer_config_rejects_unsafe_mps_memory_fraction(fraction: float) -> None:
    with pytest.raises(ValueError, match=r"\(0.0, 2.0\]"):
        AnalyzerConfig(mps_memory_fraction=fraction)


def test_analyzer_config_rejects_device_controls_without_semantic_mode() -> None:
    with pytest.raises(ValueError, match="device.*require run_semantic=True"):
        AnalyzerConfig(run_semantic=False, device="mps")

    with pytest.raises(ValueError, match="mps_fallback.*require run_semantic=True"):
        AnalyzerConfig(run_semantic=False, mps_fallback=False)

    with pytest.raises(ValueError, match="mps_memory_fraction.*require run_semantic=True"):
        AnalyzerConfig(run_semantic=False, mps_memory_fraction=0.8)

    with pytest.raises(ValueError, match="strict_revision_cache.*require run_semantic=True"):
        AnalyzerConfig(run_semantic=False, strict_revision_cache=False)
