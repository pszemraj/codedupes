"""Per-language semantic gates, threshold profiles, and the profile hybrid split."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest

import codedupes.semantic as semantic_module
from codedupes import analyzer as analyzer_module
from codedupes.analyzer import AnalyzerConfig, CodeAnalyzer, analyze_directory
from codedupes.constants import HYBRID_STATEMENT_RATIO_MIN, HYBRID_WEAK_JACCARD_MIN
from codedupes.models import CodeUnit, DuplicatePair
from codedupes.semantic_profiles import SemanticModelProfile, resolve_model_profile
from tests.analyzer_helpers import make_semantic_runner
from tests.conftest import create_project, make_code_unit


def _profile_with_gates(gates: dict[str, float], fallback: float = 0.99) -> SemanticModelProfile:
    """Build a minimal profile carrying the given per-language duplicate gates.

    :param gates: Language-to-gate map for the fake profile.
    :param fallback: Gate for languages absent from ``gates``.
    :return: Frozen profile suitable for monkeypatching ``resolve_model_profile``.
    """
    return SemanticModelProfile(
        key="test-profile",
        canonical_name="test/profile",
        aliases=(),
        family="generic",
        default_semantic_threshold=fallback,
        language_semantic_thresholds=gates,
    )


def test_analyzer_resolves_per_language_semantic_gate(tmp_path: Path, monkeypatch) -> None:
    source = "def add_one(x):\n    return x + 1\n"
    project = create_project(tmp_path, source)
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        analyzer_module,
        "resolve_model_profile",
        lambda _model: _profile_with_gates({"python": 0.77}),
    )
    monkeypatch.setattr(
        analyzer_module,
        "run_semantic_analysis",
        make_semantic_runner(capture=captured),
    )

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=False,
            run_semantic=True,
            run_unused=False,
            min_semantic_statements=0,
            semantic_threshold=None,
        )
    )
    analyzer.analyze(project)

    assert captured["threshold"] == 0.77


@pytest.mark.parametrize(
    ("model_kind", "choice", "numeric", "expected"),
    [
        ("local", "auto", None, 0.74),
        ("builtin", "auto", None, 0.74),
        ("default", "auto", None, 0.87),
        ("hub", "auto", None, 0.74),
        # Explicit profiles override the model family; test each choice once.
        ("builtin", "generic", None, 0.82),
        ("default", "embeddinggemma-300m", None, 0.74),
        ("builtin", "gte-modernbert-base", None, 0.87),
        # Numeric gates bypass profile resolution for every profile choice.
        ("local", "auto", 0.91, 0.91),
        ("hub", "generic", 0.91, 0.91),
        ("builtin", "embeddinggemma-300m", 0.91, 0.91),
        ("default", "gte-modernbert-base", 0.91, 0.91),
    ],
)
def test_analyze_directory_threshold_profiles(
    tmp_path, monkeypatch, caplog, choice, expected, numeric, model_kind
) -> None:
    model_dir = tmp_path / "approved-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        '{"model_type": "gemma3_text", "use_bidirectional_attention": true}', encoding="utf-8"
    )
    model_name = {
        "local": str(model_dir),
        "builtin": "embeddinggemma-300m",
        "default": analyzer_module.DEFAULT_MODEL,
        "hub": "someone/embeddinggemma-300m-code-ft",
    }[model_kind]
    project = create_project(tmp_path, "def alpha(x):\n    return x + 1\n")
    captured = {}
    monkeypatch.setattr(
        analyzer_module, "run_semantic_analysis", make_semantic_runner(capture=captured)
    )
    with caplog.at_level("INFO"):
        for _ in range(2):
            analyze_directory(
                project,
                model_name=model_name,
                threshold_profile=choice,
                semantic_threshold=numeric,
                min_semantic_statements=0,
                run_unused=False,
            )
    assert captured["language_thresholds"] == {"python": expected}
    assert captured["model_name"] == model_name
    assert captured["revision"] is None
    if numeric is not None:
        assert "explicit numeric override" in caplog.text
        assert "family duplicate thresholds" not in caplog.text
    else:
        assert f"threshold-profile={choice}" in caplog.text
    automatic_copy = choice == "auto" and numeric is None and model_kind in {"local", "hub"}
    assert caplog.text.count("Use --threshold-profile generic") == int(automatic_copy)
    assert caplog.text.count("score distribution may differ") == int(
        automatic_copy and model_kind == "hub"
    )


def _create_two_language_project(tmp_path: Path) -> Path:
    """Write a small mixed Python/JavaScript project for gate tests.

    :param tmp_path: Test-scoped temporary directory.
    :return: Project root containing one ``.py`` and one ``.js`` module.
    """
    project = tmp_path / "polyglot_project"
    project.mkdir()
    (project / "alpha.py").write_text(
        "def alpha_one(x):\n    y = x + 1\n    return y\n\n"
        "def alpha_two(x):\n    z = x + 2\n    return z\n"
    )
    (project / "beta.js").write_text(
        "function betaOne(x) {\n  const y = x + 1;\n  return y;\n}\n\n"
        "function betaTwo(x) {\n  const z = x + 2;\n  return z;\n}\n"
    )
    return project


@pytest.mark.grammar
def test_semantic_pairs_are_gated_per_language(tmp_path: Path, monkeypatch) -> None:
    project = _create_two_language_project(tmp_path)
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        analyzer_module,
        "resolve_model_profile",
        lambda _model: _profile_with_gates({"python": 0.90, "javascript": 0.60}),
    )
    monkeypatch.setattr(
        analyzer_module,
        "run_semantic_analysis",
        make_semantic_runner(capture=captured),
    )

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=False,
            run_semantic=True,
            run_unused=False,
            min_semantic_statements=0,
            progress="never",
        )
    )
    analyzer.analyze(project)

    # Every language group is scanned at its own gate; the scalar floor only
    # covers languages without a calibrated entry.
    assert captured["language_thresholds"] == {"python": 0.90, "javascript": 0.60}
    assert captured["progress"] == "never"
    assert captured["threshold"] == 0.60


class _FixedVectorModel:
    """Model stub returning a fixed vector per marker found in the input text."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def encode(self, texts, **_kwargs):
        rows = []
        for text in texts:
            marker = next(name for name in self.vectors if name in text)
            rows.append(self.vectors[marker])
        return np.array(rows, dtype=np.float32)


def _two_language_vectors() -> dict[str, list[float]]:
    """Build vectors whose same-language pairs sit at cosine 0.75.

    :return: Marker-to-vector map for the two-language fixture project.
    """
    off = float(np.sqrt(1.0 - 0.75**2))
    return {
        "alpha_one": [1.0, 0.0, 0.0, 0.0],
        "alpha_two": [0.75, off, 0.0, 0.0],
        "betaOne": [0.0, 0.0, 1.0, 0.0],
        "betaTwo": [0.0, 0.0, 0.75, off],
    }


@pytest.mark.grammar
def test_per_language_gates_survive_the_whole_semantic_pipeline(
    tmp_path: Path, monkeypatch
) -> None:
    project = _create_two_language_project(tmp_path)
    monkeypatch.setattr(
        analyzer_module,
        "resolve_model_profile",
        lambda _model: _profile_with_gates({"python": 0.90, "javascript": 0.60}),
    )
    monkeypatch.setattr(
        semantic_module,
        "get_model",
        lambda *args, **kwargs: _FixedVectorModel(_two_language_vectors()),
    )

    result = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=False,
            run_semantic=True,
            run_unused=False,
            min_semantic_statements=0,
            embedding_cache=False,
        )
    ).analyze(project)

    # 0.75 clears javascript's 0.60 gate but not python's 0.90.
    assert [
        (duplicate.unit_a.name, duplicate.unit_b.name) for duplicate in result.semantic_duplicates
    ] == [("betaOne", "betaTwo")]


@pytest.mark.grammar
@pytest.mark.parametrize(
    ("threshold_profile", "score", "accepted"),
    [
        ("auto", 0.70, True),
        ("generic", 0.75, False),
        ("gte-modernbert-base", 0.75, True),
        ("embeddinggemma-300m", 0.77, True),
    ],
)
def test_cross_language_pairs_require_opt_in_and_use_looser_gate(
    tmp_path: Path, monkeypatch, threshold_profile, score, accepted
) -> None:
    project = _create_two_language_project(tmp_path)
    vectors = _two_language_vectors()
    # Named profiles use a score between their Python and JavaScript gates.
    vectors["alpha_one"] = [1.0, 0.0, 0.0, 0.0]
    vectors["betaOne"] = [score, 0.0, float(np.sqrt(1.0 - score**2)), 0.0]

    monkeypatch.setattr(
        analyzer_module,
        "resolve_model_profile",
        lambda _model: _profile_with_gates({"python": 0.90, "javascript": 0.60}),
    )
    monkeypatch.setattr(
        semantic_module,
        "get_model",
        lambda *args, **kwargs: _FixedVectorModel(vectors),
    )

    base_config = {
        "run_traditional": False,
        "run_semantic": True,
        "run_unused": False,
        "min_semantic_statements": 0,
        "embedding_cache": False,
        "threshold_profile": threshold_profile,
    }
    default_result = CodeAnalyzer(AnalyzerConfig(**base_config)).analyze(project)
    assert all(
        duplicate.unit_a.language == duplicate.unit_b.language
        for duplicate in default_result.semantic_duplicates
    )

    # The selected profile supplies both language gates to the mixed scan.
    opted_result = CodeAnalyzer(AnalyzerConfig(cross_language=True, **base_config)).analyze(project)
    pairs = {
        frozenset({duplicate.unit_a.name, duplicate.unit_b.name})
        for duplicate in opted_result.semantic_duplicates
    }
    assert (frozenset({"alpha_one", "betaOne"}) in pairs) is accepted
    numeric_result = CodeAnalyzer(
        AnalyzerConfig(cross_language=True, semantic_threshold=0.78, **base_config)
    ).analyze(project)
    assert frozenset({"alpha_one", "betaOne"}) not in {
        frozenset({duplicate.unit_a.name, duplicate.unit_b.name})
        for duplicate in numeric_result.semantic_duplicates
    }


@pytest.mark.grammar
def test_explicit_semantic_threshold_applies_flat_across_languages(
    tmp_path: Path, monkeypatch
) -> None:
    project = _create_two_language_project(tmp_path)
    captured: dict[str, object] = {}

    def paired_duplicates(units: list[CodeUnit]) -> list[DuplicatePair]:
        by_name = {unit.name: unit for unit in units}
        return [
            DuplicatePair(
                unit_a=by_name["alpha_one"],
                unit_b=by_name["alpha_two"],
                similarity=0.75,
                method="semantic",
            )
        ]

    monkeypatch.setattr(
        analyzer_module,
        "run_semantic_analysis",
        make_semantic_runner(capture=captured, duplicate_factory=paired_duplicates),
    )

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=False,
            run_semantic=True,
            run_unused=False,
            min_semantic_statements=0,
            semantic_threshold=0.70,
        )
    )
    result = analyzer.analyze(project)

    # An explicit flat threshold bypasses per-language gate resolution: the
    # scan uses 0.70 for every language rather than a profile lookup. The run
    # record still resolves the model profile for provenance (its `profile`
    # field), which is a separate, informational use of the same function.
    assert captured["threshold"] == 0.70
    assert len(result.semantic_duplicates) == 1


@pytest.mark.parametrize(
    ("model_name", "expected_tiers"),
    [
        # Below GTE's Python promotion gate, identifier corroboration is required.
        (
            "gte-modernbert-base",
            {"same_size": "semantic_review", "lopsided": "semantic_review"},
        ),
        # Gemma's Python promotion gate reports both high-similarity pairs.
        (
            "embeddinggemma-300m",
            {"same_size": "semantic_high_confidence", "lopsided": "semantic_high_confidence"},
        ),
    ],
)
def test_analyzer_applies_the_profile_hybrid_split(
    tmp_path: Path, monkeypatch, model_name: str, expected_tiers: dict[str, str]
) -> None:
    """The analyzer must split semantic-only pairs with the profile's calibrated constants."""
    review_band_score = 0.88
    profile = resolve_model_profile(model_name)
    assert review_band_score >= profile.semantic_threshold_for_language("python")
    source = dedent(
        """
        def collect_total(records):
            accepted = [record for record in records if record.enabled]
            amount = sum(record.value for record in accepted)
            return amount

        def measure_sum(entries):
            chosen = [entry for entry in entries if entry.ready]
            result = sum(entry.weight for entry in chosen)
            return result

        def tiny(v):
            return v
        """
    ).strip()
    project = create_project(tmp_path, source)

    def paired(units: list[CodeUnit]) -> list[DuplicatePair]:
        by_name = {unit.name: unit for unit in units}
        return [
            DuplicatePair(
                by_name["collect_total"],
                by_name["measure_sum"],
                review_band_score,
                "semantic",
            ),
            DuplicatePair(by_name["collect_total"], by_name["tiny"], review_band_score, "semantic"),
        ]

    monkeypatch.setattr(
        analyzer_module, "run_semantic_analysis", make_semantic_runner(duplicate_factory=paired)
    )
    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=True,
            run_semantic=True,
            run_unused=False,
            min_semantic_statements=0,
            filter_tiny_traditional=False,
            model_name=model_name,
        )
    )

    result = analyzer.analyze(project)

    tiers = {
        ("same_size" if pair.unit_b.name == "measure_sum" else "lopsided"): pair.tier
        for pair in result.hybrid_duplicates
    }
    assert tiers == expected_tiers
    assert all(pair.weak_identifier_jaccard == 0.0 for pair in result.hybrid_duplicates)


def test_resolve_hybrid_split_gates_off_with_explicit_semantic_threshold(tmp_path: Path) -> None:
    """An explicit ``semantic_threshold`` keeps the profile constants but disables its gates."""
    ts_unit = make_code_unit(tmp_path, name="ts_unit", source="function f() {}\n")
    ts_unit.language = "typescript"
    py_unit = make_code_unit(tmp_path, name="py_unit", source="def f():\n    pass\n")
    py_unit.language = "python"
    units = [ts_unit, py_unit]

    profile = resolve_model_profile("gte-modernbert-base")

    default_analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=True,
            run_semantic=True,
            run_unused=False,
            model_name="gte-modernbert-base",
        )
    )
    weak_min, ratio_min, gates = default_analyzer._resolve_hybrid_split(units)
    assert gates == {"python": 0.90}
    assert weak_min == profile.hybrid_weak_identifier_jaccard_min
    assert ratio_min == profile.hybrid_statement_ratio_min

    flat_threshold_analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=True,
            run_semantic=True,
            run_unused=False,
            model_name="gte-modernbert-base",
            semantic_threshold=0.80,
        )
    )
    flat_weak_min, flat_ratio_min, flat_gates = flat_threshold_analyzer._resolve_hybrid_split(units)
    assert flat_gates == {}
    assert flat_weak_min == profile.hybrid_weak_identifier_jaccard_min
    assert flat_ratio_min == profile.hybrid_statement_ratio_min

    generic_analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=True,
            run_semantic=True,
            run_unused=False,
            model_name="gte-modernbert-base",
            threshold_profile="generic",
        )
    )
    generic_weak_min, generic_ratio_min, generic_gates = generic_analyzer._resolve_hybrid_split(
        units
    )
    assert generic_gates == {}
    assert generic_weak_min == HYBRID_WEAK_JACCARD_MIN
    assert generic_ratio_min == HYBRID_STATEMENT_RATIO_MIN


def test_analyzer_gemma_python_promotion_gate_requires_no_corroboration(
    tmp_path: Path, monkeypatch
) -> None:
    """Gemma's Python promotion gate works without identifier corroboration."""
    source = dedent(
        """
        def alpha(value):
            incremented = value + 1
            return incremented

        def omega(entry):
            doubled = entry * 2
            tripled = doubled + entry
            scaled = tripled - 1
            adjusted = scaled + 3
            return adjusted
        """
    ).strip()
    project = create_project(tmp_path, source)

    def paired(units: list[CodeUnit]) -> list[DuplicatePair]:
        by_name = {unit.name: unit for unit in units}
        return [
            DuplicatePair(by_name["alpha"], by_name["omega"], 0.90, "semantic"),
        ]

    monkeypatch.setattr(
        analyzer_module, "run_semantic_analysis", make_semantic_runner(duplicate_factory=paired)
    )

    default_analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=True,
            run_semantic=True,
            run_unused=False,
            min_semantic_statements=0,
            filter_tiny_traditional=False,
            model_name="embeddinggemma-300m",
        )
    )
    default_result = default_analyzer.analyze(project)

    [default_pair] = default_result.hybrid_duplicates
    assert default_pair.weak_identifier_jaccard < 0.30
    assert default_pair.tier == "semantic_high_confidence"

    gated_off_analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=True,
            run_semantic=True,
            run_unused=False,
            min_semantic_statements=0,
            filter_tiny_traditional=False,
            model_name="embeddinggemma-300m",
            semantic_threshold=0.74,
        )
    )
    gated_off_result = gated_off_analyzer.analyze(project)

    [gated_off_pair] = gated_off_result.hybrid_duplicates
    assert gated_off_pair.tier == "semantic_review"
