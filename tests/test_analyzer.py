from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest

import codedupes.semantic as semantic_module
from codedupes import analyzer as analyzer_module
from codedupes.analyzer import AnalyzerConfig, CodeAnalyzer, analyze_directory
from codedupes.models import AnalysisResult, CodeUnit, CodeUnitType, DuplicatePair
from codedupes.pairs import ordered_pair_key
from codedupes.semantic import SemanticBackendError
from codedupes.semantic_profiles import SemanticModelProfile
from tests.conftest import build_two_function_source, create_project, make_code_unit

_SEMANTIC_ANALYSIS_KWARG_NAMES = {
    "batch_size",
    "cache_scope",
    "cross_language",
    "device",
    "exclude_pairs",
    "instruction_prefix",
    "language_thresholds",
    "model_name",
    "mps_fallback",
    "mps_memory_fraction",
    "overflow_report",
    "revision",
    "semantic_task",
    "strict_revision_cache",
    "threshold",
    "trust_remote_code",
    "use_cache",
}
_QUERY_KWARG_NAMES = {
    "cache_scope",
    "corpus_identity",
    "device",
    "instruction_prefix",
    "model_name",
    "mps_fallback",
    "mps_memory_fraction",
    "revision",
    "semantic_task",
    "strict_revision_cache",
    "threshold",
    "top_k",
    "trust_remote_code",
    "use_cache",
}


def _embedding_identity_from_kwargs(kwargs: dict[str, object]):
    """Build the effective test identity for forwarded semantic arguments."""
    return semantic_module.resolve_embedding_space_identity(
        model_name=str(kwargs.get("model_name", analyzer_module.DEFAULT_MODEL)),
        instruction_prefix=kwargs.get("instruction_prefix"),
        revision=kwargs.get("revision"),
        trust_remote_code=kwargs.get("trust_remote_code"),
        semantic_task=kwargs.get("semantic_task"),
        device=str(kwargs.get("device", "cpu")),
        mps_fallback=kwargs.get("mps_fallback"),
        persist_local_model_manifest=False,
        strict_revision_cache=bool(kwargs.get("strict_revision_cache", False)),
    )


def _make_semantic_runner(
    *,
    duplicate_factory: Callable[[list[CodeUnit]], list[DuplicatePair]] | None = None,
    capture: dict[str, object] | None = None,
    capture_exclude_pairs: set[tuple[str, str]] | None = None,
    error: Exception | None = None,
) -> Callable[..., tuple[np.ndarray, list[DuplicatePair], object]]:
    """Build a reusable semantic-analysis test double."""

    def fake_run_semantic(units, **kwargs):
        assert set(kwargs) == _SEMANTIC_ANALYSIS_KWARG_NAMES
        if capture is not None:
            capture.update(kwargs)
        if capture_exclude_pairs is not None:
            capture_exclude_pairs.update(kwargs["exclude_pairs"] or set())
        if error is not None:
            raise error

        duplicates = duplicate_factory(units) if duplicate_factory is not None else []
        return (
            np.zeros((len(units), 2), dtype=np.float32),
            duplicates,
            _embedding_identity_from_kwargs(kwargs),
        )

    return fake_run_semantic


def _capture_query_runner(
    capture: dict[str, object],
) -> Callable[..., list[tuple[CodeUnit, float]]]:
    """Build a query runner that records and validates forwarded keyword arguments."""

    def fake_find_similar_to_query(query, units, embeddings, **kwargs):
        del query, units, embeddings
        assert set(kwargs) == _QUERY_KWARG_NAMES
        capture.update({f"query_{key}": value for key, value in kwargs.items()})
        return []

    return fake_find_similar_to_query


def _capture_semantic_unit_types(captured_types: list[CodeUnitType]):
    """Build a semantic runner that records unit types and returns no matches."""

    def fake_run_semantic(units, **_kwargs):
        captured_types.extend(unit.unit_type for unit in units)
        return (
            np.zeros((len(units), 2), dtype=np.float32),
            [],
            _embedding_identity_from_kwargs(_kwargs),
        )

    return fake_run_semantic


def _capture_traditional_units_runner(captured_units: list[CodeUnit]):
    """Build a traditional runner that records incoming units and returns no matches."""

    def fake_traditional(
        units,
        jaccard_threshold=0.85,
        compute_unused=True,
    ):
        captured_units.extend(units)
        return [], [], []

    return fake_traditional


def _traditional_single_jaccard_runner(similarity: float = 0.9):
    """Build a traditional runner returning one jaccard duplicate for first two units."""

    def fake_traditional(
        units,
        jaccard_threshold=0.85,
        compute_unused=True,
    ):
        first, second = units[:2]
        return (
            [DuplicatePair(unit_a=first, unit_b=second, similarity=similarity, method="jaccard")],
            [],
            [],
        )

    return fake_traditional


def test_all_duplicates_returns_raw_for_single_method_modes(tmp_path: Path) -> None:
    file_a = tmp_path / "a.py"
    file_b = tmp_path / "b.py"
    file_a.write_text("def foo():\n    return 1\n")
    file_b.write_text("def foo():\n    return 1\n")

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=True,
            run_semantic=False,
            run_unused=False,
            filter_tiny_traditional=False,
        )
    )
    result = analyzer.analyze(tmp_path)

    assert result.analysis_mode == "traditional"
    assert result.traditional_duplicates
    assert result.all_duplicates == result.traditional_duplicates

    unit = make_code_unit(
        tmp_path,
        name="bar",
        source="def bar():\n    return 1\n",
        lineno=1,
    )
    semantic_duplicate = DuplicatePair(
        unit_a=unit,
        unit_b=unit,
        similarity=0.95,
        method="semantic",
    )
    semantic_result = AnalysisResult(
        units=[unit],
        traditional_duplicates=[],
        semantic_duplicates=[semantic_duplicate],
        hybrid_duplicates=[],
        potentially_unused=[],
        analysis_mode="semantic",
    )

    assert semantic_result.all_duplicates == [semantic_duplicate]


@pytest.mark.parametrize(
    "analyzer_config, expected_unused",
    [
        (
            AnalyzerConfig(run_traditional=False, run_semantic=False, run_unused=True),
            set(),
        ),
        (
            AnalyzerConfig(
                run_traditional=True,
                run_semantic=False,
                run_unused=False,
                jaccard_threshold=0.5,
            ),
            set(),
        ),
        (
            AnalyzerConfig(
                run_traditional=False,
                run_semantic=False,
                run_unused=True,
                strict_unused=True,
            ),
            {"used", "unused"},
        ),
    ],
)
def test_unused_detection_config_variants(tmp_path: Path, analyzer_config, expected_unused) -> None:
    project = create_project(tmp_path, build_two_function_source())
    analyzer = CodeAnalyzer(analyzer_config)

    result = analyzer.analyze(project)

    assert {unit.name for unit in result.potentially_unused} == expected_unused


def test_integration_on_mixed_project(tmp_path: Path) -> None:
    src_root = tmp_path / "project"
    src_root.mkdir()

    (src_root / "bad.py").write_text("def bad(:\n    pass")
    (src_root / "tests").mkdir()
    (src_root / "tests" / "test_skip.py").write_text("def test_case():\n    return 1")
    (src_root / "util.py").write_text(
        dedent(
            """
            def add(a, b):
                return a + b

            def sum_values(x, y):
                return x + y

            def helper():
                return 2

            def caller():
                return helper()

            class Engine:
                def run(self):
                    return helper()

                def _internal(self):
                    return 0

            def _private_entry():
                return helper()

            def get_value():
                return 3

            def set_value(value):
                return value
            """
        ).strip()
    )

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_semantic=False,
            run_traditional=True,
            run_unused=True,
            include_private=False,
            filter_tiny_traditional=False,
        )
    )
    result = analyzer.analyze(src_root)

    assert len(result.traditional_duplicates) >= 1
    assert result.hybrid_duplicates == []
    assert not any("tests" in str(unit.file_path) for unit in result.units)
    names = {unit.name for unit in result.potentially_unused}
    assert "caller" not in names
    assert "_private_entry" not in names


def test_analyze_directory_uses_auto_revision_for_custom_model(tmp_path: Path, monkeypatch) -> None:
    source = "def add_one(x):\n    return x + 1\n"
    project = create_project(tmp_path, source)
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        analyzer_module,
        "run_semantic_analysis",
        _make_semantic_runner(capture=captured),
    )

    analyze_directory(
        project,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        min_semantic_statements=0,
        run_unused=False,
    )

    assert captured["model_name"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert captured["revision"] is None


def test_combined_mode_preserves_near_dupes_for_semantic_confirmation(
    tmp_path: Path, monkeypatch
) -> None:
    source = dedent(
        """
        def exact_a():
            return 1

        def exact_b():
            return 1

        def near_c():
            return 2
        """
    ).strip()
    project = create_project(tmp_path, source)

    captured_exclude_pairs: set[tuple[str, str]] = set()
    expected_exact_pair: tuple[str, str] = ("", "")

    def fake_traditional(
        units,
        jaccard_threshold=0.85,
        compute_unused=True,
    ):
        first, second, third = units
        nonlocal expected_exact_pair
        expected_exact_pair = tuple(sorted((first.uid, second.uid)))
        return (
            [DuplicatePair(unit_a=first, unit_b=second, similarity=1.0, method="ast_hash")],
            [DuplicatePair(unit_a=second, unit_b=third, similarity=0.9, method="jaccard")],
            [],
        )

    monkeypatch.setattr(analyzer_module, "run_traditional_analysis", fake_traditional)
    monkeypatch.setattr(
        analyzer_module,
        "run_semantic_analysis",
        _make_semantic_runner(
            capture_exclude_pairs=captured_exclude_pairs,
            duplicate_factory=lambda units: [
                DuplicatePair(
                    unit_a=units[1],
                    unit_b=units[2],
                    similarity=0.95,
                    method="semantic",
                )
            ],
        ),
    )

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=True,
            run_semantic=True,
            run_unused=False,
            min_semantic_statements=0,
            jaccard_threshold=0.85,
            semantic_threshold=0.82,
            filter_tiny_traditional=False,
        )
    )
    result = analyzer.analyze(project)

    assert set(captured_exclude_pairs) == {expected_exact_pair}
    assert len(result.traditional_duplicates) == 2
    assert len(result.semantic_duplicates) == 1
    assert len(result.hybrid_duplicates) == 2
    assert {duplicate.tier for duplicate in result.hybrid_duplicates} == {
        "exact",
        "hybrid_confirmed",
    }


def test_short_functions_are_skipped_from_semantic(tmp_path: Path) -> None:
    source = dedent(
        """
        def tiny():
            return 1

        def another_tiny():
            return 2
        """
    ).strip()
    project = create_project(tmp_path, source, module="tiny.py")
    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=False,
            run_semantic=True,
            run_unused=False,
            min_semantic_statements=3,
        )
    )
    result = analyzer.analyze(project)
    assert result.semantic_duplicates == []


def test_decorated_methods_survive_semantic_and_tiny_filters(tmp_path: Path, monkeypatch) -> None:
    source = dedent(
        """
        class First:
            @property
            def area(self):
                width = self.width
                height = self.height
                scale = self.scale
                return width * height * scale

        class Second:
            @property
            def area(self):
                width = self.width
                height = self.height
                scale = self.scale
                return width * height * scale
        """
    ).strip()
    project = create_project(tmp_path, source, module="decorated.py")
    semantic_units: list[CodeUnit] = []

    def capture_semantic_candidates(units, **_kwargs):
        semantic_units.extend(units)
        return (
            np.zeros((len(units), 2), dtype=np.float32),
            [],
            _embedding_identity_from_kwargs(_kwargs),
        )

    monkeypatch.setattr(analyzer_module, "run_semantic_analysis", capture_semantic_candidates)
    result = CodeAnalyzer(AnalyzerConfig(run_unused=False)).analyze(project)

    assert {unit.qualified_name for unit in semantic_units} == {
        "decorated.First.area",
        "decorated.Second.area",
    }
    assert any(
        {duplicate.unit_a.qualified_name, duplicate.unit_b.qualified_name}
        == {"decorated.First.area", "decorated.Second.area"}
        for duplicate in result.traditional_duplicates
    )


@pytest.mark.parametrize(
    ("semantic_unit_types", "expected_types"),
    [
        (None, {CodeUnitType.FUNCTION, CodeUnitType.METHOD}),
        (("class",), {CodeUnitType.CLASS}),
    ],
)
def test_semantic_unit_scope(
    tmp_path: Path,
    monkeypatch,
    semantic_unit_types: tuple[str, ...] | None,
    expected_types: set[CodeUnitType],
) -> None:
    source = dedent(
        """
        class Box:
            def method(self):
                return 1

        def helper():
            return 2
        """
    ).strip()
    project = create_project(tmp_path, source, module="scope.py")
    captured_types: list[CodeUnitType] = []

    monkeypatch.setattr(
        analyzer_module,
        "run_semantic_analysis",
        _capture_semantic_unit_types(captured_types),
    )

    config_kwargs = {
        "run_traditional": False,
        "run_semantic": True,
        "run_unused": False,
        "min_semantic_statements": 0,
    }
    if semantic_unit_types is not None:
        config_kwargs["semantic_unit_types"] = semantic_unit_types
    analyzer = CodeAnalyzer(AnalyzerConfig(**config_kwargs))
    analyzer.analyze(project)

    assert captured_types
    assert set(captured_types) == expected_types


@pytest.mark.parametrize(
    ("run_semantic", "expected_types"),
    [
        (True, set()),
        (
            False,
            {CodeUnitType.CLASS, CodeUnitType.METHOD, CodeUnitType.FUNCTION},
        ),
    ],
)
def test_traditional_scope_depends_on_semantic_mode(
    tmp_path: Path,
    monkeypatch,
    run_semantic: bool,
    expected_types: set[CodeUnitType],
) -> None:
    source = dedent(
        """
        class Box:
            def method(self):
                return 1

        def tiny():
            return 2
        """
    ).strip()
    project = create_project(tmp_path, source, module="scope.py")
    captured_traditional_units: list[CodeUnit] = []

    monkeypatch.setattr(
        analyzer_module,
        "run_traditional_analysis",
        _capture_traditional_units_runner(captured_traditional_units),
    )
    monkeypatch.setattr(analyzer_module, "run_semantic_analysis", _make_semantic_runner())

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=True,
            run_semantic=run_semantic,
            run_unused=False,
            min_semantic_statements=2,
        )
    )
    analyzer.analyze(project)

    assert {unit.unit_type for unit in captured_traditional_units} == expected_types


@pytest.mark.parametrize(
    ("filter_tiny_traditional", "expected_exact_duplicate"),
    [(None, False), (False, True)],
)
def test_tiny_exact_duplicate_filter(
    tmp_path: Path,
    filter_tiny_traditional: bool | None,
    expected_exact_duplicate: bool,
) -> None:
    source = dedent(
        """
        def wrapper_a():
            return helper_a()

        def wrapper_b():
            return helper_b()

        def helper_a():
            return 1

        def helper_b():
            return 1
        """
    ).strip()
    project = create_project(tmp_path, source, module="tiny_exact.py")

    config_kwargs = {
        "run_traditional": True,
        "run_semantic": False,
        "run_unused": False,
        "jaccard_threshold": 0.99,
    }
    if filter_tiny_traditional is not None:
        config_kwargs["filter_tiny_traditional"] = filter_tiny_traditional
    analyzer = CodeAnalyzer(AnalyzerConfig(**config_kwargs))
    result = analyzer.analyze(project)

    has_exact_duplicate = any(
        duplicate.method in {"ast_hash", "token_hash"}
        for duplicate in result.traditional_duplicates
    )
    assert has_exact_duplicate is expected_exact_duplicate


@pytest.mark.parametrize(
    ("similarity", "expected_count"),
    [
        (0.90, 0),
        (0.95, 1),
    ],
)
def test_tiny_near_duplicates_use_high_jaccard_floor(
    tmp_path: Path, monkeypatch, similarity: float, expected_count: int
) -> None:
    source = dedent(
        """
        def first():
            return alpha()

        def second():
            return beta()
        """
    ).strip()
    project = create_project(tmp_path, source, module="tiny_near.py")

    def fake_traditional(
        units,
        jaccard_threshold=0.85,
        compute_unused=True,
    ):
        return (
            [],
            [
                DuplicatePair(
                    unit_a=units[0], unit_b=units[1], similarity=similarity, method="jaccard"
                )
            ],
            [],
        )

    monkeypatch.setattr(analyzer_module, "run_traditional_analysis", fake_traditional)

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=True,
            run_semantic=False,
            run_unused=False,
            tiny_near_jaccard_min=0.93,
        )
    )
    result = analyzer.analyze(project)

    assert len(result.traditional_duplicates) == expected_count


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
        _make_semantic_runner(capture=captured),
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
        _make_semantic_runner(capture=captured),
    )

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=False,
            run_semantic=True,
            run_unused=False,
            min_semantic_statements=0,
        )
    )
    analyzer.analyze(project)

    # Every language group is scanned at its own gate; the scalar floor only
    # covers languages without a calibrated entry.
    assert captured["language_thresholds"] == {"python": 0.90, "javascript": 0.60}
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


def test_cross_language_pairs_require_opt_in_and_use_looser_gate(
    tmp_path: Path, monkeypatch
) -> None:
    project = _create_two_language_project(tmp_path)
    vectors = _two_language_vectors()
    # A mixed pair at 0.70: below python's gate, above javascript's.
    vectors["alpha_one"] = [1.0, 0.0, 0.0, 0.0]
    vectors["betaOne"] = [0.70, 0.0, float(np.sqrt(1.0 - 0.70**2)), 0.0]

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
    }
    default_result = CodeAnalyzer(AnalyzerConfig(**base_config)).analyze(project)
    assert all(
        duplicate.unit_a.language == duplicate.unit_b.language
        for duplicate in default_result.semantic_duplicates
    )

    # Opted in, the mixed pair is held to the looser of its two language gates:
    # 0.70 clears min(0.90, 0.60) but would fail the python gate alone.
    opted_result = CodeAnalyzer(AnalyzerConfig(cross_language=True, **base_config)).analyze(project)
    assert frozenset({"alpha_one", "betaOne"}) in {
        frozenset({duplicate.unit_a.name, duplicate.unit_b.name})
        for duplicate in opted_result.semantic_duplicates
    }


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
        "resolve_model_profile",
        lambda _model: pytest.fail("explicit threshold must bypass profile gates"),
    )
    monkeypatch.setattr(
        analyzer_module,
        "run_semantic_analysis",
        _make_semantic_runner(capture=captured, duplicate_factory=paired_duplicates),
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

    assert captured["threshold"] == 0.70
    assert len(result.semantic_duplicates) == 1


def test_unused_semantic_pairs_are_filtered(tmp_path: Path, monkeypatch) -> None:
    source = dedent(
        """
        def _a():
            x = 1
            return x + 1

        def _b():
            y = 2
            return y + 2
        """
    ).strip()
    project = create_project(tmp_path, source, module="pairs.py")

    monkeypatch.setattr(
        analyzer_module,
        "run_semantic_analysis",
        _make_semantic_runner(
            duplicate_factory=lambda units: [
                DuplicatePair(
                    unit_a=units[0],
                    unit_b=units[1],
                    similarity=0.99,
                    method="semantic",
                )
            ],
        ),
    )

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=False,
            run_semantic=True,
            run_unused=True,
            min_semantic_statements=0,
            strict_unused=False,
        )
    )

    result = analyzer.analyze(project)
    assert result.semantic_duplicates == []


def test_semantic_only_pre_excludes_exact_hash_pairs(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "src"
    project.mkdir()
    (project / "__init__.py").write_text("")
    (project / "a.py").write_text("def helper(x):\n    return x + 1\n")
    (project / "b.py").write_text("def helper(x):\n    return x + 1\n")

    captured_exclude_pairs: set[tuple[str, str]] = set()

    monkeypatch.setattr(
        analyzer_module,
        "run_semantic_analysis",
        _make_semantic_runner(capture_exclude_pairs=captured_exclude_pairs),
    )

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=False,
            run_semantic=True,
            run_unused=False,
            min_semantic_statements=0,
        )
    )

    result = analyzer.analyze(project)
    assert result.semantic_duplicates == []
    assert not captured_exclude_pairs


def test_combined_mode_excludes_tiny_filtered_ast_only_exact_pairs(
    tmp_path: Path, monkeypatch
) -> None:
    """Tiny-filtered ast-hash-only exact pairs must stay excluded from semantic scoring."""
    project = tmp_path / "src"
    project.mkdir()
    (project / "__init__.py").write_text("")
    (project / "a.py").write_text(
        "def alpha(x):\n    first = x + 1\n    second = first * 2\n    return second\n"
    )
    (project / "b.py").write_text(
        "def beta(y):\n    one = y + 1\n    two = one * 2\n    return two\n"
    )

    captured_exclude_pairs: set[tuple[str, str]] = set()
    monkeypatch.setattr(
        analyzer_module,
        "run_semantic_analysis",
        _make_semantic_runner(capture_exclude_pairs=captured_exclude_pairs),
    )

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_unused=False,
            # Both functions have 3 statements: semantic candidates at the default
            # min_semantic_statements, and tiny under this raised cutoff.
            tiny_unit_statement_cutoff=4,
        )
    )
    result = analyzer.analyze(project)

    unit_by_name = {unit.name: unit for unit in result.units}
    pair = ordered_pair_key(unit_by_name["alpha"], unit_by_name["beta"])

    # Same normalized AST, different identifiers: an ast_hash-only exact pair.
    assert unit_by_name["alpha"].structural_hash == unit_by_name["beta"].structural_hash
    assert unit_by_name["alpha"].token_hash != unit_by_name["beta"].token_hash
    # The tiny filter strips the pair from traditional output...
    assert result.traditional_duplicates == []
    # ...but semantic scoring must still treat it as an already-known exact pair.
    assert pair in captured_exclude_pairs


def test_combined_mode_fails_hard_on_runtime_semantic_error_by_default(
    tmp_path: Path, monkeypatch
) -> None:
    source = "def entry(x):\n    return x + 1\n"
    project = create_project(tmp_path, source)

    monkeypatch.setattr(
        analyzer_module,
        "run_semantic_analysis",
        _make_semantic_runner(error=RuntimeError("CUDA out of memory")),
    )

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=True,
            run_semantic=True,
            run_unused=False,
            min_semantic_statements=0,
            filter_tiny_traditional=False,
        )
    )

    with pytest.raises(RuntimeError, match="allow-semantic-fallback"):
        analyzer.analyze(project)


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


def test_combined_mode_fallback_keeps_scoped_traditional_units(tmp_path: Path, monkeypatch) -> None:
    source = dedent(
        """
        class Box:
            def method(self):
                value = 1
                return value

        def short():
            return 1

        def longer():
            value = 2
            return value
        """
    ).strip()
    project = create_project(tmp_path, source)

    traditional_calls: list[tuple[tuple[str, ...], list[str]]] = []

    def fake_traditional(
        units,
        jaccard_threshold=0.85,
        compute_unused=True,
    ):
        traditional_calls.append(
            (tuple(unit.name for unit in units), [unit.name for unit in units])
        )
        return [], [], []

    monkeypatch.setattr(analyzer_module, "run_traditional_analysis", fake_traditional)
    monkeypatch.setattr(
        analyzer_module,
        "run_semantic_analysis",
        _make_semantic_runner(error=RuntimeError("CUDA out of memory")),
    )

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=True,
            run_semantic=True,
            allow_semantic_fallback=True,
            run_unused=False,
            min_semantic_statements=2,
            filter_tiny_traditional=False,
        )
    )
    analyzer.analyze(project)

    assert len(traditional_calls) == 1
    assert set(traditional_calls[0][0]) == {"method", "longer"}


def test_combined_mode_fallback_marks_semantic_degradation(tmp_path: Path, monkeypatch) -> None:
    source = dedent(
        """
        def dead(x):
            return x
        """
    ).strip()
    project = create_project(tmp_path, source)

    monkeypatch.setattr(
        analyzer_module,
        "run_semantic_analysis",
        _make_semantic_runner(error=RuntimeError("backend unavailable")),
    )

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=True,
            run_semantic=True,
            allow_semantic_fallback=True,
            run_unused=False,
            min_semantic_statements=0,
            filter_tiny_traditional=False,
        )
    )
    result = analyzer.analyze(project)

    assert result.semantic_fallback is True
    assert result.semantic_fallback_reason is not None
    assert "backend unavailable" in result.semantic_fallback_reason


def test_search_after_analyze_uses_analysis_task_when_unset(tmp_path: Path, monkeypatch) -> None:
    source = "def entry(x):\n    return x + 1\n"
    project = create_project(tmp_path, source)
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        analyzer_module,
        "run_semantic_analysis",
        _make_semantic_runner(capture=captured),
    )

    monkeypatch.setattr(
        semantic_module,
        "find_similar_to_query",
        _capture_query_runner(captured),
    )

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=False,
            run_semantic=True,
            run_unused=False,
            min_semantic_statements=0,
        )
    )
    analyzer.analyze(project)
    analyzer.search("entry")

    assert captured["semantic_task"] == analyzer_module.DEFAULT_CHECK_SEMANTIC_TASK
    assert captured["query_semantic_task"] == analyzer_module.DEFAULT_CHECK_SEMANTIC_TASK


def test_embeddinggemma_search_after_analyze_requires_explicit_threshold(
    tmp_path: Path, monkeypatch
) -> None:
    project = create_project(tmp_path, "def entry(x):\n    return x + 1\n")
    monkeypatch.setattr(
        analyzer_module,
        "run_semantic_analysis",
        _make_semantic_runner(),
    )
    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=False,
            run_semantic=True,
            run_unused=False,
            min_semantic_statements=0,
            model_name="embeddinggemma-300m",
            embedding_cache=False,
        )
    )
    analyzer.analyze(project)

    with pytest.raises(ValueError, match=r"search\(threshold=\.\.\.\)"):
        analyzer.search("entry")


def test_search_threshold_argument_overrides_the_config_for_one_call(
    tmp_path: Path, monkeypatch
) -> None:
    project = create_project(tmp_path, "def entry(x):\n    return x + 1\n")
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        analyzer_module,
        "run_semantic_analysis",
        _make_semantic_runner(),
    )
    monkeypatch.setattr(
        semantic_module,
        "find_similar_to_query",
        _capture_query_runner(captured),
    )

    # The per-call threshold must not disturb the calibrated per-language
    # duplicate gates, which config.semantic_threshold would flatten.
    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=False,
            run_semantic=True,
            run_unused=False,
            min_semantic_statements=0,
            model_name="embeddinggemma-300m",
            embedding_cache=False,
        )
    )
    analyzer.analyze(project)
    analyzer.search("entry", threshold=0.31)

    assert analyzer.config.semantic_threshold is None
    assert captured["query_threshold"] == 0.31


def test_search_threshold_defaults_to_none_and_honors_explicit_config(
    tmp_path: Path, monkeypatch
) -> None:
    source = "def entry(x):\n    return x + 1\n"
    project = create_project(tmp_path, source)
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        analyzer_module,
        "run_semantic_analysis",
        _make_semantic_runner(capture=captured),
    )

    monkeypatch.setattr(
        semantic_module,
        "find_similar_to_query",
        _capture_query_runner(captured),
    )

    base_config = {
        "run_traditional": False,
        "run_semantic": True,
        "run_unused": False,
        "min_semantic_statements": 0,
    }
    analyzer = CodeAnalyzer(AnalyzerConfig(**base_config))
    analyzer.analyze(project)
    analyzer.search("entry")
    assert captured["query_threshold"] is None

    explicit = CodeAnalyzer(AnalyzerConfig(semantic_threshold=0.7, **base_config))
    explicit.analyze(project)
    explicit.search("entry")
    assert captured["query_threshold"] == 0.7


@pytest.mark.parametrize(
    "config_overrides",
    [
        {"semantic_task": "classification"},
        {"instruction_prefix": "CUSTOM: "},
        {"model_revision": "f" * 40},
        {"trust_remote_code": True},
    ],
)
def test_uncalibrated_duplicate_context_requires_explicit_threshold(
    tmp_path: Path, monkeypatch, config_overrides: dict[str, str]
) -> None:
    project = create_project(tmp_path, "def entry(x):\n    y = x + 1\n    return y\n")
    monkeypatch.setattr(
        analyzer_module,
        "run_semantic_analysis",
        _make_semantic_runner(),
    )
    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=False,
            run_semantic=True,
            run_unused=False,
            min_semantic_statements=0,
            model_name="embeddinggemma-300m",
            **config_overrides,
        )
    )

    with pytest.raises(ValueError, match="provide semantic_threshold explicitly"):
        analyzer.analyze(project)


def test_index_embeds_corpus_without_mining_duplicates(tmp_path: Path, monkeypatch) -> None:
    source = "def entry(x):\n    return x + 1\n"
    project = create_project(tmp_path, source)
    captured: dict[str, object] = {}
    embedded_units: list[CodeUnit] = []

    def fail_duplicate_mining(*_args, **_kwargs):
        raise AssertionError("index()/search() must never mine duplicate pairs")

    monkeypatch.setattr(analyzer_module, "run_semantic_analysis", fail_duplicate_mining)
    monkeypatch.setattr(semantic_module, "find_semantic_duplicates", fail_duplicate_mining)

    def fake_compute_embeddings(units, **kwargs):
        embedded_units.extend(units)
        captured.update(kwargs)
        return (
            np.zeros((len(units), 2), dtype=np.float32),
            _embedding_identity_from_kwargs(kwargs),
        )

    monkeypatch.setattr(analyzer_module, "compute_embeddings", fake_compute_embeddings)
    monkeypatch.setattr(
        semantic_module,
        "find_similar_to_query",
        _capture_query_runner(captured),
    )

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=False,
            run_semantic=True,
            run_unused=False,
            min_semantic_statements=0,
        )
    )
    indexed = analyzer.index(project)
    results = analyzer.search("entry")

    assert indexed == 1
    assert [unit.name for unit in embedded_units] == ["entry"]
    assert results == []
    assert captured["semantic_task"] == analyzer_module.DEFAULT_SEARCH_SEMANTIC_TASK
    assert captured["query_semantic_task"] == analyzer_module.DEFAULT_SEARCH_SEMANTIC_TASK
    assert captured["cache_scope"] == project.resolve()


def test_index_empty_corpus_yields_empty_search(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    analyzer = CodeAnalyzer(
        AnalyzerConfig(run_traditional=False, run_semantic=True, run_unused=False)
    )

    assert analyzer.index(empty) == 0
    assert analyzer.search("anything") == []


def test_search_requires_reindex_when_local_model_contents_change(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = create_project(tmp_path, "def entry(x):\n    return x + 1\n")
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    weights_path = model_dir / "model.safetensors"
    weights_path.write_bytes(b"first weights")

    monkeypatch.setattr(
        analyzer_module,
        "compute_embeddings",
        lambda units, **kwargs: (
            np.zeros((len(units), 2), dtype=np.float32),
            _embedding_identity_from_kwargs(kwargs),
        ),
    )
    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            model_name=str(model_dir),
            device="cpu",
            embedding_cache=False,
            min_semantic_statements=0,
        )
    )
    analyzer.index(project)

    weights_path.write_bytes(b"second weights")

    with pytest.raises(RuntimeError, match=r"changed since this corpus was indexed.*index\(\)"):
        analyzer.search("entry")


def test_search_requires_reindex_when_embedding_runtime_variant_changes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = create_project(tmp_path, "def entry(x):\n    return x + 1\n")
    monkeypatch.setattr(
        analyzer_module,
        "compute_embeddings",
        lambda units, **kwargs: (
            np.zeros((len(units), 2), dtype=np.float32),
            _embedding_identity_from_kwargs(kwargs),
        ),
    )
    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            device="cpu",
            embedding_cache=False,
            min_semantic_statements=0,
        )
    )
    analyzer.index(project)

    analyzer.config.instruction_prefix = "Represent this code differently: "

    with pytest.raises(RuntimeError, match=r"changed since this corpus was indexed.*index\(\)"):
        analyzer.search("entry")


def test_semantic_only_fails_hard_on_runtime_semantic_error(tmp_path: Path, monkeypatch) -> None:
    source = "def entry(x):\n    return x + 1\n"
    project = create_project(tmp_path, source)

    monkeypatch.setattr(
        analyzer_module,
        "run_semantic_analysis",
        _make_semantic_runner(error=RuntimeError("CUDA out of memory")),
    )

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=False,
            run_semantic=True,
            run_unused=False,
            min_semantic_statements=0,
        )
    )

    with pytest.raises(RuntimeError):
        analyzer.analyze(project)


def test_suppress_test_semantic_matches_filters_test_named_pairs(
    tmp_path: Path, monkeypatch
) -> None:
    source = dedent(
        """
        def test_alpha():
            return 1

        def test_beta():
            return 2

        def helper_alpha():
            return 3

        def helper_beta():
            return 4
        """
    ).strip()
    project = create_project(tmp_path, source, module="tests_like.py")

    def fake_run_semantic(
        units,
        model_name="Alibaba-NLP/gte-modernbert-base",
        instruction_prefix=None,
        threshold=0.82,
        exclude_pairs=None,
        batch_size=32,
        revision=None,
        trust_remote_code=None,
        semantic_task=None,
        **_device_kwargs,
    ):
        by_name = {unit.name: unit for unit in units}
        duplicates = [
            DuplicatePair(
                unit_a=by_name["test_alpha"],
                unit_b=by_name["test_beta"],
                similarity=0.99,
                method="semantic",
            ),
            DuplicatePair(
                unit_a=by_name["helper_alpha"],
                unit_b=by_name["helper_beta"],
                similarity=0.99,
                method="semantic",
            ),
        ]
        identity_kwargs = {
            "model_name": model_name,
            "instruction_prefix": instruction_prefix,
            "revision": revision,
            "trust_remote_code": trust_remote_code,
            "semantic_task": semantic_task,
            **_device_kwargs,
        }
        return (
            np.zeros((len(units), 2), dtype=np.float32),
            duplicates,
            _embedding_identity_from_kwargs(identity_kwargs),
        )

    monkeypatch.setattr(analyzer_module, "run_semantic_analysis", fake_run_semantic)

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=False,
            run_semantic=True,
            run_unused=False,
            min_semantic_statements=0,
            suppress_test_semantic_matches=True,
        )
    )

    result = analyzer.analyze(project)

    assert len(result.semantic_duplicates) == 1
    assert {
        result.semantic_duplicates[0].unit_a.name,
        result.semantic_duplicates[0].unit_b.name,
    } == {"helper_alpha", "helper_beta"}


def test_hybrid_synthesis_exact_only_included(tmp_path: Path) -> None:
    unit_a = make_code_unit(tmp_path, name="a", source="def a(x):\n    return x + 1\n", lineno=1)
    unit_b = make_code_unit(tmp_path, name="b", source="def b(y):\n    return y + 1\n", lineno=5)
    traditional = [DuplicatePair(unit_a=unit_a, unit_b=unit_b, similarity=1.0, method="ast_hash")]

    hybrid = analyzer_module._synthesize_hybrid_duplicates(
        traditional,
        [],
        jaccard_threshold=0.85,
    )

    assert len(hybrid) == 1
    assert hybrid[0].tier == "exact"
    assert hybrid[0].confidence == 1.0


def test_hybrid_synthesis_jaccard_only_included(tmp_path: Path) -> None:
    unit_a = make_code_unit(tmp_path, name="a", source="def a(x):\n    return x + 1\n", lineno=1)
    unit_b = make_code_unit(tmp_path, name="b", source="def b(y):\n    return y + 2\n", lineno=5)
    traditional = [DuplicatePair(unit_a=unit_a, unit_b=unit_b, similarity=0.9, method="jaccard")]

    hybrid = analyzer_module._synthesize_hybrid_duplicates(
        traditional,
        [],
        jaccard_threshold=0.85,
    )

    assert len(hybrid) == 1
    assert hybrid[0].tier == "traditional_near"
    assert hybrid[0].jaccard_similarity == pytest.approx(0.9)


def test_hybrid_synthesis_hybrid_confirmed(tmp_path: Path) -> None:
    unit_a = make_code_unit(tmp_path, name="a", source="def a(x):\n    return x + 1\n", lineno=1)
    unit_b = make_code_unit(tmp_path, name="b", source="def b(y):\n    return y + 1\n", lineno=5)
    traditional = [DuplicatePair(unit_a=unit_a, unit_b=unit_b, similarity=0.88, method="jaccard")]
    semantic = [DuplicatePair(unit_a=unit_a, unit_b=unit_b, similarity=0.93, method="semantic")]

    hybrid = analyzer_module._synthesize_hybrid_duplicates(
        traditional,
        semantic,
        jaccard_threshold=0.85,
    )

    assert len(hybrid) == 1
    assert hybrid[0].tier == "hybrid_confirmed"
    assert hybrid[0].confidence == pytest.approx((0.5 * 0.93) + (0.5 * 0.88))


def test_hybrid_synthesis_semantic_only_corroboration_sets_tier(tmp_path: Path) -> None:
    unit_a = make_code_unit(
        tmp_path, name="a", source="def alpha(v):\n    z = v + 1\n    return z\n", lineno=1
    )
    unit_b = make_code_unit(
        tmp_path, name="b", source="def beta(v):\n    q = v + 2\n    return q\n", lineno=6
    )

    # Semantic pairs arrive pre-gated. Corroborating lexical/size evidence
    # promotes them to the high-confidence tier.
    gated_semantic = [
        DuplicatePair(unit_a=unit_a, unit_b=unit_b, similarity=0.75, method="semantic")
    ]
    hybrid = analyzer_module._synthesize_hybrid_duplicates(
        [],
        gated_semantic,
        jaccard_threshold=0.85,
    )
    assert len(hybrid) == 1
    assert hybrid[0].tier == "semantic_high_confidence"
    assert hybrid[0].confidence == pytest.approx(0.45 + (0.55 * 0.75))

    weak_sources_a = make_code_unit(
        tmp_path,
        name="c",
        source="def c(a):\n    x = a + 1\n    y = x + 1\n    z = y + 1\n    return z\n",
        lineno=12,
    )
    weak_sources_b = make_code_unit(
        tmp_path,
        name="d",
        source="def d(v):\n    return v\n",
        lineno=20,
    )
    weak_semantic = [
        DuplicatePair(
            unit_a=weak_sources_a, unit_b=weak_sources_b, similarity=0.95, method="semantic"
        )
    ]
    hybrid_weak = analyzer_module._synthesize_hybrid_duplicates(
        [],
        weak_semantic,
        jaccard_threshold=0.85,
    )
    assert len(hybrid_weak) == 1
    assert hybrid_weak[0].tier == "semantic_review"
    assert hybrid_weak[0].confidence == pytest.approx(0.40 + (0.45 * 0.95))
    assert hybrid_weak[0].weak_identifier_jaccard == 0.0
    assert hybrid_weak[0].statement_count_ratio == pytest.approx(0.25)


def test_semantic_review_never_outranks_a_corroborated_pair(tmp_path: Path) -> None:
    # Same cosine, different corroboration: the least-evidenced tier must sort
    # below every tier that carries extra evidence.
    review_a = make_code_unit(
        tmp_path,
        name="review_a",
        source="def review_a(a):\n    x = a + 1\n    y = x + 1\n    z = y + 1\n    return z\n",
        lineno=1,
    )
    review_b = make_code_unit(
        tmp_path, name="review_b", source="def review_b(v):\n    return v\n", lineno=12
    )
    confirmed_a = make_code_unit(
        tmp_path, name="confirmed_a", source="def confirmed_a(x):\n    return x + 1\n", lineno=20
    )
    confirmed_b = make_code_unit(
        tmp_path, name="confirmed_b", source="def confirmed_b(y):\n    return y + 1\n", lineno=26
    )

    hybrid = analyzer_module._synthesize_hybrid_duplicates(
        [DuplicatePair(unit_a=confirmed_a, unit_b=confirmed_b, similarity=0.86, method="jaccard")],
        [
            DuplicatePair(unit_a=review_a, unit_b=review_b, similarity=0.97, method="semantic"),
            DuplicatePair(
                unit_a=confirmed_a, unit_b=confirmed_b, similarity=0.97, method="semantic"
            ),
        ],
        jaccard_threshold=0.85,
    )

    assert [duplicate.tier for duplicate in hybrid] == ["hybrid_confirmed", "semantic_review"]
    assert hybrid[0].confidence > hybrid[1].confidence


def test_hybrid_synthesis_publishes_alpha_renamed_semantic_pair(tmp_path: Path) -> None:
    unit_a = make_code_unit(
        tmp_path,
        name="collect_total",
        source=(
            "def collect_total(records):\n"
            "    accepted = [record for record in records if record.enabled]\n"
            "    amount = sum(record.value for record in accepted)\n"
            "    return amount\n"
        ),
        lineno=1,
    )
    unit_b = make_code_unit(
        tmp_path,
        name="measure_sum",
        source=(
            "def measure_sum(entries):\n"
            "    chosen = [entry for entry in entries if entry.ready]\n"
            "    result = sum(entry.weight for entry in chosen)\n"
            "    return result\n"
        ),
        lineno=8,
    )
    semantic = [DuplicatePair(unit_a=unit_a, unit_b=unit_b, similarity=0.91, method="semantic")]

    hybrid = analyzer_module._synthesize_hybrid_duplicates(
        [],
        semantic,
        jaccard_threshold=0.85,
    )

    assert len(hybrid) == 1
    assert hybrid[0].tier == "semantic_review"
    assert hybrid[0].weak_identifier_jaccard == 0.0
    assert hybrid[0].statement_count_ratio == 1.0


def test_mixed_mode_semantic_failure_still_builds_hybrid_from_traditional(
    tmp_path: Path, monkeypatch
) -> None:
    source = dedent(
        """
        def one(x):
            return x + 1

        def two(y):
            return y + 2
        """
    ).strip()
    project = create_project(tmp_path, source)

    monkeypatch.setattr(
        analyzer_module,
        "run_traditional_analysis",
        _traditional_single_jaccard_runner(0.9),
    )
    monkeypatch.setattr(
        analyzer_module,
        "run_semantic_analysis",
        _make_semantic_runner(error=SemanticBackendError("semantic backend mismatch")),
    )

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=True,
            run_semantic=True,
            allow_semantic_fallback=True,
            run_unused=False,
            min_semantic_statements=0,
            filter_tiny_traditional=False,
        )
    )
    result = analyzer.analyze(project)

    assert len(result.traditional_duplicates) == 1
    assert len(result.hybrid_duplicates) == 1
    assert result.hybrid_duplicates[0].tier == "traditional_near"


def test_single_method_modes_bypass_hybrid_synthesis(tmp_path: Path, monkeypatch) -> None:
    source = dedent(
        """
        def one(x):
            return x + 1

        def two(y):
            return y + 2
        """
    ).strip()
    project = create_project(tmp_path, source)

    monkeypatch.setattr(
        analyzer_module,
        "run_traditional_analysis",
        _traditional_single_jaccard_runner(0.9),
    )
    monkeypatch.setattr(
        analyzer_module,
        "run_semantic_analysis",
        _make_semantic_runner(
            duplicate_factory=lambda units: [
                DuplicatePair(
                    unit_a=units[0],
                    unit_b=units[1],
                    similarity=0.96,
                    method="semantic",
                )
            ],
        ),
    )

    traditional_result = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=True,
            run_semantic=False,
            run_unused=False,
            min_semantic_statements=0,
            filter_tiny_traditional=False,
        )
    ).analyze(project)
    assert len(traditional_result.traditional_duplicates) == 1
    assert traditional_result.hybrid_duplicates == []

    semantic_result = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=False,
            run_semantic=True,
            run_unused=False,
            min_semantic_statements=0,
        )
    ).analyze(project)
    assert len(semantic_result.semantic_duplicates) == 1
    assert semantic_result.hybrid_duplicates == []


def test_search_requires_embeddings(tmp_path: Path) -> None:
    source = "def entry():\n    return 1\n"
    create_project(tmp_path, source)
    project = tmp_path / "src"
    analyzer = CodeAnalyzer(
        AnalyzerConfig(run_semantic=False, run_traditional=False, run_unused=False)
    )

    analyzer.analyze(project)
    with pytest.raises(RuntimeError, match="run_semantic=True"):
        analyzer.search("entry")


def test_empty_reanalysis_clears_previous_search_state(tmp_path: Path, monkeypatch) -> None:
    project = create_project(tmp_path, "def entry():\n    return 1\n")
    empty_project = tmp_path / "empty"
    empty_project.mkdir()
    monkeypatch.setattr(
        analyzer_module,
        "run_semantic_analysis",
        _make_semantic_runner(),
    )
    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=False,
            run_semantic=True,
            run_unused=False,
            min_semantic_statements=0,
        )
    )

    analyzer.analyze(project)
    result = analyzer.analyze(empty_project)

    assert result.analysis_mode == "none"
    with pytest.raises(RuntimeError, match="run_semantic=True"):
        analyzer.search("entry")


def test_invalid_threshold_raises() -> None:
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

    with pytest.raises(ValueError, match="tiny_near_jaccard_min"):
        AnalyzerConfig(tiny_near_jaccard_min=1.1)


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


def test_empty_directory_analysis(tmp_path: Path) -> None:
    analyzer = CodeAnalyzer()
    result = analyzer.analyze(tmp_path)

    assert result.units == []
    assert result.traditional_duplicates == []
    assert result.semantic_duplicates == []
    assert result.hybrid_duplicates == []
    assert result.potentially_unused == []


def test_empty_extraction_still_validates_explicit_device(tmp_path: Path, monkeypatch) -> None:
    """An empty corpus must not turn an unavailable explicit device into a success.

    ``analyze()`` returns before the semantic layer runs when extraction finds
    no units, so the explicit-device contract has to be enforced on that
    shortcut too; only a combined-mode fallback opt-in may downgrade it.
    """

    def _raise_unavailable(*_args, **_kwargs):
        raise SemanticBackendError("mps is not available in this environment")

    def _fail_if_called(*_args, **_kwargs):
        raise AssertionError("an empty corpus must not reach the semantic backend")

    monkeypatch.setattr(semantic_module, "_resolve_semantic_device_request", _raise_unavailable)
    monkeypatch.setattr(analyzer_module, "run_semantic_analysis", _fail_if_called)
    empty_project = tmp_path / "empty"
    empty_project.mkdir()

    with pytest.raises(SemanticBackendError, match="mps is not available"):
        CodeAnalyzer(AnalyzerConfig(device="mps")).analyze(empty_project)

    # Opting into combined-mode semantic fallback degrades instead of raising.
    fallback_result = CodeAnalyzer(
        AnalyzerConfig(device="mps", allow_semantic_fallback=True)
    ).analyze(empty_project)
    assert fallback_result.units == []

    # A device that always has a CPU path stays torch-import-free.
    monkeypatch.setattr(semantic_module, "_resolve_semantic_device_request", _fail_if_called)
    assert CodeAnalyzer(AnalyzerConfig(device="auto")).analyze(empty_project).units == []


@pytest.mark.parametrize(
    "semantic_error",
    [
        ModuleNotFoundError("No module named 'sentence_transformers'"),
        SemanticBackendError("semantic backend mismatch"),
    ],
)
def test_semantic_failures_fall_back_when_traditional_enabled(
    tmp_path: Path, monkeypatch, caplog, semantic_error
) -> None:
    source = dedent(
        """
        def used():
            return 1

        def unused():
            return 2
        """
    ).strip()
    project = create_project(tmp_path, source)

    monkeypatch.setattr(
        analyzer_module,
        "run_semantic_analysis",
        _make_semantic_runner(error=semantic_error),
    )

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=True,
            run_semantic=True,
            allow_semantic_fallback=True,
            run_unused=False,
        )
    )

    result = analyzer.analyze(project)
    assert len(result.units) == 2
    assert result.semantic_duplicates == []
    assert "Retry with `codedupes check" in caplog.text


class _WhitespaceTokenizer:
    """Tokenizer stub whose token count is the whitespace-separated word count."""

    def encode(self, text, **_kwargs):
        return text.split()


class _ContextLimitedModel:
    """Model stub that rejects nothing but exposes a tiny context window."""

    max_seq_length = 20
    tokenizer = _WhitespaceTokenizer()

    def __init__(self) -> None:
        self.encoded: list[str] = []

    def encode(self, texts, **_kwargs):
        self.encoded.extend(texts)
        return np.array(
            [[0.0, 1.0] if "second_axis" in text else [1.0, 0.0] for text in texts],
            dtype=np.float32,
        )


_OVERFLOW_PROJECT_SOURCE = dedent(
    """
    def short_one(x):
        y = x + 1
        return y

    def short_two(x):
        total = x * 3
        print(total)
        return total

    def long_tail(x):
        words = "aa bb cc dd ee ff gg hh ii jj kk ll mm nn oo pp qq rr ss tt"
        return words
    """
).strip()


def test_over_context_units_are_skipped_with_a_diagnostic(tmp_path: Path, monkeypatch) -> None:
    project = create_project(tmp_path, _OVERFLOW_PROJECT_SOURCE)
    model = _ContextLimitedModel()
    monkeypatch.setattr(semantic_module, "get_model", lambda *args, **kwargs: model)

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=True,
            run_semantic=True,
            run_unused=False,
            min_semantic_statements=0,
            embedding_cache=False,
        )
    )
    result = analyzer.analyze(project)

    assert [unit.name for unit in result.units] == ["short_one", "short_two", "long_tail"]
    assert [
        (duplicate.unit_a.name, duplicate.unit_b.name) for duplicate in result.semantic_duplicates
    ] == [("short_one", "short_two")]
    assert [diagnostic.code for diagnostic in result.semantic_diagnostics] == [
        "semantic-context-overflow"
    ]
    diagnostic = result.semantic_diagnostics[0]
    assert "long_tail" in diagnostic.message
    assert diagnostic.severity == "warning"
    assert diagnostic.language == "python"
    assert not any("long_tail" in text for text in model.encoded)


def test_skipped_over_context_units_never_enter_the_embedding_cache(
    tmp_path: Path, monkeypatch
) -> None:
    project = create_project(tmp_path, _OVERFLOW_PROJECT_SOURCE)
    model = _ContextLimitedModel()
    monkeypatch.setattr(semantic_module, "get_model", lambda *args, **kwargs: model)

    def run() -> AnalysisResult:
        analyzer = CodeAnalyzer(
            AnalyzerConfig(
                run_traditional=False,
                run_semantic=True,
                run_unused=False,
                min_semantic_statements=0,
                embedding_cache=True,
            )
        )
        return analyzer.analyze(project)

    first = run()
    second = run()

    # A warm second run must not resurrect the skipped unit from the cache.
    for result in (first, second):
        assert [diagnostic.code for diagnostic in result.semantic_diagnostics] == [
            "semantic-context-overflow"
        ]
        assert [
            (duplicate.unit_a.name, duplicate.unit_b.name)
            for duplicate in result.semantic_duplicates
        ] == [("short_one", "short_two")]


def test_index_drops_over_context_units_and_keeps_search_rows_aligned(
    tmp_path: Path, monkeypatch
) -> None:
    source = dedent(
        """
        def long_tail(x):
            words = "aa bb cc dd ee ff gg hh ii jj kk ll mm nn oo pp qq rr ss tt"
            return words

        def wanted(x):
            y = x + 1
            return y

        def second_axis(x):
            z = x + 2
            return z
        """
    ).strip()
    project = create_project(tmp_path, source)
    model = _ContextLimitedModel()
    monkeypatch.setattr(semantic_module, "get_model", lambda *args, **kwargs: model)

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=False,
            run_semantic=True,
            run_unused=False,
            min_semantic_statements=0,
            embedding_cache=False,
        )
    )
    indexed = analyzer.index(project)
    results = analyzer.search("anything", top_k=1)

    assert indexed == 2
    assert [diagnostic.code for diagnostic in analyzer.semantic_diagnostics] == [
        "semantic-context-overflow"
    ]
    assert [unit.name for unit, _score in results] == ["wanted"]


def test_over_context_search_query_still_fails_hard(tmp_path: Path, monkeypatch) -> None:
    project = create_project(tmp_path, "def wanted(x):\n    y = x + 1\n    return y\n")
    model = _ContextLimitedModel()
    monkeypatch.setattr(semantic_module, "get_model", lambda *args, **kwargs: model)

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=False,
            run_semantic=True,
            run_unused=False,
            min_semantic_statements=0,
            embedding_cache=False,
        )
    )
    analyzer.index(project)

    with pytest.raises(semantic_module.SemanticInputTooLongError, match="search query"):
        analyzer.search(" ".join(["word"] * 40))


@pytest.mark.parametrize(
    "semantic_error",
    [
        ModuleNotFoundError("No module named 'sentence_transformers'"),
        SemanticBackendError("semantic backend mismatch"),
    ],
)
@pytest.mark.parametrize("run_unused", [False, True])
def test_semantic_failures_raise_when_semantic_required(
    tmp_path: Path, monkeypatch, semantic_error, run_unused
) -> None:
    source = "def only_func():\n    return 1\n"
    project = create_project(tmp_path, source)

    monkeypatch.setattr(
        analyzer_module,
        "run_semantic_analysis",
        _make_semantic_runner(error=semantic_error),
    )

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=False,
            run_semantic=True,
            run_unused=run_unused,
            min_semantic_statements=0,
        )
    )

    with pytest.raises(type(semantic_error)):
        analyzer.analyze(project)


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
        AnalyzerConfig(run_semantic=False, strict_revision_cache=True)


@pytest.mark.parametrize(
    ("config_overrides", "expected_values"),
    [
        pytest.param(
            {
                "device": "mps",
                "mps_fallback": False,
                "mps_memory_fraction": 0.8,
            },
            {
                "device": "mps",
                "mps_fallback": False,
                "mps_memory_fraction": 0.8,
            },
            id="device-controls",
        ),
        pytest.param(
            {"embedding_cache": False},
            {"use_cache": False},
            id="cache-control",
        ),
        pytest.param(
            {"strict_revision_cache": True},
            {"strict_revision_cache": True},
            id="strict-revision-cache-control",
        ),
    ],
)
def test_analyzer_passes_semantic_controls_to_index_and_query(
    tmp_path: Path,
    monkeypatch,
    config_overrides: dict[str, object],
    expected_values: dict[str, object],
) -> None:
    source = "def entry(x):\n    return x + 1\n"
    project = create_project(tmp_path, source)
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        analyzer_module,
        "run_semantic_analysis",
        _make_semantic_runner(capture=captured),
    )
    monkeypatch.setattr(
        semantic_module,
        "find_similar_to_query",
        _capture_query_runner(captured),
    )

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=False,
            run_semantic=True,
            run_unused=False,
            min_semantic_statements=0,
            **config_overrides,
        )
    )
    analyzer.analyze(project)
    analyzer.search("entry")

    for key, expected in expected_values.items():
        assert captured[key] == expected
        assert captured[f"query_{key}"] == expected
    assert captured["cache_scope"] == project
    assert captured["query_cache_scope"] == project


def test_analyzer_default_embedding_cache_enabled_and_scoped_to_analyzed_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = "def entry(x):\n    return x + 1\n"
    project = create_project(tmp_path, source)
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        analyzer_module,
        "run_semantic_analysis",
        _make_semantic_runner(capture=captured),
    )

    analyzer = CodeAnalyzer(
        AnalyzerConfig(
            run_traditional=False,
            run_semantic=True,
            run_unused=False,
            min_semantic_statements=0,
        )
    )
    analyzer.analyze(project)

    assert captured["use_cache"] is True
    assert captured["cache_scope"] == project
