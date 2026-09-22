"""Embedding validation and the pairwise semantic-duplicate scan."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from codedupes import semantic
from codedupes.models import CodeUnit, CodeUnitType
from codedupes.pairs import ordered_pair_key
from codedupes.semantic import (
    find_semantic_duplicates,
    find_similar_to_query,
    get_code_unit_statement_count,
    run_semantic_analysis,
)
from tests.conftest import extract_arithmetic_units, extract_units
from tests.semantic_helpers import FakeModel


def test_run_semantic_analysis_with_mock_model(tmp_path, monkeypatch):
    units = extract_arithmetic_units(tmp_path)
    fake = FakeModel()
    monkeypatch.setattr(semantic, "_model", None)
    monkeypatch.setattr(semantic, "get_model", lambda *args, **kwargs: fake)

    _, duplicates = run_semantic_analysis(units, threshold=0.9)

    assert len(duplicates) == 1
    assert duplicates[0].method == "semantic"
    assert duplicates[0].similarity > 0.9


def test_code_unit_statement_count_trusts_the_unit(tmp_path: Path) -> None:
    """Extraction stores the count; a unit built without one measures as empty."""
    unit = extract_arithmetic_units(tmp_path)[0]
    assert unit.statement_count == 1
    assert get_code_unit_statement_count(unit) == 1

    unit.statement_count = None
    assert get_code_unit_statement_count(unit) == 0


def test_decorated_method_statement_count_excludes_the_decorator(tmp_path: Path) -> None:
    units = extract_units(
        tmp_path,
        """
        class Widget:
            @property
            def area(self):
                width = self.width
                height = self.height
                scale = self.scale
                return width * height * scale

            def perimeter(self):
                width = self.width
                height = self.height
                scale = self.scale
                return (width + height) * 2 * scale
        """,
        include_private=True,
    )
    by_name = {unit.name: unit for unit in units}

    assert by_name["area"].source.startswith("@property")
    assert get_code_unit_statement_count(by_name["area"]) == 4
    assert get_code_unit_statement_count(by_name["perimeter"]) == 4


@pytest.mark.parametrize(
    "embeddings",
    [
        pytest.param(np.ones((1, 2), dtype=np.float32), id="short"),
        pytest.param(np.ones((3, 2), dtype=np.float32), id="long"),
        pytest.param(np.ones(2, dtype=np.float32), id="non-2d"),
    ],
)
def test_precomputed_embeddings_require_2d_row_alignment(
    tmp_path: Path, monkeypatch, embeddings: np.ndarray
) -> None:
    units = extract_arithmetic_units(tmp_path)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("invalid precomputed embeddings must fail before model loading")

    monkeypatch.setattr(semantic, "get_model", fail_if_called)

    with pytest.raises(
        ValueError, match=r"embeddings must (be a 2D matrix|contain one row per unit)"
    ):
        find_semantic_duplicates(units, embeddings, threshold=0.0)
    with pytest.raises(
        ValueError, match=r"embeddings must (be a 2D matrix|contain one row per unit)"
    ):
        find_similar_to_query(
            "find addition",
            units,
            embeddings,
            threshold=0.0,
            device="cpu",
            use_cache=False,
        )


@pytest.mark.parametrize(
    ("embeddings", "message"),
    [
        pytest.param(
            np.array([[np.nan, 0.0], [1.0, 0.0]], dtype=np.float32),
            "NaN or infinity",
            id="nan",
        ),
        pytest.param(
            np.array([[np.inf, 0.0], [1.0, 0.0]], dtype=np.float32),
            "NaN or infinity",
            id="infinity",
        ),
        pytest.param(
            np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
            "zero or invalid vector",
            id="zero-row",
        ),
        pytest.param(
            np.array([[1.0 + 100.0j, 0.0], [1.0 + 0.0j, 0.0]], dtype=np.complex64),
            "real-valued vectors",
            id="complex",
        ),
        pytest.param(np.empty((2, 0), dtype=np.float32), "zero columns", id="zero-width"),
    ],
)
def test_precomputed_embeddings_reject_invalid_rows_before_similarity_or_model_loading(
    tmp_path: Path, monkeypatch, embeddings: np.ndarray, message: str
) -> None:
    units = extract_arithmetic_units(tmp_path)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("invalid precomputed embeddings must fail before model loading")

    monkeypatch.setattr(semantic, "get_model", fail_if_called)

    with pytest.raises(ValueError, match=message):
        find_semantic_duplicates(units, embeddings, threshold=0.0)
    with pytest.raises(ValueError, match=message):
        find_similar_to_query(
            "find addition",
            units,
            embeddings,
            threshold=0.0,
            device="cpu",
            use_cache=False,
        )


def test_direct_embeddings_are_normalized_before_duplicate_scoring(tmp_path: Path) -> None:
    units = extract_arithmetic_units(tmp_path)
    embeddings = np.array([[5.0, 0.0], [1.0, 0.5]], dtype=np.float32)

    # The raw dot product is 5.0, but the true cosine is ~0.894. The pair must
    # not clear a 0.95 duplicate gate merely because a caller supplied scaled rows.
    assert find_semantic_duplicates(units, embeddings, threshold=0.95) == []


def test_canonical_precomputed_embeddings_reuse_storage_with_bounded_validation(
    tmp_path: Path, monkeypatch
) -> None:
    units = extract_arithmetic_units(tmp_path)
    units = [units[index % len(units)] for index in range(2050)]
    embeddings = np.zeros((len(units), 2), dtype=np.float32)
    embeddings[:, 0] = 1.0
    observed_rows = []
    original = semantic.canonicalize_embeddings

    def traced_canonicalize(values, *, expected_rows, expected_dim=None):
        observed_rows.append(expected_rows)
        return original(values, expected_rows=expected_rows, expected_dim=expected_dim)

    monkeypatch.setattr(semantic, "canonicalize_embeddings", traced_canonicalize)

    validated = semantic._validate_precomputed_embeddings(units, embeddings)

    assert validated is embeddings
    assert observed_rows
    assert max(observed_rows) <= semantic._PRECOMPUTED_VALIDATION_BLOCK_ROWS


def test_fresh_embedding_normalization_bounds_float64_working_rows(monkeypatch) -> None:
    row_count = semantic._PRECOMPUTED_VALIDATION_BLOCK_ROWS * 2 + 3
    embeddings = np.ones((row_count, 4), dtype=np.float32)
    converted_rows: list[int] = []
    original_asarray = semantic.np.asarray

    def traced_asarray(values, dtype=None, *args, **kwargs):
        shape = getattr(values, "shape", ())
        if dtype is not None and np.dtype(dtype) == np.dtype(np.float64) and shape:
            converted_rows.append(shape[0])
        return original_asarray(values, dtype, *args, **kwargs)

    monkeypatch.setattr(semantic.np, "asarray", traced_asarray)

    canonical = semantic.canonicalize_embeddings(embeddings, expected_rows=row_count)

    assert canonical.shape == embeddings.shape
    assert converted_rows
    assert max(converted_rows) <= semantic._PRECOMPUTED_VALIDATION_BLOCK_ROWS
    np.testing.assert_allclose(np.linalg.norm(canonical, axis=1), 1.0)


def test_near_unit_direct_embeddings_are_still_normalized_before_scoring(tmp_path: Path) -> None:
    units = extract_arithmetic_units(tmp_path)
    second_component = math.sqrt(1.0 - 0.9**2)
    embeddings = np.array([[1.0000005, 0.0], [0.9, second_component]], dtype=np.float32)

    # A tolerance-based unit-vector shortcut would preserve the first row's
    # scale and admit this pair just above the caller's exact decision boundary.
    assert find_semantic_duplicates(units, embeddings, threshold=0.9000003) == []


@pytest.mark.parametrize(
    "scale",
    [
        pytest.param(np.float32(3e38), id="near-float32-maximum"),
        pytest.param(np.nextafter(np.float32(0), np.float32(1)), id="float32-subnormal"),
    ],
)
def test_direct_embeddings_stably_normalize_finite_float32_extremes(
    tmp_path: Path, scale: np.float32
) -> None:
    units = extract_arithmetic_units(tmp_path)
    embeddings = np.array([[scale, scale], [scale, 0.0]], dtype=np.float32)

    duplicates = find_semantic_duplicates(units, embeddings, threshold=0.7)

    assert len(duplicates) == 1
    assert duplicates[0].similarity == pytest.approx(1 / math.sqrt(2), abs=1e-6)


def test_direct_embedding_apis_return_to_base_ndarray_semantics(
    tmp_path: Path, monkeypatch
) -> None:
    units = extract_arithmetic_units(tmp_path)
    embeddings = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32).view(np.matrix)

    assert len(find_semantic_duplicates(units, embeddings, threshold=0.9)) == 1

    class QueryModel:
        def encode(self, texts, **kwargs):
            return np.array([[1.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr(semantic, "get_model", lambda *args, **kwargs: QueryModel())
    results = find_similar_to_query(
        "find addition",
        units,
        embeddings,
        threshold=-1.0,
        device="cpu",
        use_cache=False,
    )
    assert {unit.uid for unit, _score in results} == {unit.uid for unit in units}


def test_semantic_pair_scores_bound_float32_cosine_overshoot(tmp_path: Path) -> None:
    units = extract_arithmetic_units(tmp_path)
    vector = [-1.2083186, -0.004454133, 0.65647495]
    embeddings = semantic.canonicalize_embeddings([vector, vector], expected_rows=2)

    duplicates = find_semantic_duplicates(units, embeddings, threshold=1.0)

    assert len(duplicates) == 1
    assert duplicates[0].similarity == 1.0


@pytest.mark.parametrize("threshold", [0.82, 0.78, 0.70, 0.90])
def test_find_semantic_duplicates_rechecks_threshold_after_numpy_prefilter(
    tmp_path: Path, threshold: float
) -> None:
    units = extract_arithmetic_units(tmp_path)
    rounded_down = np.float32(threshold)
    assert float(rounded_down) < threshold
    embeddings = np.array(
        [[1.0, 0.0], [rounded_down, math.sqrt(1.0 - rounded_down**2)]],
        dtype=np.float32,
    )

    duplicates = find_semantic_duplicates(units, embeddings, threshold=threshold)

    assert duplicates == []


def test_find_semantic_duplicates_skips_incompatible_unit_types(tmp_path: Path) -> None:
    source_path = tmp_path / "sample.py"
    source_path.write_text("class C:\n    pass\n\ndef f():\n    return 1\n")

    class_unit = CodeUnit(
        name="C",
        qualified_name="sample.C",
        unit_type=CodeUnitType.CLASS,
        file_path=source_path,
        lineno=1,
        end_lineno=2,
        source="class C:\n    pass",
        is_public=True,
        is_exported=False,
    )
    function_unit = CodeUnit(
        name="f",
        qualified_name="sample.f",
        unit_type=CodeUnitType.FUNCTION,
        file_path=source_path,
        lineno=4,
        end_lineno=5,
        source="def f():\n    return 1",
        is_public=True,
        is_exported=False,
    )
    embeddings = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )

    duplicates = find_semantic_duplicates(
        units=[class_unit, function_unit],
        embeddings=embeddings,
        threshold=0.9,
    )

    assert duplicates == []


def test_find_semantic_duplicates_cross_language_requires_opt_in(tmp_path: Path) -> None:
    python_path = tmp_path / "sample.py"
    python_path.write_text("def f():\n    return 1\n")
    rust_path = tmp_path / "sample.rs"
    rust_path.write_text("fn f() -> i64 { 1 }\n")

    python_unit = CodeUnit(
        name="f",
        qualified_name="sample.f",
        unit_type=CodeUnitType.FUNCTION,
        file_path=python_path,
        lineno=1,
        end_lineno=2,
        source="def f():\n    return 1",
        is_public=True,
        is_exported=False,
        language="python",
    )
    rust_unit = CodeUnit(
        name="f",
        qualified_name="sample::f",
        unit_type=CodeUnitType.FUNCTION,
        file_path=rust_path,
        lineno=1,
        end_lineno=1,
        source="fn f() -> i64 { 1 }",
        is_public=True,
        is_exported=False,
        language="rust",
    )
    embeddings = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )

    same_language_only = find_semantic_duplicates(
        units=[python_unit, rust_unit],
        embeddings=embeddings,
        threshold=0.9,
    )
    assert same_language_only == []

    cross = find_semantic_duplicates(
        units=[python_unit, rust_unit],
        embeddings=embeddings,
        threshold=0.9,
        cross_language=True,
    )
    assert [(pair.unit_a.language, pair.unit_b.language) for pair in cross] == [("python", "rust")]


def _scan_fixture() -> tuple[list[CodeUnit], np.ndarray]:
    """Build a deterministic two-language corpus for the pairwise-scan fuzz.

    Every component is +/-0.25 across 16 dimensions, so rows are exactly
    unit-norm and every dot product is an exact multiple of 1/16. Blocking a
    matrix multiply therefore cannot perturb a single similarity, which lets the
    reference below compare bit-exactly, and the resulting ties make the scan's
    emission order observable through the stable final sort.

    :return: Units and their row-aligned embedding matrix.
    """
    count, dimensions = 1040, 16
    rng = np.random.default_rng(20260825)
    embeddings = (
        rng.integers(0, 2, size=(count, dimensions)).astype(np.float32) * 2.0 - 1.0
    ) * 0.25

    # A deterministic all-positive row exercises the dense positive-score
    # branch without violating the direct-embedding finite-row contract.
    embeddings[0] = 0.25

    # Perfect-similarity rows for the three post-gate filters: an overlapping
    # same-file pair, a surviving cross-file twin, and a class/function pair.
    embeddings[9] = embeddings[8]
    embeddings[13] = embeddings[12]
    embeddings[17] = embeddings[16]

    units: list[CodeUnit] = []
    for index in range(count):
        language = "python" if index % 4 < 2 else "rust"
        if (index + 1) % 17 == 0:
            unit_type = CodeUnitType.CLASS
        elif index % 5 == 0:
            unit_type = CodeUnitType.METHOD
        else:
            unit_type = CodeUnitType.FUNCTION
        units.append(
            CodeUnit(
                name=f"unit_{index}",
                qualified_name=f"mod_{index}.unit_{index}",
                unit_type=unit_type,
                file_path=Path(f"mod_{index}.{'py' if language == 'python' else 'rs'}"),
                lineno=1,
                end_lineno=4,
                source=f"def unit_{index}(): ...",
                language=language,
                start_byte=0,
                end_byte=40,
            )
        )

    units[9].file_path = units[8].file_path
    units[9].start_byte = 20
    units[9].end_byte = 60
    return units, embeddings


def _reference_semantic_duplicates(
    units: list[CodeUnit],
    embeddings: np.ndarray,
    threshold: float,
    *,
    exclude_exact: set[tuple[str, str]] | None = None,
    cross_language: bool = False,
    language_thresholds: dict[str, float] | None = None,
) -> list[tuple[tuple[str, str], float]]:
    """Restate the documented duplicate-scan semantics as a naive O(N^2) loop.

    :param units: Candidate units row-aligned with ``embeddings``.
    :param embeddings: Embedding matrix.
    :param threshold: Fallback gate for languages without a calibrated gate.
    :param exclude_exact: Pairs the scan must not report.
    :param cross_language: Whether to scan one mixed group instead of per-language groups.
    :param language_thresholds: Per-language duplicate gates.
    :return: ``(ordered pair key, similarity)`` in the order the scan must report.
    """
    excluded = exclude_exact or set()
    gates = dict(language_thresholds or {})
    similarity_matrix = embeddings @ embeddings.T

    if cross_language:
        groups = {"*": list(range(len(units)))}
    else:
        groups = {}
        for index, unit in enumerate(units):
            groups.setdefault(unit.language, []).append(index)

    reported: list[tuple[tuple[str, str], float]] = []
    for language, indices in groups.items():
        if cross_language:
            group_gate = min(gates.get(units[index].language, threshold) for index in indices)
        else:
            group_gate = gates.get(language, threshold)
        for position, index_a in enumerate(indices):
            unit_a = units[index_a]
            row = similarity_matrix[index_a].tolist()
            for index_b in indices[position + 1 :]:
                similarity = row[index_b]
                if not math.isfinite(similarity) or similarity < group_gate:
                    continue
                unit_b = units[index_b]
                if cross_language and similarity < min(
                    gates.get(unit_a.language, threshold),
                    gates.get(unit_b.language, threshold),
                ):
                    continue
                kinds = {unit_a.unit_type, unit_b.unit_type}
                if len(kinds) > 1 and kinds != {CodeUnitType.FUNCTION, CodeUnitType.METHOD}:
                    continue
                # Every fixture unit has a real byte range, so overlap is the
                # same-file byte-interval test.
                if (
                    unit_a.file_path == unit_b.file_path
                    and unit_a.start_byte < unit_b.end_byte
                    and unit_b.start_byte < unit_a.end_byte
                ):
                    continue
                key = ordered_pair_key(unit_a, unit_b)
                if key in excluded:
                    continue
                reported.append((key, similarity))
    reported.sort(key=lambda entry: entry[1], reverse=True)
    return reported


def _pair_view(duplicates) -> list[tuple[tuple[str, str], float]]:
    """Reduce reported duplicates to comparable ``(pair key, similarity)`` entries.

    :param duplicates: Duplicate pairs as reported by the scan.
    :return: Pair keys with similarities, in report order.
    """
    return [(ordered_pair_key(pair.unit_a, pair.unit_b), pair.similarity) for pair in duplicates]


def _random_float_scan_fixture() -> tuple[list[CodeUnit], np.ndarray]:
    """Build a random-float fixture that exposes width-dependent BLAS scores.

    :return: Units and their row-aligned embedding matrix.
    """
    count, dimensions = 520, 768
    rng = np.random.default_rng(20260826)
    embeddings = rng.normal(size=(count, dimensions)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Make the final partial row block mutually similar, leaving a small,
    # deterministic above-gate result set whose scores still use all 768 dims.
    base = rng.normal(size=dimensions).astype(np.float32)
    base /= np.linalg.norm(base)
    embeddings[-20:] = base + rng.normal(scale=0.005, size=(20, dimensions)).astype(np.float32)
    embeddings[-20:] /= np.linalg.norm(embeddings[-20:], axis=1, keepdims=True)

    units = [
        CodeUnit(
            name=f"random_{index}",
            qualified_name=f"random_{index}",
            unit_type=CodeUnitType.FUNCTION,
            file_path=Path(f"random_{index}.py"),
            lineno=1,
            end_lineno=2,
            source=f"def random_{index}(): ...",
            language="python",
            start_byte=0,
            end_byte=20,
        )
        for index in range(count)
    ]
    return units, embeddings


def test_vectorized_pair_scan_matches_naive_reference() -> None:
    # The scan multiplies full-width row chunks and thresholds column blocks in
    # numpy; a wrong column offset or mis-ordered candidate walk only shows up
    # past the 500-row chunk boundary, which this 520-per-language corpus crosses
    # in every mode.
    units, embeddings = _scan_fixture()
    gates = {"python": 0.875, "rust": 0.75}

    same_language = find_semantic_duplicates(
        units, embeddings, threshold=0.75, language_thresholds=gates
    )
    expected = _reference_semantic_duplicates(units, embeddings, 0.75, language_thresholds=gates)
    assert _pair_view(same_language) == expected
    assert len(expected) > 100

    reported_keys = {key for key, _ in _pair_view(same_language)}
    assert ordered_pair_key(units[12], units[13]) in reported_keys
    assert ordered_pair_key(units[8], units[9]) not in reported_keys
    assert ordered_pair_key(units[16], units[17]) not in reported_keys
    assert ordered_pair_key(units[0], units[1]) not in reported_keys

    cross = find_semantic_duplicates(
        units, embeddings, threshold=0.9, cross_language=True, language_thresholds=gates
    )
    cross_expected = _reference_semantic_duplicates(
        units, embeddings, 0.9, cross_language=True, language_thresholds=gates
    )
    assert _pair_view(cross) == cross_expected
    assert any(pair.unit_a.language != pair.unit_b.language for pair in cross)

    excluded = {key for key, _ in expected[::37]}
    filtered = find_semantic_duplicates(
        units, embeddings, threshold=0.75, exclude_exact=excluded, language_thresholds=gates
    )
    filtered_expected = _reference_semantic_duplicates(
        units, embeddings, 0.75, exclude_exact=excluded, language_thresholds=gates
    )
    assert _pair_view(filtered) == filtered_expected
    assert len(filtered) == len(expected) - len(excluded)


def test_vectorized_pair_scan_preserves_full_width_float32_scores_and_order() -> None:
    units, embeddings = _random_float_scan_fixture()
    threshold = 0.97

    # This is the product shape used by the original scalar candidate walk.
    canonical = semantic.canonicalize_embeddings(embeddings, expected_rows=len(units))
    full_width_scores = canonical[500:] @ canonical.T

    expected: list[tuple[tuple[str, str], float]] = []
    for local_idx, group_i in enumerate(range(500, 520)):
        for group_j in range(group_i + 1, 520):
            similarity = float(full_width_scores[local_idx, group_j])
            if similarity >= threshold:
                expected.append((ordered_pair_key(units[group_i], units[group_j]), similarity))
    expected.sort(key=lambda entry: entry[1], reverse=True)
    assert len(expected) == 190

    reported = find_semantic_duplicates(units, embeddings, threshold=threshold)

    assert _pair_view(reported) == expected


def test_vectorized_pair_scan_bounds_candidate_extraction_and_masks_lower_triangle(
    monkeypatch,
) -> None:
    units, _ = _random_float_scan_fixture()
    shared_path = Path("overlapping.py")
    for unit in units:
        unit.file_path = shared_path
    embeddings = np.ones((len(units), 1), dtype=np.float32)

    nonzero_calls: list[tuple[tuple[int, ...], int]] = []
    original_nonzero = np.nonzero

    def recording_nonzero(mask):
        result = original_nonzero(mask)
        nonzero_calls.append((mask.shape, len(result[0])))
        return result

    monkeypatch.setattr(semantic.np, "nonzero", recording_nonzero)

    # Every score clears the gate, while overlap filtering keeps the returned
    # result empty so this test measures intermediate candidate batching only.
    assert find_semantic_duplicates(units, embeddings, threshold=0.0) == []

    block_size = semantic._PAIRWISE_SCAN_BLOCK_SIZE
    assert nonzero_calls == [
        ((500, 500), 500 * 499 // 2),
        ((500, 20), 500 * 20),
        ((20, 20), 20 * 19 // 2),
    ]
    assert max(rows * columns for (rows, columns), _ in nonzero_calls) <= block_size**2
