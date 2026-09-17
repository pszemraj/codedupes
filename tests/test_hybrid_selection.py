"""Focused pure-score checks for hybrid gate selection."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts import sweep_hybrid_gates


def _project_and_measurement(
    project_id: str,
    language: str,
    pairs: list[tuple[str, float, float]],
) -> tuple[SimpleNamespace, dict]:
    """Build reviewed pair scores in the raw-measurement shape."""
    annotations = []
    rows = []
    units = []
    for index, (judgment, cosine, identifiers) in enumerate(pairs):
        a = f"{project_id}-a-{index}"
        b = f"{project_id}-b-{index}"
        annotations.append({"a": a, "b": b, "judgment": judgment})
        rows.append(
            {
                "a": a,
                "b": b,
                "cosine": cosine,
                "comparable": True,
                "identifier_jaccard": identifiers,
                "statement_ratio": 1.0,
                "traditional": [],
            }
        )
        units.extend(({"id": a, "language": language}, {"id": b, "language": language}))
    project = SimpleNamespace(
        id=project_id,
        spec={"languages": [language]},
        annotations={"pairs": annotations},
    )
    return project, {"metadata": {"model": "gte-modernbert-base"}, "units": units, "pairs": rows}


def test_joint_selection_can_trade_corroboration_for_promotion(monkeypatch: pytest.MonkeyPatch):
    """Joint F1 must not lock corroboration before testing promotion gates."""
    monkeypatch.setattr(sweep_hybrid_gates, "WEAK_GRID", (0.0, 0.4))
    monkeypatch.setattr(sweep_hybrid_gates, "RATIO_GRID", (0.0,))
    python, python_measurement = _project_and_measurement(
        "python",
        "python",
        [
            *(("positive", 0.85, 0.5) for _ in range(3)),
            *(("positive", 0.90, 0.1) for _ in range(3)),
            *(("positive", 0.70, 0.1) for _ in range(4)),
            *(("negative", 0.89, 0.1) for _ in range(3)),
        ],
    )
    c, c_measurement = _project_and_measurement(
        "c",
        "c",
        [
            *(("positive", 0.85, 0.5) for _ in range(2)),
            ("positive", 0.70, 0.1),
        ],
    )
    projects = [python, c]
    measurements = {"python": python_measurement, "c": c_measurement}
    admissions = {"python": 0.8, "c": 0.8}
    promotion_disabled_loose = sweep_hybrid_gates._pooled_metrics(
        projects, measurements, admissions, 0.0, 0.0, {"python": None, "c": None}
    )
    promotion_disabled_strict = sweep_hybrid_gates._pooled_metrics(
        projects, measurements, admissions, 0.4, 0.0, {"python": None, "c": None}
    )
    selected, options = sweep_hybrid_gates._select_joint(
        projects=projects,
        measurements=measurements,
        admissions=admissions,
    )

    # With promotion disabled, weak=0 accepts Python's three 0.89 negatives
    # (pooled F1=2/3), while weak=.4 initially loses the low-overlap positives.
    # A .90 promotion restores those positives without restoring the negatives.
    assert promotion_disabled_loose["f1"] == pytest.approx(2 / 3)
    assert promotion_disabled_strict["f1"] == pytest.approx(10 / 18)
    assert selected["weak_identifier_jaccard_min"] == 0.4
    assert selected["f1"] == pytest.approx(16 / 21)
    assert selected["high_gates"] == {"c": None, "python": 0.9}
    assert {language: option["high_gate"] for language, option in options.items()} == {
        "c": None,
        "python": 0.9,
    }


def test_joint_selection_uses_pooled_f1_instead_of_language_f1(monkeypatch: pytest.MonkeyPatch):
    """A language-local F1 winner need not maximize the pooled objective."""
    monkeypatch.setattr(sweep_hybrid_gates, "WEAK_GRID", (0.4,))
    monkeypatch.setattr(sweep_hybrid_gates, "RATIO_GRID", (0.0,))
    python, python_measurement = _project_and_measurement(
        "python",
        "python",
        [
            *(("positive", 0.85, 0.5) for _ in range(5)),
            *(("positive", 0.85, 0.1) for _ in range(5)),
            *(("negative", 0.85, 0.1) for _ in range(9)),
        ],
    )
    c, c_measurement = _project_and_measurement(
        "c",
        "c",
        [("positive", 0.85, 0.5), ("positive", 0.85, 0.5)],
    )
    projects = [python, c]
    measurements = {"python": python_measurement, "c": c_measurement}
    admissions = {"python": 0.8, "c": 0.8}
    python_options = {
        option["high_gate"]: option
        for option in sweep_hybrid_gates._promotion_options(
            projects, measurements, admissions, 0.4, 0.0
        )["python"]
    }

    # Python alone prefers the low gate: it recovers five positives despite
    # admitting nine negatives. Pooling with C correctly prefers the clean
    # disabled gate, whose lower false-positive count wins overall F1.
    assert python_options[0.8]["f1"] > python_options[None]["f1"]
    selected, _options = sweep_hybrid_gates._select_joint(projects, measurements, admissions)
    assert selected["high_gates"] == {"c": None, "python": None}
    assert selected["f1"] == pytest.approx(14 / 19)


@pytest.mark.parametrize("admission", [0.99, 1.0])
def test_promotion_sweep_accepts_admissions_through_one(admission: float):
    project, measurement = _project_and_measurement("python", "python", [("positive", 1.0, 0.1)])
    options = sweep_hybrid_gates._promotion_options(
        [project], {"python": measurement}, {"python": admission}, 0.4, 0.0
    )["python"]
    promoted = next(option for option in options if option["tp"] == 1)
    assert promoted["high_gate"] == admission


def test_promotion_sweep_can_separate_pairs_above_point_98():
    project, measurement = _project_and_measurement(
        "python", "python", [("positive", 0.995, 0.1), ("negative", 0.985, 0.1)]
    )
    options = sweep_hybrid_gates._promotion_options(
        [project], {"python": measurement}, {"python": 0.98}, 0.4, 0.0
    )["python"]
    best = max(options, key=lambda option: option["f1"])
    assert (best["high_gate"], best["tp"], best["fp"]) == (0.99, 1, 0)
