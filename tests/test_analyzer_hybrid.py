"""Hybrid synthesis: tiers, corroboration, promotion, and ordering."""

from __future__ import annotations

from pathlib import Path

import pytest

from codedupes import analyzer as analyzer_module
from codedupes.models import CodeUnit, DuplicatePair
from tests.conftest import make_code_unit


def test_hybrid_synthesis_exact_only_included(tmp_path: Path) -> None:
    unit_a = make_code_unit(tmp_path, name="a", source="def a(x):\n    return x + 1\n", lineno=1)
    unit_b = make_code_unit(tmp_path, name="b", source="def b(y):\n    return y + 1\n", lineno=5)
    traditional = [
        DuplicatePair(unit_a=unit_a, unit_b=unit_b, similarity=1.0, method="structural_hash")
    ]

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


# The corroborator mechanism tests below pin the identifier/size thresholds
# explicitly: the shipped split is calibrated per model profile and asserted
# end-to-end by ``test_analyzer_applies_the_profile_hybrid_split``.
_MECHANISM_SPLIT = {"weak_identifier_jaccard_min": 0.20, "statement_ratio_min": 0.35}


def test_hybrid_synthesis_semantic_only_corroboration_sets_tier(tmp_path: Path) -> None:
    # One shared identifier out of five clears the 0.20 overlap floor exactly.
    unit_a = make_code_unit(
        tmp_path,
        name="a",
        source="def alpha(v):\n    z = v + 1\n    return z\n",
        lineno=1,
        identifiers=frozenset({"alpha", "v", "z"}),
        statement_count=2,
    )
    unit_b = make_code_unit(
        tmp_path,
        name="b",
        source="def beta(v):\n    q = v + 2\n    return q\n",
        lineno=6,
        identifiers=frozenset({"beta", "v", "q"}),
        statement_count=2,
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
        **_MECHANISM_SPLIT,
    )
    assert len(hybrid) == 1
    assert hybrid[0].tier == "semantic_high_confidence"
    assert hybrid[0].confidence == pytest.approx(0.45 + (0.55 * 0.75))

    weak_sources_a = make_code_unit(
        tmp_path,
        name="c",
        source="def c(a):\n    x = a + 1\n    y = x + 1\n    z = y + 1\n    return z\n",
        lineno=12,
        identifiers=frozenset({"c", "a", "x", "y", "z"}),
        statement_count=4,
    )
    weak_sources_b = make_code_unit(
        tmp_path,
        name="d",
        source="def d(v):\n    return v\n",
        lineno=20,
        identifiers=frozenset({"d", "v"}),
        statement_count=1,
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
        **_MECHANISM_SPLIT,
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
        identifiers=frozenset({"review_a", "a", "x", "y", "z"}),
        statement_count=4,
    )
    review_b = make_code_unit(
        tmp_path,
        name="review_b",
        source="def review_b(v):\n    return v\n",
        lineno=12,
        identifiers=frozenset({"review_b", "v"}),
        statement_count=1,
    )
    confirmed_a = make_code_unit(
        tmp_path,
        name="confirmed_a",
        source="def confirmed_a(x):\n    return x + 1\n",
        lineno=20,
        identifiers=frozenset({"confirmed_a", "x"}),
        statement_count=1,
    )
    confirmed_b = make_code_unit(
        tmp_path,
        name="confirmed_b",
        source="def confirmed_b(y):\n    return y + 1\n",
        lineno=26,
        identifiers=frozenset({"confirmed_b", "y"}),
        statement_count=1,
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
        **_MECHANISM_SPLIT,
    )

    assert [duplicate.tier for duplicate in hybrid] == ["hybrid_confirmed", "semantic_review"]
    assert hybrid[0].confidence > hybrid[1].confidence


def test_exact_pairs_outrank_perfect_score_near_and_confirmed_pairs(tmp_path: Path) -> None:
    # traditional_near and hybrid_confirmed both reach confidence 1.0 at
    # perfect scores; their real similarities used to sort ahead of an exact
    # pair's None scores. Exact must lead regardless of uid order, so the
    # exact units are named to sort last.
    def unit(name: str, lineno: int) -> CodeUnit:
        return make_code_unit(
            tmp_path,
            name=name,
            source=f"def {name}(a):\n    b = a + 1\n    c = b + 1\n    return c\n",
            lineno=lineno,
            identifiers=frozenset({name, "a", "b", "c"}),
            statement_count=3,
        )

    near_a, near_b = unit("a_near_a", 1), unit("a_near_b", 10)
    confirmed_a, confirmed_b = unit("b_confirmed_a", 20), unit("b_confirmed_b", 30)
    exact_a, exact_b = unit("z_exact_a", 40), unit("z_exact_b", 50)

    hybrid = analyzer_module._synthesize_hybrid_duplicates(
        [
            DuplicatePair(unit_a=near_a, unit_b=near_b, similarity=1.0, method="jaccard"),
            DuplicatePair(unit_a=confirmed_a, unit_b=confirmed_b, similarity=1.0, method="jaccard"),
            DuplicatePair(unit_a=exact_a, unit_b=exact_b, similarity=1.0, method="token_hash"),
        ],
        [DuplicatePair(unit_a=confirmed_a, unit_b=confirmed_b, similarity=1.0, method="semantic")],
        jaccard_threshold=0.85,
        **_MECHANISM_SPLIT,
    )

    assert [duplicate.tier for duplicate in hybrid] == [
        "exact",
        "hybrid_confirmed",
        "traditional_near",
    ]
    assert [duplicate.confidence for duplicate in hybrid] == [1.0, 1.0, 1.0]
    assert hybrid[0].exact_method == "token_hash"
    assert [duplicate.exact_method for duplicate in hybrid[1:]] == [None, None]


def _alpha_renamed_pair(tmp_path: Path, similarity: float) -> list[DuplicatePair]:
    """Build a same-shape pair whose identifier sets are fully disjoint.

    :param tmp_path: Test directory the units' file path points into.
    :param similarity: Semantic similarity to record on the pair.
    :return: One semantic duplicate pair with no lexical overlap.
    """
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
        identifiers=frozenset(
            {"collect_total", "records", "accepted", "record", "enabled", "amount", "value"}
        ),
        statement_count=3,
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
        identifiers=frozenset(
            {"measure_sum", "entries", "chosen", "entry", "ready", "result", "weight"}
        ),
        statement_count=3,
    )
    return [DuplicatePair(unit_a=unit_a, unit_b=unit_b, similarity=similarity, method="semantic")]


def test_hybrid_synthesis_publishes_alpha_renamed_semantic_pair(tmp_path: Path) -> None:
    hybrid = analyzer_module._synthesize_hybrid_duplicates(
        [],
        _alpha_renamed_pair(tmp_path, 0.91),
        jaccard_threshold=0.85,
        **_MECHANISM_SPLIT,
    )

    assert len(hybrid) == 1
    assert hybrid[0].tier == "semantic_review"
    assert hybrid[0].weak_identifier_jaccard == 0.0
    assert hybrid[0].statement_count_ratio == 1.0


@pytest.mark.parametrize(
    ("gates", "expected_tier"),
    [
        (None, "semantic_review"),
        ({}, "semantic_review"),
        ({"python": 0.92}, "semantic_review"),
        ({"python": 0.91}, "semantic_high_confidence"),
        ({"rust": 0.50}, "semantic_review"),
    ],
)
def test_hybrid_synthesis_promotes_uncorroborated_pair_only_above_its_high_gate(
    tmp_path: Path, gates, expected_tier
) -> None:
    hybrid = analyzer_module._synthesize_hybrid_duplicates(
        [],
        _alpha_renamed_pair(tmp_path, 0.91),
        jaccard_threshold=0.85,
        semantic_high_gates=gates,
        **_MECHANISM_SPLIT,
    )

    assert [pair.tier for pair in hybrid] == [expected_tier]
    assert hybrid[0].weak_identifier_jaccard == 0.0
    if expected_tier == "semantic_high_confidence":
        assert hybrid[0].confidence == pytest.approx(0.45 + 0.55 * 0.91)
    else:
        assert hybrid[0].confidence == pytest.approx(0.40 + 0.45 * 0.91)


def test_hybrid_synthesis_cross_language_promotion_uses_the_stricter_gate(
    tmp_path: Path,
) -> None:
    semantic = _alpha_renamed_pair(tmp_path, 0.93)
    semantic[0].unit_b.language = "rust"
    gates = {"python": 0.90, "rust": 0.95}

    review = analyzer_module._synthesize_hybrid_duplicates(
        [], semantic, jaccard_threshold=0.85, semantic_high_gates=gates, **_MECHANISM_SPLIT
    )
    promoted = analyzer_module._synthesize_hybrid_duplicates(
        [],
        semantic,
        jaccard_threshold=0.85,
        semantic_high_gates={"python": 0.90, "rust": 0.93},
        **_MECHANISM_SPLIT,
    )

    assert review[0].tier == "semantic_review"
    assert promoted[0].tier == "semantic_high_confidence"
