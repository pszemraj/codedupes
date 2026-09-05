from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from codedupes.analyzer import AnalyzerConfig, CodeAnalyzer
from codedupes.constants import DEFAULT_MODEL, DEFAULT_SEARCH_SEMANTIC_TASK
from codedupes.semantic import clear_model_cache, compute_embeddings_with_identity, get_model
from codedupes.semantic_profiles import SemanticModelProfile, list_supported_models


@pytest.mark.network
def test_network_smoke_default_model_encode() -> None:
    if os.getenv("CODEDUPES_SMOKE_NETWORK") != "1":
        pytest.skip("Set CODEDUPES_SMOKE_NETWORK=1 to enable network smoke tests.")

    clear_model_cache()
    model = get_model(DEFAULT_MODEL)
    embeddings = model.encode(
        ["def smoke_test(x):\n    return x + 1"],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    assert embeddings.shape[0] == 1


@pytest.mark.network
@pytest.mark.gpu
def test_gpu_smoke_default_model_encode() -> None:
    if os.getenv("CODEDUPES_SMOKE_GPU") != "1":
        pytest.skip("Set CODEDUPES_SMOKE_GPU=1 to enable GPU smoke tests.")

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available in this environment.")

    clear_model_cache()
    model = get_model(DEFAULT_MODEL, device="cuda")
    embeddings = model.encode(
        ["def gpu_smoke_test(x):\n    return x * 2"],
        convert_to_numpy=True,
        normalize_embeddings=True,
        device="cuda",
    )
    assert embeddings.shape[0] == 1


@pytest.mark.network
@pytest.mark.gpu
def test_gpu_smoke_cowsay_fixture_detects_labeled_clones() -> None:
    if os.getenv("CODEDUPES_SMOKE_GPU") != "1":
        pytest.skip("Set CODEDUPES_SMOKE_GPU=1 to enable GPU smoke tests.")

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available in this environment.")

    fixture = Path(__file__).resolve().parents[1] / "test_fixtures" / "cowsay_wasm" / "src"
    result = CodeAnalyzer(
        AnalyzerConfig(
            model_name=DEFAULT_MODEL,
            device="cuda",
            run_unused=False,
            embedding_cache=False,
            progress="never",
        )
    ).analyze(fixture)
    duplicate_names = {
        frozenset((duplicate.unit_a.qualified_name, duplicate.unit_b.qualified_name))
        for duplicate in result.hybrid_duplicates
    }

    assert {
        frozenset(("bubble.speech.make_borders", "bubble.thought.make_borders")),
        frozenset(("bubble.speech.render_bubble", "bubble.thought.render_bubble")),
        frozenset(("wrapping.fold.wrap", "wrapping.scanner.wrap")),
    } <= duplicate_names
    assert frozenset(("cow.render", "bubble.render")) not in duplicate_names
    assert frozenset(("bubble.render", "wrapping.wrap")) not in duplicate_names


@pytest.mark.network
@pytest.mark.parametrize(
    "profile",
    list_supported_models(),
    ids=lambda profile: profile.key,
)
def test_search_smoke_default_threshold_separates_relevant_from_noise(
    profile: SemanticModelProfile,
) -> None:
    if os.getenv("CODEDUPES_SMOKE_SEARCH") != "1":
        pytest.skip("Set CODEDUPES_SMOKE_SEARCH=1 to enable search smoke tests.")

    from codedupes.extractor import CodeExtractor
    from codedupes.semantic import find_similar_to_query

    fixture_dir = Path(__file__).resolve().parent.parent / "test_fixtures" / "search_probes"
    spec = json.loads((fixture_dir / "queries.json").read_text())
    units = list(CodeExtractor(fixture_dir).extract_from_file(fixture_dir / "probes.py"))
    assert {unit.name for unit in units} == {probe["expected"] for probe in spec["relevant"]}

    clear_model_cache()
    embeddings, identity = compute_embeddings_with_identity(
        units,
        model_name=profile.key,
        semantic_task=DEFAULT_SEARCH_SEMANTIC_TASK,
    )

    for probe in spec["relevant"]:
        results = find_similar_to_query(
            probe["query"],
            units,
            embeddings,
            model_name=profile.key,
            top_k=3,
            corpus_identity=identity,
        )
        names = [unit.name for unit, _score in results]
        assert probe["expected"] in names, (
            f"{profile.key}: {probe['query']!r} missed its target: {names}"
        )

    for query in spec["noise"]:
        results = find_similar_to_query(
            query,
            units,
            embeddings,
            model_name=profile.key,
            top_k=3,
            corpus_identity=identity,
        )
        hits = [(unit.name, score) for unit, score in results]
        assert not hits, f"{profile.key}: noise query {query!r} cleared the search floor: {hits}"
