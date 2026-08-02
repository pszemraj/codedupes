from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from codedupes.constants import DEFAULT_MODEL
from codedupes.semantic import clear_model_cache, compute_embeddings, get_model


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
def test_search_smoke_default_threshold_separates_relevant_from_noise() -> None:
    if os.getenv("CODEDUPES_SMOKE_SEARCH") != "1":
        pytest.skip("Set CODEDUPES_SMOKE_SEARCH=1 to enable search smoke tests.")

    from codedupes.extractor import CodeExtractor
    from codedupes.semantic import find_similar_to_query

    fixture_dir = Path(__file__).resolve().parent.parent / "test_fixtures" / "search_probes"
    spec = json.loads((fixture_dir / "queries.json").read_text())
    units = list(CodeExtractor(fixture_dir).extract_from_file(fixture_dir / "probes.py"))
    assert {unit.name for unit in units} == {probe["expected"] for probe in spec["relevant"]}

    clear_model_cache()
    embeddings = compute_embeddings(units, model_name=DEFAULT_MODEL)

    for probe in spec["relevant"]:
        results = find_similar_to_query(probe["query"], units, embeddings, top_k=3)
        names = [unit.name for unit, _score in results]
        assert probe["expected"] in names, f"{probe['query']!r} missed its target: {names}"

    # The default model is revision-unpinned, so a future upstream revision could
    # shift scores across the 0.50 floor; a failure here means the search default
    # needs recalibration, which is exactly what this opt-in smoke test is for.
    for query in spec["noise"]:
        results = find_similar_to_query(query, units, embeddings, top_k=3)
        hits = [(unit.name, score) for unit, score in results]
        assert not hits, f"noise query {query!r} cleared the search floor: {hits}"
