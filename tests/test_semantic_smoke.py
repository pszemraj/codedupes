from __future__ import annotations

import os

import pytest

from codedupes.constants import DEFAULT_MODEL
from codedupes.semantic import clear_model_cache, get_model


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
    model = get_model(DEFAULT_MODEL)
    embeddings = model.encode(
        ["def gpu_smoke_test(x):\n    return x * 2"],
        convert_to_numpy=True,
        normalize_embeddings=True,
        device="cuda",
    )
    assert embeddings.shape[0] == 1
