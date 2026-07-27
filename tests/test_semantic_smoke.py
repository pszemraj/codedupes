from __future__ import annotations

import os
from pathlib import Path

import pytest

from codedupes.constants import DEFAULT_MODEL
from codedupes.models import CodeUnit, CodeUnitType
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
@pytest.mark.mps
def test_mps_smoke_default_model_encode(tmp_path: Path) -> None:
    if os.getenv("CODEDUPES_SMOKE_MPS") != "1":
        pytest.skip("Set CODEDUPES_SMOKE_MPS=1 to enable MPS smoke tests.")

    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        pytest.skip("PyTorch MPS is not available in this environment.")

    source_path = tmp_path / "mps_smoke.py"
    source = "def mps_smoke_test(x):\n    return x * 3\n"
    source_path.write_text(source)
    unit = CodeUnit(
        name="mps_smoke_test",
        qualified_name="mps_smoke.mps_smoke_test",
        unit_type=CodeUnitType.FUNCTION,
        file_path=source_path,
        lineno=1,
        end_lineno=2,
        source=source,
        is_public=True,
        is_exported=False,
    )

    clear_model_cache()
    embeddings = compute_embeddings([unit], model_name=DEFAULT_MODEL, device="mps", batch_size=1)
    model = get_model(DEFAULT_MODEL, device="mps")
    torch.mps.synchronize()

    assert embeddings.shape[0] == 1
    assert str(getattr(model, "device", "")).startswith("mps")
