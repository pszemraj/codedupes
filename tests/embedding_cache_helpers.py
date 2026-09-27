"""Shared builders and runners for the test_embedding_cache test modules."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from codedupes import embedding_cache, semantic
from codedupes.models import CodeUnit
from tests.conftest import extract_units

REVISION_1 = "1" * 40
REVISION_2 = "2" * 40

FIVE_FUNCTION_SOURCE = """
def alpha(x):
    return x + 1

def beta(x):
    return x + 2

def gamma(x):
    return x + 3

def delta(x):
    return x + 4

def epsilon(x):
    return x + 5
"""


def vector_for_text(text: str, dim: int = 4) -> np.ndarray:
    """Derive a deterministic unit-normalized float32 vector from a text's MD5 digest."""
    digest = hashlib.md5(text.encode()).digest()
    raw = np.array([float(b) + 1.0 for b in digest[:dim]], dtype=np.float32)
    return (raw / np.linalg.norm(raw)).astype(np.float32)


class CountingModel:
    """Deterministic fake embedding model that records every encode call."""

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.encode_calls: list[list[str]] = []
        self.prompts_seen: list[str | None] = []

    def _record_encode_call(self, texts, **kwargs) -> list[str]:
        text_list = list(texts)
        self.encode_calls.append(text_list)
        self.prompts_seen.append(kwargs.get("prompt"))
        return text_list

    def encode(self, texts, **kwargs):
        text_list = self._record_encode_call(texts, **kwargs)
        return np.stack([vector_for_text(text, self.dim) for text in text_list], axis=0)


def five_units(tmp_path: Path) -> list[CodeUnit]:
    return extract_units(tmp_path, FIVE_FUNCTION_SOURCE, filename="mod.py")


def patch_get_model(monkeypatch, model: CountingModel) -> dict[str, int]:
    counts = {"count": 0}

    def fake_get_model(*_args, **_kwargs):
        counts["count"] += 1
        return model

    monkeypatch.setattr(semantic, "get_model", fake_get_model)
    return counts


def active_vectors_path(shard_dir: Path) -> Path:
    payload = json.loads((shard_dir / embedding_cache.INDEX_FILENAME).read_text(encoding="utf-8"))
    return shard_dir / embedding_cache._vectors_filename(payload["generation"])


class MidEncodeCpuFallbackModel(CountingModel):
    """Fake that lands on CPU during encode, like the OOM/invalid-output retry ladder."""

    def encode(self, texts, **kwargs):
        result = super().encode(texts, **kwargs)
        self.device = "cpu"
        return result
