"""Shared builders and runners for the test_semantic test modules."""

from __future__ import annotations

import numpy as np


def constant_embeddings(row_count: int, vector: tuple[float, float]) -> np.ndarray:
    """Return one float32 embedding vector for each requested row."""
    return np.tile(np.asarray(vector, dtype=np.float32), (row_count, 1))


class FakeModel:
    """Simple deterministic embedding model stub."""

    def __init__(self) -> None:
        self.codes = 0

    def encode(self, texts, **kwargs):
        self.codes += 1
        if len(texts) == 2:
            return np.array(
                [
                    [1.0, 0.0],
                    [0.97, 0.243],
                ],
                dtype=np.float32,
            )
        return constant_embeddings(len(texts), (1.0, 0.0))


# Saved prompts exactly as they appear in EmbeddingGemma's
# config_sentence_transformers.json; the fake below composes them the same way
# SentenceTransformers does, so these tests assert the *effective* model input.
EMBEDDINGGEMMA_SAVED_PROMPTS = {
    "query": "task: search result | query: ",
    "document": "title: none | text: ",
    "STS": "task: sentence similarity | query: ",
    "InstructionRetrieval": "task: code retrieval | query: ",
}


class PromptAwareGemmaModel:
    """Fake EmbeddingGemma emulating SentenceTransformers prompt composition.

    ``encode_query``/``encode_document`` fall back to the saved query/document
    prompt whenever the caller provides no explicit ``prompt``/``prompt_name``,
    exactly like the real backend, so a manually prefixed input would surface
    here as a double prompt.
    """

    def __init__(self) -> None:
        self.prompts = dict(EMBEDDINGGEMMA_SAVED_PROMPTS)
        self.calls: list[tuple[str, list[str]]] = []

    def _run(
        self,
        method: str,
        texts,
        prompt: str | None,
        prompt_name: str | None,
        default_prompt_name: str | None,
    ) -> np.ndarray:
        if prompt is None:
            name = prompt_name if prompt_name is not None else default_prompt_name
            prompt = self.prompts.get(name, "") if name is not None else ""
        effective = [f"{prompt}{text}" for text in texts]
        self.calls.append((method, effective))
        return np.array(
            [[1.0, 0.0] if i == 0 else [0.0, 1.0] for i in range(len(texts))],
            dtype=np.float32,
        )

    def encode(self, texts, prompt=None, prompt_name=None, **kwargs):
        return self._run("encode", texts, prompt, prompt_name, None)

    def encode_query(self, texts, prompt=None, prompt_name=None, **kwargs):
        return self._run("encode_query", texts, prompt, prompt_name, "query")

    def encode_document(self, texts, prompt=None, prompt_name=None, **kwargs):
        return self._run("encode_document", texts, prompt, prompt_name, "document")


class WhitespaceTokenizer:
    """Tokenizer stub whose token count is the whitespace-separated word count."""

    def __call__(self, texts, **_kwargs):
        return {"input_ids": [text.split() for text in texts]}


class RecordingModel:
    """Model stub returning one fixed vector per call and recording its inputs."""

    def __init__(self) -> None:
        self.encoded: list[list[str]] = []

    def encode(self, texts, **_kwargs):
        self.encoded.append(list(texts))
        return constant_embeddings(len(texts), (1.0, 0.0))


FULL_REVISION = "1" * 40


class WarmCacheModel:
    """Deterministic embedding model stub used to populate a warm on-disk cache."""

    def __init__(self, dim: int = 2) -> None:
        self.dim = dim
        self.encode_calls = 0

    def encode(self, texts, **_kwargs):
        self.encode_calls += 1
        return constant_embeddings(len(texts), (1.0, 0.0))


def fail_if_called(*_args, **_kwargs):
    """Fail the test whenever the mocked callable it replaces is invoked."""
    raise AssertionError("this callable must not run on a fully warm cache hit")
