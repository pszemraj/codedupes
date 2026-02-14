"""Semantic duplicate detection using code embeddings.

Uses C2LLM (Code Contrastive LLM) with PMA-based pooling for code embeddings.
C2LLM is SOTA for sub-1B models on MTEB-Code (75.46 avg score).

Model: codefuse-ai/C2LLM-0.5B
  - Base: Qwen2.5-Coder-0.5B-Instruct
  - Pooling: Multihead Attention (PMA) with 32 heads
  - Requires: trust_remote_code=True, padding_side="left"
  - Supports: Code2Code, Text2Code, Code2Text retrieval
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

from .models import CodeUnit, DuplicatePair

logger = logging.getLogger(__name__)

# Lazy-loaded model
_model = None
_model_name = None

# C2LLM task-specific instruction prefixes.
# These are prepended to every input to steer the PMA pooling toward the right
# representation space. Using an empty string is valid but instruction-tuned
# prefixes consistently improve retrieval quality per the C2LLM paper (Table 1).
C2LLM_INSTRUCTIONS: dict[str, str] = {
    # Code2Code: for pairwise code similarity / dedup
    "code": "Represent this code snippet for retrieval: ",
    # Text2Code: natural-language query -> code search
    "query": "Represent this query for searching relevant code: ",
    # Code2Text: code -> natural-language description search
    "describe": "Represent this code for finding its description: ",
}

# Fallback instruction for non-C2LLM models that don't use instruction prefixes
_GENERIC_INSTRUCTIONS: dict[str, str] = {
    "code": "",
    "query": "Represent this query for searching relevant code: ",
    "describe": "",
}

# Models known to need C2LLM-specific loading args
_C2LLM_MODELS = {"codefuse-ai/C2LLM-0.5B", "codefuse-ai/C2LLM-7B"}


def _is_c2llm(model_name: str) -> bool:
    """Check if model is a C2LLM variant."""
    return model_name in _C2LLM_MODELS or "C2LLM" in model_name


def get_model(model_name: str = "codefuse-ai/C2LLM-0.5B"):
    """Lazy-load the embedding model.

    For C2LLM models, applies required loading args:
      - trust_remote_code=True (custom PMA pooling layer)
      - padding_side="left" (causal LM backbone requirement)
    """
    global _model, _model_name

    if _model is None or _model_name != model_name:
        logger.info(f"Loading embedding model: {model_name}")
        from sentence_transformers import SentenceTransformer

        if _is_c2llm(model_name):
            _model = SentenceTransformer(
                model_name,
                trust_remote_code=True,
                tokenizer_kwargs={"padding_side": "left"},
            )
        else:
            # Fallback for other models (e.g. CodeRankEmbed)
            _model = SentenceTransformer(model_name, trust_remote_code=True)

        _model_name = model_name

    return _model


def clear_model_cache() -> None:
    """Clear cached embedding model state."""
    global _model, _model_name
    _model = None
    _model_name = None


def _get_instruction(model_name: str, mode: Literal["code", "query", "describe"]) -> str:
    """Get the appropriate instruction prefix for the model and task."""
    if _is_c2llm(model_name):
        return C2LLM_INSTRUCTIONS[mode]
    return _GENERIC_INSTRUCTIONS.get(mode, "")


def prepare_code_for_embedding(
    unit: CodeUnit,
    model_name: str = "codefuse-ai/C2LLM-0.5B",
    mode: Literal["code", "query"] = "code",
) -> str:
    """Prepare code unit for embedding.

    C2LLM expects: instruction_prefix + source_code
    The instruction steers the PMA cross-attention toward the right
    representation for the downstream task.

    For dedup (code2code), both sides get the "code" instruction.
    For search (text2code), queries get the "query" instruction.
    """
    source = unit.source.strip()
    instruction = _get_instruction(model_name, mode)
    return f"{instruction}{source}"


def compute_embeddings(
    units: list[CodeUnit],
    model_name: str = "codefuse-ai/C2LLM-0.5B",
    batch_size: int = 32,
) -> np.ndarray:
    """Compute embeddings for all code units.

    Returns:
        numpy array of shape (len(units), embedding_dim)
    """
    model = get_model(model_name)

    texts = [prepare_code_for_embedding(unit, model_name=model_name) for unit in units]

    logger.info(f"Computing embeddings for {len(texts)} code units")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 100,
        convert_to_numpy=True,
        normalize_embeddings=True,  # For cosine similarity via dot product
    )

    return embeddings


def find_semantic_duplicates(
    units: list[CodeUnit],
    embeddings: np.ndarray,
    threshold: float = 0.85,
    exclude_exact: set[tuple[str, str]] | None = None,
) -> list[DuplicatePair]:
    """Find semantically similar code units via embedding cosine similarity.

    Args:
        units: List of code units
        embeddings: Precomputed embeddings (normalized)
        threshold: Minimum similarity to consider a duplicate
        exclude_exact: Set of (uid_a, uid_b) pairs to exclude (already found by exact methods)

    Returns:
        List of duplicate pairs above threshold
    """
    exclude_exact = exclude_exact or set()
    n = len(units)

    # Compute pairwise cosine similarity (embeddings are normalized, so dot product = cosine)
    # Do this in batches to avoid memory issues for large codebases
    logger.info(f"Computing pairwise similarities for {n} units")

    duplicates = []

    # Chunk-based computation to handle large matrices
    chunk_size = 500
    for i in range(0, n, chunk_size):
        end_i = min(i + chunk_size, n)
        chunk_embeddings = embeddings[i:end_i]

        # Compare this chunk against all units (including itself)
        similarities = chunk_embeddings @ embeddings.T

        for local_idx in range(end_i - i):
            global_idx = i + local_idx
            unit_a = units[global_idx]

            # Only look at upper triangle to avoid duplicates
            for j in range(global_idx + 1, n):
                sim = similarities[local_idx, j]

                if sim < threshold:
                    continue

                unit_b = units[j]

                # Skip if same file and overlapping lines
                if unit_a.file_path == unit_b.file_path:
                    if not (unit_a.end_lineno < unit_b.lineno or unit_b.end_lineno < unit_a.lineno):
                        continue

                # Skip if already found by exact methods
                pair_key = tuple(sorted([unit_a.uid, unit_b.uid]))
                if pair_key in exclude_exact:
                    continue

                duplicates.append(
                    DuplicatePair(
                        unit_a=unit_a,
                        unit_b=unit_b,
                        similarity=float(sim),
                        method="semantic",
                    )
                )

    # Sort by similarity descending
    duplicates.sort(key=lambda x: x.similarity, reverse=True)

    logger.info(f"Found {len(duplicates)} semantic duplicates above threshold {threshold}")
    return duplicates


def find_similar_to_query(
    query: str,
    units: list[CodeUnit],
    embeddings: np.ndarray,
    model_name: str = "codefuse-ai/C2LLM-0.5B",
    top_k: int = 10,
) -> list[tuple[CodeUnit, float]]:
    """Find code units most similar to a natural language query.

    Useful for ad-hoc exploration: "find functions that parse JSON"
    """
    model = get_model(model_name)

    # Embed query with task-specific instruction
    instruction = _get_instruction(model_name, "query")
    query_text = f"{instruction}{query}"

    query_embedding = model.encode(
        [query_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]

    # Compute similarities
    similarities = embeddings @ query_embedding

    # Get top-k
    top_indices = np.argsort(similarities)[::-1][:top_k]

    return [(units[i], float(similarities[i])) for i in top_indices]


def run_semantic_analysis(
    units: list[CodeUnit],
    model_name: str = "codefuse-ai/C2LLM-0.5B",
    threshold: float = 0.85,
    exclude_pairs: set[tuple[str, str]] | None = None,
    batch_size: int = 32,
) -> tuple[np.ndarray, list[DuplicatePair]]:
    """Run full semantic duplicate detection.

    Returns:
        (embeddings, duplicate_pairs)
    """
    if not units:
        return np.array([]), []

    embeddings = compute_embeddings(units, model_name=model_name, batch_size=batch_size)
    duplicates = find_semantic_duplicates(
        units, embeddings, threshold=threshold, exclude_exact=exclude_pairs
    )

    return embeddings, duplicates
