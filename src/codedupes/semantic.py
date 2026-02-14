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

import ast
import importlib
import logging
from typing import Literal

import numpy as np

from codedupes.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MODEL,
    DEFAULT_SEMANTIC_THRESHOLD,
    DEFAULT_TOP_K,
)
from codedupes.models import CodeUnit, DuplicatePair

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
    "code": "Represent this code for finding similar code: ",
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


def _require_dependency(module_name: str, install_hint: str) -> None:
    """Raise a clear error when a required dependency is unavailable."""
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise
        raise ModuleNotFoundError(
            f"{module_name} is required for semantic analysis. Install with {install_hint}."
        ) from exc


def _check_semantic_dependencies(model_name: str) -> None:
    """Validate required runtime dependencies before model loading."""
    _require_dependency("sentence_transformers", "pip install codedupes")
    _require_dependency("transformers", "pip install codedupes")
    _require_dependency("torch", "pip install codedupes")
    if _is_c2llm(model_name):
        _require_dependency(
            "deepspeed",
            "pip install codedupes[gpu] or pip install deepspeed",
        )


def get_code_unit_statement_count(unit: CodeUnit) -> int:
    """Get effective statement count for a unit, excluding docstring.

    This returns the number of top-level AST statements in the unit source after
    removing a leading docstring, when present.
    """
    if not unit.source:
        return 0

    text = unit.source.strip()
    if not text:
        return 0

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return 0

    if not tree.body:
        return 0

    top_node = tree.body[0]
    body = []
    if isinstance(top_node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        body = top_node.body
    else:
        body = tree.body

    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]

    return len(body)


def _is_c2llm(model_name: str) -> bool:
    """Check if model is a C2LLM variant."""
    return model_name in _C2LLM_MODELS or "C2LLM" in model_name


def get_model(model_name: str = DEFAULT_MODEL):
    """Lazy-load the embedding model.

    For C2LLM models, applies required loading args:
      - trust_remote_code=True (custom PMA pooling layer)
      - padding_side="left" (causal LM backbone requirement)
    """
    global _model, _model_name

    if _model is None or _model_name != model_name:
        logger.info(f"Loading embedding model: {model_name}")
        _check_semantic_dependencies(model_name)

        try:
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError as exc:
            if exc.name == "sentence_transformers":
                raise ModuleNotFoundError(
                    "sentence-transformers is not installed. Install it with `pip install codedupes`."
                ) from exc
            if exc.name == "deepspeed":
                raise ModuleNotFoundError(
                    "deepspeed is required for C2LLM models. "
                    "Install with `pip install codedupes[gpu]` or `pip install deepspeed`."
                ) from exc
            raise

        try:
            if _is_c2llm(model_name):
                _model = SentenceTransformer(
                    model_name,
                    trust_remote_code=True,
                    tokenizer_kwargs={"padding_side": "left"},
                )
            else:
                # Avoid remote-code execution for arbitrary user-supplied models.
                _model = SentenceTransformer(model_name)
        except ModuleNotFoundError as exc:
            if exc.name == "deepspeed":
                raise ModuleNotFoundError(
                    "deepspeed is required for C2LLM models. "
                    "Install with `pip install codedupes[gpu]` or `pip install deepspeed`."
                ) from exc
            raise

        _model_name = model_name

    return _model


def clear_model_cache() -> None:
    """Clear cached embedding model state."""
    global _model, _model_name
    _model = None
    _model_name = None


def _is_cuda_oom_error(error: RuntimeError) -> bool:
    """Return True when an exception is likely a CUDA out-of-memory condition."""
    msg = str(error).lower()
    return "out of memory" in msg or "cuda out of memory" in msg


def _truncate_code_if_needed(text: str, unit_name: str, model) -> str:
    """Truncate code input to the model max token length with best-effort safety."""
    max_tokens = getattr(model, "max_seq_length", None)
    tokenizer = getattr(model, "tokenizer", None)

    if not max_tokens or not tokenizer:
        return text

    try:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
    except Exception:
        logger.debug(
            "Tokenization failed while preparing '%s'; using full text", unit_name, exc_info=True
        )
        return text

    token_count = len(token_ids)
    if token_count <= max_tokens:
        return text

    logger.warning(
        "Code unit '%s' is long (%d tokens), truncating to %d tokens for semantic embedding",
        unit_name,
        token_count,
        max_tokens,
    )
    try:
        truncated_ids = tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_tokens,
        )
        return tokenizer.decode(truncated_ids, skip_special_tokens=True)
    except Exception:
        logger.debug(
            "Token decode failed while truncating '%s'; using char fallback",
            unit_name,
            exc_info=True,
        )
        return text[: max_tokens * 4]


def _get_instruction(model_name: str, mode: Literal["code", "query", "describe"]) -> str:
    """Get the appropriate instruction prefix for the model and task."""
    if _is_c2llm(model_name):
        return C2LLM_INSTRUCTIONS[mode]
    return _GENERIC_INSTRUCTIONS.get(mode, "")


def prepare_code_for_embedding(
    unit: CodeUnit,
    model_name: str = DEFAULT_MODEL,
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
    model_name: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    """Compute embeddings for all code units.

    Returns:
        numpy array of shape (len(units), embedding_dim)
    """
    model = get_model(model_name)

    texts = []
    for unit in units:
        prepared = prepare_code_for_embedding(unit, model_name=model_name)
        texts.append(_truncate_code_if_needed(prepared, unit.qualified_name, model))

    logger.info(f"Computing embeddings for {len(texts)} code units")
    try:
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
            normalize_embeddings=True,  # For cosine similarity via dot product
        )
    except RuntimeError as e:  # pragma: no cover - defensive handling
        if not _is_cuda_oom_error(e):
            raise

        logger.warning("CUDA OOM during semantic embedding; retrying on CPU")
        model.to("cpu")
        embeddings = model.encode(
            texts,
            batch_size=1,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
            normalize_embeddings=True,  # For cosine similarity via dot product
            device="cpu",
        )

    return embeddings


def find_semantic_duplicates(
    units: list[CodeUnit],
    embeddings: np.ndarray,
    threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
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
    model_name: str = DEFAULT_MODEL,
    top_k: int = DEFAULT_TOP_K,
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
    model_name: str = DEFAULT_MODEL,
    threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
    exclude_pairs: set[tuple[str, str]] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
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
