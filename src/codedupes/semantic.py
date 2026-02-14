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
from importlib import metadata as importlib_metadata
import logging
import os
import sys
from typing import Literal

import numpy as np
from packaging.version import InvalidVersion, Version

from codedupes.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_C2LLM_REVISION,
    DEFAULT_MODEL,
    DEFAULT_SEMANTIC_THRESHOLD,
    DEFAULT_TOP_K,
)
from codedupes.models import CodeUnit, DuplicatePair

logger = logging.getLogger(__name__)

# Lazy-loaded model
_model = None
_model_name = None
_model_revision = None
_model_trust_remote_code = None

_DEFAULT_TRANSFORMERS_MIN = Version("4.51")
_DEFAULT_TRANSFORMERS_MAX_EXCLUSIVE = Version("5")
_DEFAULT_ST_MIN = Version("5")
_DEFAULT_ST_MAX_EXCLUSIVE = Version("6")

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


class SemanticBackendError(RuntimeError):
    """Raised when semantic model loading or inference backend is incompatible."""


def _configure_semantic_runtime_env() -> None:
    """Set runtime env guards to avoid optional framework noise/import paths."""
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _resolve_model_revision(model_name: str, revision: str | None) -> str | None:
    """Resolve the model revision default."""
    if revision is not None:
        if model_name != DEFAULT_MODEL and revision == DEFAULT_C2LLM_REVISION:
            return None
        return revision
    if model_name == DEFAULT_MODEL:
        return DEFAULT_C2LLM_REVISION
    return None


def _resolve_trust_remote_code(model_name: str, trust_remote_code: bool | None) -> bool:
    """Resolve trust-remote-code default for a model."""
    if trust_remote_code is not None:
        return trust_remote_code
    return _is_c2llm(model_name)


def _safe_package_version(package_name: str) -> str | None:
    """Get installed package version string, returning None if unavailable."""
    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return None


def get_semantic_runtime_versions() -> dict[str, str]:
    """Return semantic runtime versions for diagnostics."""
    return {
        "python": sys.version.split()[0],
        "torch": _safe_package_version("torch") or "missing",
        "transformers": _safe_package_version("transformers") or "missing",
        "sentence-transformers": _safe_package_version("sentence-transformers") or "missing",
        "deepspeed": _safe_package_version("deepspeed") or "missing",
    }


def _validate_version_range(
    package_name: str,
    min_version: Version,
    max_exclusive: Version,
) -> None:
    """Validate that a package version is within an inclusive/exclusive range."""
    raw = _safe_package_version(package_name)
    if raw is None:
        raise SemanticBackendError(
            f"{package_name} is not installed. Install compatible dependencies before semantic runs."
        )

    try:
        parsed = Version(raw)
    except InvalidVersion as exc:
        raise SemanticBackendError(f"Could not parse {package_name} version: {raw}") from exc

    if not (min_version <= parsed < max_exclusive):
        raise SemanticBackendError(
            f"Incompatible {package_name} version {raw} for default model {DEFAULT_MODEL}. "
            f"Supported range is >={min_version},<{max_exclusive}. "
            "Run: pip install 'transformers>=4.51,<5' 'sentence-transformers>=5,<6'."
        )


def _check_default_model_compatibility(model_name: str) -> None:
    """Check dependency compatibility for the default C2LLM model."""
    if model_name != DEFAULT_MODEL:
        return

    _validate_version_range(
        "transformers",
        _DEFAULT_TRANSFORMERS_MIN,
        _DEFAULT_TRANSFORMERS_MAX_EXCLUSIVE,
    )
    _validate_version_range(
        "sentence-transformers",
        _DEFAULT_ST_MIN,
        _DEFAULT_ST_MAX_EXCLUSIVE,
    )


def _is_known_semantic_backend_error(error: Exception) -> bool:
    """Return True when an exception is likely caused by semantic backend compatibility."""
    text = str(error).lower()
    if isinstance(error, ModuleNotFoundError):
        return True
    if isinstance(error, AttributeError) and "all_tied_weights_keys" in text:
        return True
    keywords = (
        "trust_remote_code",
        "deepspeed",
        "flash_attn",
        "c2llm",
        "auto_map",
        "tokenizer",
        "modeling_c2llm",
    )
    return any(keyword in text for keyword in keywords)


def _wrap_semantic_backend_error(
    error: Exception,
    *,
    model_name: str,
    revision: str | None,
    trust_remote_code: bool,
    stage: str,
) -> SemanticBackendError:
    """Convert backend exceptions into a stable semantic error with remediation guidance."""
    versions = get_semantic_runtime_versions()
    version_info = ", ".join(f"{key}={value}" for key, value in versions.items())
    revision_text = revision or "default"
    message = (
        f"Semantic backend failed during {stage} for model={model_name} revision={revision_text} "
        f"trust_remote_code={trust_remote_code}. "
        f"Versions: {version_info}. "
        "Fix suggestions: ensure compatible deps with "
        '\'pip install "transformers>=4.51,<5" "sentence-transformers>=5,<6"\', '
        "or run traditional-only mode with '--traditional-only'."
    )
    wrapped = SemanticBackendError(message)
    wrapped.__cause__ = error
    return wrapped


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
    _check_default_model_compatibility(model_name)


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


def _resolve_c2llm_torch_dtype():
    """Choose torch dtype for C2LLM without fp16 fallback."""
    try:
        import torch
    except ModuleNotFoundError:
        return None

    if torch.cuda.is_available():
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return None

    # CPU path: prefer bfloat16 to reduce memory footprint when supported.
    # We intentionally do not use fp16 anywhere.
    return torch.bfloat16
    return None


def get_model(
    model_name: str = DEFAULT_MODEL,
    revision: str | None = None,
    trust_remote_code: bool | None = None,
):
    """Lazy-load the embedding model.

    For C2LLM models, applies required loading args:
      - trust_remote_code=True (custom PMA pooling layer)
      - padding_side="left" (causal LM backbone requirement)
    """
    global _model, _model_name, _model_revision, _model_trust_remote_code

    resolved_revision = _resolve_model_revision(model_name, revision)
    resolved_trust_remote_code = _resolve_trust_remote_code(model_name, trust_remote_code)

    cache_miss = any(
        (
            _model is None,
            _model_name != model_name,
            _model_revision != resolved_revision,
            _model_trust_remote_code != resolved_trust_remote_code,
        )
    )

    if cache_miss:
        logger.info(f"Loading embedding model: {model_name}")
        _configure_semantic_runtime_env()
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

        st_kwargs: dict[str, object] = {
            "trust_remote_code": resolved_trust_remote_code,
        }
        if resolved_revision is not None:
            st_kwargs["revision"] = resolved_revision

        model_kwargs: dict[str, object] = {}
        tokenizer_kwargs: dict[str, object] = {}
        config_kwargs: dict[str, object] = {}

        if _is_c2llm(model_name):
            tokenizer_kwargs["padding_side"] = "left"
            model_kwargs["low_cpu_mem_usage"] = True
            selected_dtype = _resolve_c2llm_torch_dtype()
            if selected_dtype is not None:
                model_kwargs["dtype"] = selected_dtype
                logger.info("Using C2LLM torch dtype: %s", selected_dtype)

        if resolved_revision is not None:
            model_kwargs["revision"] = resolved_revision
            tokenizer_kwargs["revision"] = resolved_revision
            config_kwargs["revision"] = resolved_revision

        if resolved_trust_remote_code:
            model_kwargs["trust_remote_code"] = True
            tokenizer_kwargs["trust_remote_code"] = True
            config_kwargs["trust_remote_code"] = True

        if model_kwargs:
            st_kwargs["model_kwargs"] = model_kwargs
        if tokenizer_kwargs:
            st_kwargs["tokenizer_kwargs"] = tokenizer_kwargs
        if config_kwargs:
            st_kwargs["config_kwargs"] = config_kwargs

        try:
            _model = SentenceTransformer(model_name, **st_kwargs)
        except ModuleNotFoundError as exc:
            if exc.name == "deepspeed":
                raise ModuleNotFoundError(
                    "deepspeed is required for C2LLM models. "
                    "Install with `pip install codedupes[gpu]` or `pip install deepspeed`."
                ) from exc
            raise
        except Exception as exc:
            if _is_known_semantic_backend_error(exc):
                raise _wrap_semantic_backend_error(
                    exc,
                    model_name=model_name,
                    revision=resolved_revision,
                    trust_remote_code=resolved_trust_remote_code,
                    stage="model loading",
                )
            raise

        _model_name = model_name
        _model_revision = resolved_revision
        _model_trust_remote_code = resolved_trust_remote_code

    return _model


def clear_model_cache() -> None:
    """Clear cached embedding model state."""
    global _model, _model_name, _model_revision, _model_trust_remote_code
    _model = None
    _model_name = None
    _model_revision = None
    _model_trust_remote_code = None


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


def _resolve_instruction_prefix(
    model_name: str,
    mode: Literal["code", "query", "describe"],
    instruction_prefix: str | None,
) -> str:
    """Resolve instruction prefix override for embedding inputs."""
    if instruction_prefix is not None:
        return instruction_prefix
    return _get_instruction(model_name, mode)


def prepare_code_for_embedding(
    unit: CodeUnit,
    model_name: str = DEFAULT_MODEL,
    mode: Literal["code", "query"] = "code",
    instruction_prefix: str | None = None,
) -> str:
    """Prepare code unit for embedding.

    C2LLM expects: instruction_prefix + source_code
    The instruction steers the PMA cross-attention toward the right
    representation for the downstream task.

    For dedup (code2code), both sides get the "code" instruction.
    For search (text2code), queries get the "query" instruction.
    """
    source = unit.source.strip()
    instruction = _resolve_instruction_prefix(model_name, mode, instruction_prefix)
    return f"{instruction}{source}"


def compute_embeddings(
    units: list[CodeUnit],
    model_name: str = DEFAULT_MODEL,
    instruction_prefix: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    revision: str | None = None,
    trust_remote_code: bool | None = None,
) -> np.ndarray:
    """Compute embeddings for all code units.

    Returns:
        numpy array of shape (len(units), embedding_dim)
    """
    resolved_revision = _resolve_model_revision(model_name, revision)
    resolved_trust_remote_code = _resolve_trust_remote_code(model_name, trust_remote_code)
    model = get_model(
        model_name,
        revision=resolved_revision,
        trust_remote_code=resolved_trust_remote_code,
    )

    texts = []
    for unit in units:
        prepared = prepare_code_for_embedding(
            unit,
            model_name=model_name,
            instruction_prefix=instruction_prefix,
        )
        texts.append(_truncate_code_if_needed(prepared, unit.qualified_name, model))

    logger.info(f"Computing embeddings for {len(texts)} code units")
    current_batch_size = max(1, batch_size)
    attempted_cpu_fallback = False
    while True:
        try:
            encode_kwargs = {
                "batch_size": current_batch_size,
                "show_progress_bar": len(texts) > 100,
                "convert_to_numpy": True,
                "normalize_embeddings": True,  # For cosine similarity via dot product
            }
            if attempted_cpu_fallback:
                encode_kwargs["device"] = "cpu"
            embeddings = model.encode(
                texts,
                **encode_kwargs,
            )
            break
        except RuntimeError as e:  # pragma: no cover - defensive handling
            if not _is_cuda_oom_error(e):
                if _is_known_semantic_backend_error(e):
                    raise _wrap_semantic_backend_error(
                        e,
                        model_name=model_name,
                        revision=resolved_revision,
                        trust_remote_code=resolved_trust_remote_code,
                        stage="embedding inference",
                    )
                raise

            if current_batch_size > 1:
                next_batch_size = max(1, current_batch_size // 2)
                logger.warning(
                    "CUDA OOM during semantic embedding at batch_size=%d; retrying with "
                    "batch_size=%d",
                    current_batch_size,
                    next_batch_size,
                )
                current_batch_size = next_batch_size
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    logger.debug("Failed to clear CUDA cache after OOM", exc_info=True)
                continue

            if attempted_cpu_fallback:
                logger.warning(
                    "CUDA OOM during semantic embedding on CPU fallback (batch_size=1); aborting"
                )
                raise

            logger.warning("CUDA OOM during semantic embedding at batch_size=1; retrying on CPU")
            attempted_cpu_fallback = True
            if hasattr(model, "to"):
                model.to("cpu")
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                logger.debug("Failed to clear CUDA cache after OOM", exc_info=True)
            continue
        except Exception as e:
            if _is_known_semantic_backend_error(e):
                raise _wrap_semantic_backend_error(
                    e,
                    model_name=model_name,
                    revision=resolved_revision,
                    trust_remote_code=resolved_trust_remote_code,
                    stage="embedding inference",
                )
            raise

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

    def _types_compatible(unit_a: CodeUnit, unit_b: CodeUnit) -> bool:
        if unit_a.unit_type == unit_b.unit_type:
            return True
        function_like = {"function", "method"}
        return (
            unit_a.unit_type.name.lower() in function_like
            and unit_b.unit_type.name.lower() in function_like
        )

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

                if not _types_compatible(unit_a, unit_b):
                    continue

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
    instruction_prefix: str | None = None,
    top_k: int = DEFAULT_TOP_K,
    revision: str | None = None,
    trust_remote_code: bool | None = None,
) -> list[tuple[CodeUnit, float]]:
    """Find code units most similar to a natural language query.

    Useful for ad-hoc exploration: "find functions that parse JSON"
    """
    resolved_revision = _resolve_model_revision(model_name, revision)
    resolved_trust_remote_code = _resolve_trust_remote_code(model_name, trust_remote_code)
    model = get_model(
        model_name,
        revision=resolved_revision,
        trust_remote_code=resolved_trust_remote_code,
    )

    # Embed query with task-specific instruction
    instruction = _resolve_instruction_prefix(model_name, "query", instruction_prefix)
    query_text = f"{instruction}{query}"

    try:
        query_embedding = model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
    except Exception as exc:
        if _is_known_semantic_backend_error(exc):
            raise _wrap_semantic_backend_error(
                exc,
                model_name=model_name,
                revision=resolved_revision,
                trust_remote_code=resolved_trust_remote_code,
                stage="query embedding",
            )
        raise

    # Compute similarities
    similarities = embeddings @ query_embedding

    # Get top-k
    top_indices = np.argsort(similarities)[::-1][:top_k]

    return [(units[i], float(similarities[i])) for i in top_indices]


def run_semantic_analysis(
    units: list[CodeUnit],
    model_name: str = DEFAULT_MODEL,
    instruction_prefix: str | None = None,
    threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
    exclude_pairs: set[tuple[str, str]] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    revision: str | None = None,
    trust_remote_code: bool | None = None,
) -> tuple[np.ndarray, list[DuplicatePair]]:
    """Run full semantic duplicate detection.

    Returns:
        (embeddings, duplicate_pairs)
    """
    if not units:
        return np.array([]), []

    embeddings = compute_embeddings(
        units,
        model_name=model_name,
        instruction_prefix=instruction_prefix,
        batch_size=batch_size,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )
    duplicates = find_semantic_duplicates(
        units, embeddings, threshold=threshold, exclude_exact=exclude_pairs
    )

    return embeddings, duplicates
