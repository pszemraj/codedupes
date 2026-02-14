"""Semantic duplicate detection using embedding similarity."""

from __future__ import annotations

import ast
import importlib
from importlib import metadata as importlib_metadata
import logging
import os
import sys
from typing import Any, Callable, Literal, TypeVar, cast

import numpy as np
from packaging.version import InvalidVersion, Version

from codedupes.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHECK_SEMANTIC_TASK,
    DEFAULT_MODEL,
    DEFAULT_SEARCH_SEMANTIC_TASK,
    DEFAULT_SEMANTIC_THRESHOLD,
    DEFAULT_TOP_K,
    SEMANTIC_TASK_CHOICES,
)
from codedupes.models import CodeUnit, DuplicatePair
from codedupes.pairs import ordered_pair_key
from codedupes.semantic_profiles import get_default_semantic_threshold, resolve_model_profile

logger = logging.getLogger(__name__)

# Lazy-loaded model
_model = None
_model_name: str | None = None
_model_revision: str | None = None
_model_trust_remote_code: bool | None = None

_DEFAULT_TRANSFORMERS_MIN = Version("4.51")
_DEFAULT_TRANSFORMERS_MAX_EXCLUSIVE = Version("5")
_DEFAULT_ST_MIN = Version("5")
_DEFAULT_ST_MAX_EXCLUSIVE = Version("6")

SemanticTask = Literal[
    "semantic-similarity",
    "code-retrieval",
    "retrieval",
    "question-answering",
    "fact-verification",
    "classification",
    "clustering",
]

# C2LLM task-specific instruction prefixes.
C2LLM_INSTRUCTIONS: dict[str, str] = {
    "code": "Represent this code for finding similar code: ",
    "query": "Represent this query for searching relevant code: ",
}

EMBEDDINGGEMMA_QUERY_PREFIXES: dict[SemanticTask, str] = {
    "semantic-similarity": "task: sentence similarity | query: ",
    "code-retrieval": "task: code retrieval | query: ",
    "retrieval": "task: search result | query: ",
    "question-answering": "task: question answering | query: ",
    "fact-verification": "task: fact checking | query: ",
    "classification": "task: classification | query: ",
    "clustering": "task: clustering | query: ",
}
EMBEDDINGGEMMA_DOCUMENT_PREFIX = "title: none | text: "
_DEEPSPEED_REQUIRED_MESSAGE = (
    "deepspeed is required for C2LLM models. "
    "Install with `pip install codedupes[gpu]` or `pip install deepspeed`."
)


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


T = TypeVar("T")


def _resolve_profile_default(
    model_name: str,
    override: T | None,
    *,
    accessor: Callable[[Any], T],
) -> T:
    """Resolve a profile-derived default with optional explicit override.

    :param model_name: Requested model identifier.
    :param override: Explicit override value.
    :param accessor: Profile accessor for default lookup.
    :return: Explicit override when provided, otherwise profile default.
    """
    if override is not None:
        return override
    profile = resolve_model_profile(model_name)
    return accessor(profile)


def _resolve_model_revision(model_name: str, revision: str | None) -> str | None:
    """Resolve model revision for a model, honoring explicit overrides.

    :param model_name: Requested model identifier.
    :param revision: Optional explicit revision.
    :return: Profile default revision when no explicit revision is provided.
    """
    return _resolve_profile_default(
        model_name,
        revision,
        accessor=lambda profile: cast(str | None, profile.default_revision),
    )


def _resolve_trust_remote_code(model_name: str, trust_remote_code: bool | None) -> bool:
    """Resolve trust-remote-code mode for a model, honoring explicit overrides.

    :param model_name: Requested model identifier.
    :param trust_remote_code: Optional explicit trust setting.
    :return: Profile default trust setting when no override is provided.
    """
    return _resolve_profile_default(
        model_name,
        trust_remote_code,
        accessor=lambda profile: cast(bool, profile.default_trust_remote_code),
    )


def _safe_package_version(package_name: str) -> str | None:
    """Get installed package version string, returning ``None`` if unavailable.

    :param package_name: Package to inspect.
    :return: Installed version string, or ``None``.
    """
    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return None


def get_semantic_runtime_versions() -> dict[str, str]:
    """Return semantic runtime versions for diagnostics.

    :return: Mapping of runtime component names to version strings.
    """
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
    """Validate that a package version is within an inclusive/exclusive range.

    :param package_name: Package to validate.
    :param min_version: Lower bound inclusive.
    :param max_exclusive: Upper bound exclusive.
    :return: ``None``.
    :raises SemanticBackendError: If package version is invalid or incompatible.
    """
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
            f"Incompatible {package_name} version {raw} for C2LLM models. "
            f"Supported range is >={min_version},<{max_exclusive}. "
            "Run: pip install 'transformers>=4.51,<5' 'sentence-transformers>=5,<6'."
        )


def _check_c2llm_model_compatibility(model_name: str) -> None:
    """Check dependency compatibility for C2LLM-family models."""
    profile = resolve_model_profile(model_name)
    if profile.family != "c2llm":
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
    """Return True when an exception is likely caused by semantic backend compatibility.

    :param error: Captured exception.
    :return: ``True`` when exception text matches known backend issues.
    """
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
        "embeddinggemma",
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
    """Convert backend exceptions into a stable semantic error with remediation guidance.

    :param error: Original exception.
    :param model_name: Model involved in the failure.
    :param revision: Resolved model revision.
    :param trust_remote_code: Trust-remote-code flag used.
    :param stage: Backend stage where failure occurred.
    :return: Wrapped ``SemanticBackendError`` with remediation guidance.
    """
    versions = get_semantic_runtime_versions()
    version_info = ", ".join(f"{key}={value}" for key, value in versions.items())
    revision_text = revision or "default"
    profile = resolve_model_profile(model_name)
    hints: list[str] = []
    if profile.requires_deepspeed:
        hints.append(
            "install C2LLM-compatible deps via "
            '\'pip install "transformers>=4.51,<5" "sentence-transformers>=5,<6"\'.'
        )
    hints.append("or run traditional-only mode with '--traditional-only'.")

    message = (
        f"Semantic backend failed during {stage} for model={model_name} revision={revision_text} "
        f"trust_remote_code={trust_remote_code}. "
        f"Versions: {version_info}. "
        "Fix suggestions: " + " ".join(hints)
    )
    wrapped = SemanticBackendError(message)
    wrapped.__cause__ = error
    return wrapped


def _require_dependency(module_name: str, install_hint: str) -> None:
    """Raise a clear error when a required dependency is unavailable.

    :param module_name: Required module name.
    :param install_hint: Suggested installation command.
    :return: ``None``.
    :raises ModuleNotFoundError: When dependency is missing.
    """
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

    profile = resolve_model_profile(model_name)
    if profile.requires_deepspeed:
        _require_dependency(
            "deepspeed",
            "pip install codedupes[gpu] or pip install deepspeed",
        )

    _check_c2llm_model_compatibility(model_name)


def _raise_missing_deepspeed(exc: ModuleNotFoundError) -> None:
    """Raise a stable deepspeed dependency error for C2LLM-family loads."""
    raise ModuleNotFoundError(_DEEPSPEED_REQUIRED_MESSAGE) from exc


def _clear_cuda_cache() -> None:
    """Best-effort CUDA cache clear after OOM retries."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        logger.debug("Failed to clear CUDA cache after OOM", exc_info=True)


def get_code_unit_statement_count(unit: CodeUnit) -> int:
    """Get effective statement count for a unit, excluding docstring.

    :param unit: Unit to measure.
    :return: Number of executable statements.
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


def _resolve_c2llm_torch_dtype() -> Any:
    """Choose torch dtype for C2LLM without fp16 fallback.

    :return: Suggested dtype object for Torch models, or ``None``.
    """
    try:
        import torch
    except ModuleNotFoundError:
        return None

    if torch.cuda.is_available():
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return None

    return torch.bfloat16


def _resolve_embeddinggemma_torch_dtype() -> Any:
    """Choose torch dtype for EmbeddingGemma without fp16 usage.

    :return: Suggested dtype object for Torch models, or ``None``.
    """
    try:
        import torch
    except ModuleNotFoundError:
        return None

    if torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported"):
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16

    return torch.float32


def _patch_c2llm_runtime_compat() -> None:
    """Patch known C2LLM remote-code symbol gaps for newer runtimes."""
    import builtins

    if not hasattr(builtins, "is_torch_npu_available"):
        setattr(builtins, "is_torch_npu_available", lambda: False)


def get_model(
    model_name: str = DEFAULT_MODEL,
    revision: str | None = None,
    trust_remote_code: bool | None = None,
) -> object:
    """Lazy-load the embedding model.

    :param model_name: Model alias or identifier.
    :param revision: Optional model revision.
    :param trust_remote_code: Optional remote code trust setting.
    :return: Loaded model instance.
    """
    global _model, _model_name, _model_revision, _model_trust_remote_code

    profile = resolve_model_profile(model_name)
    resolved_model_name = profile.canonical_name
    resolved_revision = _resolve_model_revision(model_name, revision)
    resolved_trust_remote_code = _resolve_trust_remote_code(model_name, trust_remote_code)

    cache_miss = any(
        (
            _model is None,
            _model_name != resolved_model_name,
            _model_revision != resolved_revision,
            _model_trust_remote_code != resolved_trust_remote_code,
        )
    )

    if cache_miss:
        logger.info("Loading embedding model: %s", resolved_model_name)
        _configure_semantic_runtime_env()
        _check_semantic_dependencies(resolved_model_name)
        if profile.family == "c2llm":
            _patch_c2llm_runtime_compat()

        try:
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError as exc:
            if exc.name == "sentence_transformers":
                raise ModuleNotFoundError(
                    "sentence-transformers is not installed. Install it with `pip install codedupes`."
                ) from exc
            if exc.name == "deepspeed":
                _raise_missing_deepspeed(exc)
            raise

        st_kwargs: dict[str, object] = {
            "trust_remote_code": resolved_trust_remote_code,
        }
        if resolved_revision is not None:
            st_kwargs["revision"] = resolved_revision

        model_kwargs: dict[str, object] = {}
        tokenizer_kwargs: dict[str, object] = {}
        config_kwargs: dict[str, object] = {}

        if profile.left_padding:
            tokenizer_kwargs["padding_side"] = "left"
        if profile.low_cpu_mem_usage:
            model_kwargs["low_cpu_mem_usage"] = True

        if profile.family == "c2llm":
            selected_dtype = _resolve_c2llm_torch_dtype()
            if selected_dtype is not None:
                model_kwargs["torch_dtype"] = selected_dtype
                logger.info("Using C2LLM torch dtype: %s", selected_dtype)
        elif profile.family == "embeddinggemma":
            selected_dtype = _resolve_embeddinggemma_torch_dtype()
            if selected_dtype is not None:
                model_kwargs["torch_dtype"] = selected_dtype
                logger.info("Using EmbeddingGemma torch dtype: %s", selected_dtype)

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
            _model = SentenceTransformer(resolved_model_name, **st_kwargs)
        except ModuleNotFoundError as exc:
            if exc.name == "deepspeed":
                _raise_missing_deepspeed(exc)
            raise
        except Exception as exc:
            if _is_known_semantic_backend_error(exc):
                raise _wrap_semantic_backend_error(
                    exc,
                    model_name=resolved_model_name,
                    revision=resolved_revision,
                    trust_remote_code=resolved_trust_remote_code,
                    stage="model loading",
                )
            raise

        _model_name = resolved_model_name
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
    """Return True when an exception is likely a CUDA out-of-memory condition.

    :param error: Runtime error thrown by model execution.
    :return: ``True`` when text indicates CUDA OOM.
    """
    msg = str(error).lower()
    return "out of memory" in msg or "cuda out of memory" in msg


def _truncate_code_if_needed(text: str, unit_name: str, model: Any) -> str:
    """Truncate code input to the model max token length with best-effort safety.

    :param text: Source text to truncate.
    :param unit_name: Unit name for logging context.
    :param model: Model object with tokenizer metadata.
    :return: Possibly truncated source text.
    """
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


def _normalize_semantic_task(
    semantic_task: str | None,
    *,
    default_task: SemanticTask,
) -> SemanticTask:
    """Validate and normalize semantic task names.

    :param semantic_task: Candidate task value.
    :param default_task: Fallback task when no value is provided.
    :return: Normalized task enum.
    """
    if semantic_task is None:
        return default_task

    normalized = semantic_task.strip().lower()
    if normalized not in SEMANTIC_TASK_CHOICES:
        allowed = ", ".join(SEMANTIC_TASK_CHOICES)
        raise ValueError(f"Unknown semantic task '{semantic_task}'. Expected one of: {allowed}")
    return normalized  # type: ignore[return-value]


def _get_embeddinggemma_prefix(task: SemanticTask, mode: Literal["code", "query"]) -> str:
    """Get task-aware prompt prefixes for EmbeddingGemma.

    :param task: Normalized task.
    :param mode: Input mode.
    :return: Instruction prefix.
    """
    if mode == "query":
        return EMBEDDINGGEMMA_QUERY_PREFIXES[task]

    if task in {"retrieval", "code-retrieval"}:
        return EMBEDDINGGEMMA_DOCUMENT_PREFIX

    return EMBEDDINGGEMMA_QUERY_PREFIXES[task]


def _get_instruction(
    model_name: str,
    mode: Literal["code", "query"],
    semantic_task: SemanticTask,
) -> str:
    """Get default instruction prefix for model/task/mode.

    :param model_name: Model identifier.
    :param mode: Input mode.
    :param semantic_task: Normalized task.
    :return: Instruction prefix.
    """
    profile = resolve_model_profile(model_name)

    if profile.family == "c2llm":
        if mode == "query":
            return C2LLM_INSTRUCTIONS["query"]
        return C2LLM_INSTRUCTIONS["code"]

    if profile.family == "embeddinggemma":
        return _get_embeddinggemma_prefix(semantic_task, mode)

    return ""


def _resolve_instruction_prefix(
    model_name: str,
    mode: Literal["code", "query"],
    instruction_prefix: str | None,
    *,
    semantic_task: SemanticTask,
) -> str:
    """Resolve instruction prefix override for embedding inputs.

    :param model_name: Model identifier.
    :param mode: Input mode.
    :param instruction_prefix: Optional override.
    :param semantic_task: Resolved task.
    :return: Instruction prefix.
    """
    if instruction_prefix is not None:
        return instruction_prefix
    return _get_instruction(model_name, mode, semantic_task)


def prepare_code_for_embedding(
    unit: CodeUnit,
    model_name: str = DEFAULT_MODEL,
    mode: Literal["code", "query"] = "code",
    instruction_prefix: str | None = None,
    semantic_task: str | None = None,
) -> str:
    """Prepare code unit for embedding.

    :param unit: Source unit to embed.
    :param model_name: Model identifier.
    :param mode: Embedding mode.
    :param instruction_prefix: Optional explicit instruction.
    :param semantic_task: Optional task override.
    :return: Prefixed source payload.
    """
    source = unit.source.strip()
    task_default = DEFAULT_SEARCH_SEMANTIC_TASK if mode == "query" else DEFAULT_CHECK_SEMANTIC_TASK
    resolved_task = _normalize_semantic_task(
        semantic_task,
        default_task=task_default,  # type: ignore[arg-type]
    )
    instruction = _resolve_instruction_prefix(
        model_name,
        mode,
        instruction_prefix,
        semantic_task=resolved_task,
    )
    return f"{instruction}{source}"


def _encode_texts(
    encode_fn: Callable[..., np.ndarray],
    texts: list[str],
    *,
    batch_size: int,
    show_progress_bar: bool,
    convert_to_numpy: bool,
    normalize_embeddings: bool,
    device: str | None = None,
) -> np.ndarray:
    """Encode texts with defensive kwargs handling across model backends.

    :param encode_fn: Model encode callable.
    :param texts: Input texts to encode.
    :param batch_size: Batch size for encoder.
    :param show_progress_bar: Whether to surface a progress bar.
    :param convert_to_numpy: Return NumPy arrays.
    :param normalize_embeddings: Normalize outputs.
    :param device: Optional execution device.
    :return: Encoded vectors.
    """
    kwargs = {
        "batch_size": batch_size,
        "show_progress_bar": show_progress_bar,
        "convert_to_numpy": convert_to_numpy,
        "normalize_embeddings": normalize_embeddings,
    }
    if device is not None:
        kwargs["device"] = device

    try:
        return encode_fn(texts, **kwargs)
    except TypeError:
        kwargs.pop("device", None)
        return encode_fn(texts, **kwargs)


def compute_embeddings(
    units: list[CodeUnit],
    model_name: str = DEFAULT_MODEL,
    instruction_prefix: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    revision: str | None = None,
    trust_remote_code: bool | None = None,
    semantic_task: str | None = None,
) -> np.ndarray:
    """Compute embeddings for all code units.

    :param units: Units to encode.
    :param model_name: Model identifier.
    :param instruction_prefix: Optional embedding instruction override.
    :param batch_size: Encoding batch size.
    :param revision: Optional model revision.
    :param trust_remote_code: Optional trust setting.
    :param semantic_task: Optional semantic task override.
    :return: Dense embedding matrix.
    """
    resolved_revision = _resolve_model_revision(model_name, revision)
    resolved_trust_remote_code = _resolve_trust_remote_code(model_name, trust_remote_code)
    profile = resolve_model_profile(model_name)
    resolved_task = _normalize_semantic_task(
        semantic_task,
        default_task=DEFAULT_CHECK_SEMANTIC_TASK,
    )
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
            semantic_task=resolved_task,
        )
        texts.append(_truncate_code_if_needed(prepared, unit.qualified_name, model))

    logger.info("Computing embeddings for %d code units", len(texts))
    current_batch_size = max(1, batch_size)
    attempted_cpu_fallback = False
    while True:
        try:
            encode_fn = model.encode
            if profile.family == "embeddinggemma" and hasattr(model, "encode_document"):
                encode_fn = model.encode_document

            embeddings = _encode_texts(
                encode_fn,
                texts,
                batch_size=current_batch_size,
                show_progress_bar=len(texts) > 100,
                convert_to_numpy=True,
                normalize_embeddings=True,
                device="cpu" if attempted_cpu_fallback else None,
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
                _clear_cuda_cache()
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
            _clear_cuda_cache()
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

    :param units: Candidate units in the same order as ``embeddings``.
    :param embeddings: Embedding matrix.
    :param threshold: Minimum cosine similarity.
    :param exclude_exact: Pairs to exclude from semantic output.
    :return: Similar pairs sorted by confidence.
    """
    exclude_exact = exclude_exact or set()
    n = len(units)

    logger.info("Computing pairwise similarities for %d units", n)

    duplicates = []

    def _types_compatible(unit_a: CodeUnit, unit_b: CodeUnit) -> bool:
        """Check whether unit kinds are compatible for semantic comparison.

        :param unit_a: First unit.
        :param unit_b: Second unit.
        :return: ``True`` when types are comparable.
        """
        if unit_a.unit_type == unit_b.unit_type:
            return True
        function_like = {"function", "method"}
        return (
            unit_a.unit_type.name.lower() in function_like
            and unit_b.unit_type.name.lower() in function_like
        )

    chunk_size = 500
    for i in range(0, n, chunk_size):
        end_i = min(i + chunk_size, n)
        chunk_embeddings = embeddings[i:end_i]

        similarities = chunk_embeddings @ embeddings.T

        for local_idx in range(end_i - i):
            global_idx = i + local_idx
            unit_a = units[global_idx]

            for j in range(global_idx + 1, n):
                sim = similarities[local_idx, j]

                if sim < threshold:
                    continue

                unit_b = units[j]

                if not _types_compatible(unit_a, unit_b):
                    continue

                if unit_a.file_path == unit_b.file_path:
                    if not (unit_a.end_lineno < unit_b.lineno or unit_b.end_lineno < unit_a.lineno):
                        continue

                pair_key = ordered_pair_key(unit_a, unit_b)
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

    duplicates.sort(key=lambda x: x.similarity, reverse=True)

    logger.info("Found %d semantic duplicates above threshold %s", len(duplicates), threshold)
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
    threshold: float | None = None,
    semantic_task: str | None = None,
) -> list[tuple[CodeUnit, float]]:
    """Find code units most similar to a natural language query.

    :param query: Search text.
    :param units: Candidate units.
    :param embeddings: Embeddings aligned with ``units``.
    :param model_name: Model identifier.
    :param instruction_prefix: Optional query instruction override.
    :param top_k: Maximum number of results.
    :param revision: Optional model revision.
    :param trust_remote_code: Optional trust setting.
    :param threshold: Optional result cutoff.
    :param semantic_task: Optional semantic task override.
    :return: Ranked results and cosine scores.
    """
    resolved_revision = _resolve_model_revision(model_name, revision)
    resolved_trust_remote_code = _resolve_trust_remote_code(model_name, trust_remote_code)
    profile = resolve_model_profile(model_name)
    resolved_threshold = (
        threshold if threshold is not None else get_default_semantic_threshold(model_name)
    )
    resolved_task = _normalize_semantic_task(
        semantic_task,
        default_task=DEFAULT_SEARCH_SEMANTIC_TASK,
    )
    model = get_model(
        model_name,
        revision=resolved_revision,
        trust_remote_code=resolved_trust_remote_code,
    )

    instruction = _resolve_instruction_prefix(
        model_name,
        "query",
        instruction_prefix,
        semantic_task=resolved_task,
    )
    query_text = f"{instruction}{query}"

    try:
        encode_fn = model.encode
        if profile.family == "embeddinggemma" and hasattr(model, "encode_query"):
            encode_fn = model.encode_query

        query_embedding = _encode_texts(
            encode_fn,
            [query_text],
            batch_size=1,
            show_progress_bar=False,
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

    similarities = embeddings @ query_embedding

    sorted_indices = np.argsort(similarities)[::-1]
    filtered_indices = [idx for idx in sorted_indices if similarities[idx] >= resolved_threshold]
    top_indices = filtered_indices[:top_k]

    return [(units[i], float(similarities[i])) for i in top_indices]


def run_semantic_analysis(
    units: list[CodeUnit],
    model_name: str = DEFAULT_MODEL,
    instruction_prefix: str | None = None,
    threshold: float | None = None,
    exclude_pairs: set[tuple[str, str]] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    revision: str | None = None,
    trust_remote_code: bool | None = None,
    semantic_task: str | None = None,
) -> tuple[np.ndarray, list[DuplicatePair]]:
    """Run full semantic duplicate detection.

    :param units: Units to analyze.
    :param model_name: Model identifier.
    :param instruction_prefix: Optional embedding instruction override.
    :param threshold: Optional similarity threshold.
    :param exclude_pairs: Pairs to skip from output.
    :param batch_size: Encoding batch size.
    :param revision: Optional model revision.
    :param trust_remote_code: Optional trust setting.
    :param semantic_task: Optional semantic task override.
    :return: Embeddings and semantic duplicate pairs.
    """
    if not units:
        return np.array([]), []
    resolved_threshold = (
        threshold if threshold is not None else get_default_semantic_threshold(model_name)
    )

    embeddings = compute_embeddings(
        units,
        model_name=model_name,
        instruction_prefix=instruction_prefix,
        batch_size=batch_size,
        revision=revision,
        trust_remote_code=trust_remote_code,
        semantic_task=semantic_task,
    )
    duplicates = find_semantic_duplicates(
        units, embeddings, threshold=resolved_threshold, exclude_exact=exclude_pairs
    )

    return embeddings, duplicates
