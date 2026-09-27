"""Local model directory fingerprinting and its walk-scope memoization."""

from __future__ import annotations

from pathlib import Path

import pytest

from codedupes import semantic


def test_fingerprint_local_model_dir_follows_symlinked_subdirectories(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")

    real_shards_dir = tmp_path / "real-shards"
    real_shards_dir.mkdir()
    shard_path = real_shards_dir / "model-00001.safetensors"
    shard_path.write_text("weights-v1")
    (model_dir / "shards").symlink_to(real_shards_dir, target_is_directory=True)

    before = semantic._fingerprint_local_model_dir(model_dir, persist_manifest=False)

    shard_path.write_text("weights-v2-changed")

    after = semantic._fingerprint_local_model_dir(model_dir, persist_manifest=False)

    assert before is not None
    assert after is not None
    assert before != after


@pytest.mark.parametrize(
    "relative",
    [
        "README.md",
        "README",
        "ReadMe.txt",
        "LICENSE",
        "license.txt",
        "NOTICE",
        "Notice.txt",
        "notes.MD",
        "guide.rst",
        ".git/config",
        ".gitignore",
        ".gitattributes",
        ".cache/huggingface/download/model.metadata",
    ],
)
def test_local_fingerprint_ignores_documentation_and_metadata(tmp_path, relative) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    before = semantic._fingerprint_local_model_dir(tmp_path, persist_manifest=False)
    metadata = tmp_path / relative
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text("metadata", encoding="utf-8")
    assert semantic._fingerprint_local_model_dir(tmp_path, persist_manifest=False) == before
    metadata.write_text("changed", encoding="utf-8")
    assert semantic._fingerprint_local_model_dir(tmp_path, persist_manifest=False) == before
    metadata.unlink()
    assert semantic._fingerprint_local_model_dir(tmp_path, persist_manifest=False) == before


@pytest.mark.parametrize(
    "relative",
    [
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "tokenizer.model",
        "vocab.txt",
        "tokenizer_config.json",
        "modules.json",
        "config_sentence_transformers.json",
        "1_Pooling/config.json",
        "2_Dense/model.safetensors",
        "3_Dense/config.json",
        "weights/model-00001-of-00002.safetensors",
        "modeling_custom.py",
        "license_head.safetensors",
        "notice_tokens.json",
        "readme_encoder.py",
        "LICENSE.safetensors",
        "NOTICE.json",
        "README.py",
    ],
)
def test_local_fingerprint_tracks_embedding_assets(tmp_path, relative) -> None:
    asset = tmp_path / relative
    asset.parent.mkdir(parents=True, exist_ok=True)
    asset.write_bytes(b"before")
    before = semantic._fingerprint_local_model_dir(tmp_path, persist_manifest=False)
    asset.write_bytes(b"after!")
    assert semantic._fingerprint_local_model_dir(tmp_path, persist_manifest=False) != before


def test_fingerprint_local_model_dir_handles_symlink_cycles(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    (model_dir / "loop").symlink_to(model_dir, target_is_directory=True)

    fingerprint = semantic._fingerprint_local_model_dir(model_dir, persist_manifest=False)

    assert fingerprint is not None


# --- Finding 1: memoized identity-path fingerprint --------------------------


def _counting_fingerprint_stub(monkeypatch, return_value="dir-stub"):
    """Patch the raw walker with a call-counting stub and return the counter list."""
    calls: list[Path] = []

    def _stub(model_dir: Path, *, persist_manifest: bool = True) -> str | None:
        calls.append(model_dir)
        return return_value

    monkeypatch.setattr(semantic, "_fingerprint_local_model_dir", _stub)
    return calls


def test_fingerprint_local_model_dir_cached_walks_once_within_an_open_scope(
    tmp_path: Path, monkeypatch
) -> None:
    model_dir = tmp_path / "model"
    calls = _counting_fingerprint_stub(monkeypatch)

    with semantic._local_model_fingerprint_walk_scope():
        first = semantic._fingerprint_local_model_dir_cached(model_dir)
        second = semantic._fingerprint_local_model_dir_cached(model_dir)

    assert first == "dir-stub"
    assert second == "dir-stub"
    assert len(calls) == 1


def test_fingerprint_local_model_dir_cached_walks_fresh_outside_any_scope(
    tmp_path: Path, monkeypatch
) -> None:
    """No active scope (the default) must match ``_fingerprint_local_model_dir`` exactly.

    This is the safe-by-default behavior every external caller of
    :func:`resolve_embedding_space_identity`/``_resolve_revision_for_cache``
    relies on: without an explicitly opened call scope, every lookup walks.
    """
    assert semantic._local_model_fingerprint_scope is None
    model_dir = tmp_path / "model"
    calls = _counting_fingerprint_stub(monkeypatch)

    semantic._fingerprint_local_model_dir_cached(model_dir)
    semantic._fingerprint_local_model_dir_cached(model_dir)

    assert len(calls) == 2


def test_fingerprint_local_model_dir_cached_recomputes_across_separate_scopes(
    tmp_path: Path, monkeypatch
) -> None:
    """Each top-level scope starts empty: nothing survives from an earlier one."""
    model_dir = tmp_path / "model"
    calls = _counting_fingerprint_stub(monkeypatch)

    with semantic._local_model_fingerprint_walk_scope():
        semantic._fingerprint_local_model_dir_cached(model_dir)
    with semantic._local_model_fingerprint_walk_scope():
        semantic._fingerprint_local_model_dir_cached(model_dir)

    assert len(calls) == 2


def test_fingerprint_local_model_dir_cached_recomputes_for_a_different_path(
    tmp_path: Path, monkeypatch
) -> None:
    calls = _counting_fingerprint_stub(monkeypatch)

    with semantic._local_model_fingerprint_walk_scope():
        semantic._fingerprint_local_model_dir_cached(tmp_path / "model-a")
        semantic._fingerprint_local_model_dir_cached(tmp_path / "model-b")

    assert len(calls) == 2


def test_local_model_fingerprint_walk_scope_is_reentrant(tmp_path: Path, monkeypatch) -> None:
    """A nested scope open (recursion under the same lock) extends the outer one."""
    model_dir = tmp_path / "model"
    calls = _counting_fingerprint_stub(monkeypatch)

    with semantic._local_model_fingerprint_walk_scope():
        semantic._fingerprint_local_model_dir_cached(model_dir)
        with semantic._local_model_fingerprint_walk_scope():
            # Reuses the outer scope's memo instead of starting a fresh one.
            semantic._fingerprint_local_model_dir_cached(model_dir)
        # The inner "with" must not have torn down the outer scope.
        assert semantic._local_model_fingerprint_scope is not None
        semantic._fingerprint_local_model_dir_cached(model_dir)

    assert semantic._local_model_fingerprint_scope is None
    assert len(calls) == 1


def test_remember_local_model_fingerprint_in_scope_seeds_without_a_walk(
    tmp_path: Path, monkeypatch
) -> None:
    model_dir = tmp_path / "model"
    calls = _counting_fingerprint_stub(monkeypatch)

    with semantic._local_model_fingerprint_walk_scope():
        semantic._remember_local_model_fingerprint_in_scope(model_dir, "dir-from-load")
        result = semantic._fingerprint_local_model_dir_cached(model_dir)

    assert result == "dir-from-load"
    assert calls == []


def test_remember_local_model_fingerprint_in_scope_is_a_no_op_without_a_scope(
    tmp_path: Path, monkeypatch
) -> None:
    assert semantic._local_model_fingerprint_scope is None
    model_dir = tmp_path / "model"
    calls = _counting_fingerprint_stub(monkeypatch)

    semantic._remember_local_model_fingerprint_in_scope(model_dir, "dir-from-load")
    result = semantic._fingerprint_local_model_dir_cached(model_dir)

    assert result == "dir-stub"
    assert len(calls) == 1


def test_resolve_embedding_space_identity_shares_one_walk_within_an_open_scope(
    tmp_path: Path, monkeypatch
) -> None:
    """Repeated identity resolution inside one open scope shares a single walk."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type": "test"}')
    (model_dir / "model.safetensors").write_text("weights")

    walk_calls = _counting_fingerprint_stub(monkeypatch, return_value="dir-once")

    with semantic._local_model_fingerprint_walk_scope():
        first = semantic.resolve_embedding_space_identity(model_name=str(model_dir), device="cpu")
        second = semantic.resolve_embedding_space_identity(model_name=str(model_dir), device="cpu")

    assert first == second
    assert first.resolved_revision == "dir-once"
    assert len(walk_calls) == 1


def test_resolve_embedding_space_identity_detects_disk_changes_across_separate_calls(
    tmp_path: Path,
) -> None:
    """Outside a shared scope, every call must reflect the live on-disk state.

    This is the provenance guarantee finding 1 must preserve: without an
    explicitly opened call scope (the state every caller outside
    search()/compute_embeddings is in), a directory edit between two calls
    must never be masked by a leftover memoized fingerprint.
    """
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type": "test"}')
    weights_path = model_dir / "model.safetensors"
    weights_path.write_text("weights-v1")

    before = semantic.resolve_embedding_space_identity(model_name=str(model_dir), device="cpu")
    weights_path.write_text("weights-v2-changed")
    after = semantic.resolve_embedding_space_identity(model_name=str(model_dir), device="cpu")

    assert before.resolved_revision != after.resolved_revision
