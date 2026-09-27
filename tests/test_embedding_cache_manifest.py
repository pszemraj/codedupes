"""Corpus selection manifest lifecycle: diffing, GC, orphan collection, and selection limits."""

from __future__ import annotations

import numpy as np
import pytest

from codedupes import embedding_cache
from codedupes.embedding_cache import CorpusSelectionManifest, EmbeddingCache, diff_manifest


def _corpus_manifest(
    units: dict[str, str],
    *,
    orphans: dict[str, int] | None = None,
) -> CorpusSelectionManifest:
    """Build a minimal complete manifest for transition-diff tests."""
    return CorpusSelectionManifest(
        last_seen_generation=1,
        complete_scan=True,
        units=units,
        orphans=dict(orphans or {}),
    )


@pytest.mark.parametrize(
    ("previous", "current", "moved", "deleted", "orphaned"),
    [
        ({"old": "key"}, {"new": "key"}, ["new"], [], set()),
        ({"old": "key"}, {}, [], ["old"], {"key"}),
        ({"old": "old-key"}, {"new": "new-key"}, [], ["old"], {"old-key"}),
        ({"first": "key"}, {"first": "key", "second": "key"}, [], [], set()),
        ({"first": "key", "second": "key"}, {"first": "key"}, [], ["second"], set()),
        ({"first": "key", "second": "key"}, {}, [], ["first", "second"], {"key"}),
        ({"first": "key", "second": "key"}, {"new": "key"}, ["new"], ["first"], set()),
        ({"old": "key"}, {"first": "key", "second": "key"}, ["first"], [], set()),
        (
            {"a.py::python::f::0": "old", "a.py::python::f::40": "shared"},
            {"a.py::python::f::20": "new"},
            [],
            ["a.py::python::f::0"],
            {"old", "shared"},
        ),
        (
            {"a.py::python::f::0": "key"},
            {"b.py::python::f::0": "key", "a.py::python::f::20": "key"},
            [],
            [],
            set(),
        ),
    ],
    ids=[
        "move",
        "delete",
        "edit",
        "identical-body-added",
        "identical-body-deleted",
        "all-identical-bodies-deleted",
        "identical-body-moved-and-deleted",
        "identical-body-moved-and-added",
        "repeated-name-shift-edit-and-delete",
        "same-file-match-before-copy",
    ],
)
def test_diff_manifest_classifies_corpus_transitions(
    previous,
    current,
    moved,
    deleted,
    orphaned,
) -> None:
    diff = diff_manifest(_corpus_manifest(previous), current)

    assert diff.moved == moved
    assert diff.deleted == deleted
    assert diff.orphaned == orphaned


def test_manifest_revert_clears_orphan_age(tmp_path) -> None:
    cache = EmbeddingCache()
    scope = tmp_path / "repo"
    scope.mkdir()
    key = "content-key"
    cache.put_many(scope, "model", "revision", [(key, np.ones(2, dtype=np.float32))])
    cache.publish_corpus_manifest(
        scope,
        "model",
        "revision",
        selection="selection",
        units={"unit": key},
        complete_scan=True,
    )
    orphaned = cache.publish_corpus_manifest(
        scope,
        "model",
        "revision",
        selection="selection",
        units={},
        complete_scan=True,
    )
    restored = cache.publish_corpus_manifest(
        scope,
        "model",
        "revision",
        selection="selection",
        units={"unit": key},
        complete_scan=True,
    )

    assert orphaned is not None
    assert orphaned.orphan_rows_retained == 1
    assert restored is not None
    assert restored.orphan_rows_retained == 0
    assert restored.orphan_rows_collected == 0


def test_manifest_gc_never_collects_query_rows(tmp_path) -> None:
    cache = EmbeddingCache()
    scope = tmp_path / "repo"
    scope.mkdir()
    code_key = "code-key"
    query_key = "query-key"
    vector = np.ones(2, dtype=np.float32)
    cache.put_many(scope, "model", "revision", [(code_key, vector)], namespace="code")
    cache.put_many(scope, "model", "revision", [(query_key, vector)], namespace="query")
    cache.publish_corpus_manifest(
        scope,
        "model",
        "revision",
        selection="selection",
        units={"unit": code_key},
        complete_scan=True,
    )

    published = None
    for _ in range(4):
        published = cache.publish_corpus_manifest(
            scope,
            "model",
            "revision",
            selection="selection",
            units={},
            complete_scan=True,
        )

    assert published is not None
    assert published.orphan_rows_collected == 1
    assert cache.get_many(scope, "model", "revision", [code_key]) == {}
    assert query_key in cache.get_many(scope, "model", "revision", [query_key])


def test_manifest_gc_expires_unrefreshed_selection_pin(tmp_path) -> None:
    cache = EmbeddingCache()
    scope = tmp_path / "repo"
    scope.mkdir()
    key = "shared-key"
    cache.put_many(scope, "model", "revision", [(key, np.ones(2, dtype=np.float32))])
    for selection in ("first", "second"):
        cache.publish_corpus_manifest(
            scope,
            "model",
            "revision",
            selection=selection,
            units={"unit": key},
            complete_scan=True,
        )

    orphaned = cache.publish_corpus_manifest(
        scope,
        "model",
        "revision",
        selection="first",
        units={},
        complete_scan=True,
    )
    assert orphaned is not None
    assert orphaned.orphan_rows_retained == 1
    assert orphaned.orphan_rows_collected == 0

    for expected_generation in (4, 5):
        retained = cache.publish_corpus_manifest(
            scope,
            "model",
            "revision",
            selection="first",
            units={},
            complete_scan=True,
        )
        assert retained is not None
        assert retained.generation == expected_generation
        assert retained.orphan_rows_retained == 1
        assert retained.orphan_rows_collected == 0
    assert key in cache.get_many(scope, "model", "revision", [key])

    published = cache.publish_corpus_manifest(
        scope,
        "model",
        "revision",
        selection="first",
        units={},
        complete_scan=True,
    )

    assert published is not None
    assert published.generation == 6
    assert published.orphan_rows_retained == 0
    assert published.orphan_rows_collected == 1
    assert cache.get_many(scope, "model", "revision", [key]) == {}

    rediscovered = cache.publish_corpus_manifest(
        scope,
        "model",
        "revision",
        selection="second",
        units={},
        complete_scan=True,
    )
    assert rediscovered is not None
    assert len(rediscovered.diff.deleted) == 1
    assert rediscovered.orphan_rows_retained == 0
    assert rediscovered.orphan_rows_collected == 0


def test_manifest_gc_revalidates_references_before_collection(tmp_path, monkeypatch) -> None:
    cache = EmbeddingCache()
    concurrent_cache = EmbeddingCache(cache.cache_root)
    scope = tmp_path / "repo"
    scope.mkdir()
    key = "reintroduced-key"
    cache.put_many(scope, "model", "revision", [(key, np.ones(2, dtype=np.float32))])
    cache.publish_corpus_manifest(
        scope,
        "model",
        "revision",
        selection="selection",
        units={"unit": key},
        complete_scan=True,
    )
    cache.publish_corpus_manifest(
        scope,
        "model",
        "revision",
        selection="selection",
        units={},
        complete_scan=True,
    )
    cache.publish_corpus_manifest(
        scope,
        "model",
        "revision",
        selection="selection",
        units={},
        complete_scan=True,
    )
    cache.publish_corpus_manifest(
        scope,
        "model",
        "revision",
        selection="selection",
        units={},
        complete_scan=True,
    )
    collect_orphans = cache.collect_orphans

    def reintroduce_before_collection(*args, **kwargs):
        concurrent_cache.publish_corpus_manifest(
            scope,
            "model",
            "revision",
            selection="selection",
            units={"unit": key},
            complete_scan=True,
        )
        return collect_orphans(*args, **kwargs)

    monkeypatch.setattr(cache, "collect_orphans", reintroduce_before_collection)

    raced = cache.publish_corpus_manifest(
        scope,
        "model",
        "revision",
        selection="selection",
        units={},
        complete_scan=True,
    )

    assert raced is not None
    assert raced.orphan_rows_collected == 0
    manifest = cache.load_manifest(scope, "model", "revision")
    assert manifest is not None
    assert manifest.selections["selection"].units == {"unit": key}
    assert key in cache.get_many(scope, "model", "revision", [key])


def test_cache_stats_accepts_manifest_without_selections(tmp_path) -> None:
    cache = EmbeddingCache()
    scope = tmp_path / "repo"
    scope.mkdir()
    cache.put_many(
        scope,
        "model",
        "revision",
        [("key", np.ones(2, dtype=np.float32))],
    )
    shard_dir = cache.shard_dir(scope, "model", "revision")
    embedding_cache.atomic_write_json(
        shard_dir / embedding_cache.MANIFEST_FILENAME,
        {
            "schema": embedding_cache.MANIFEST_SCHEMA,
            "generation": 7,
            "selections": {},
        },
    )

    stats = cache.stats()

    assert stats["repos"][0]["last_complete_generation"] == 7


def test_manifest_limits_inactive_selection_baselines(tmp_path) -> None:
    cache = EmbeddingCache()
    scope = tmp_path / "repo"
    scope.mkdir()
    selection_count = embedding_cache.MANIFEST_SELECTION_LIMIT + 3

    for index in range(selection_count):
        cache.publish_corpus_manifest(
            scope,
            "model",
            "revision",
            selection=f"selection-{index:02d}",
            units={"unit": f"key-{index}"},
            complete_scan=True,
        )

    manifest = cache.load_manifest(scope, "model", "revision")
    assert manifest is not None
    assert len(manifest.selections) == embedding_cache.MANIFEST_SELECTION_LIMIT
    assert "selection-00" not in manifest.selections
    assert f"selection-{selection_count - 1:02d}" in manifest.selections


def test_manifest_selection_limit_uses_incomplete_publish_recency(tmp_path) -> None:
    cache = EmbeddingCache()
    scope = tmp_path / "repo"
    scope.mkdir()
    for index in range(embedding_cache.MANIFEST_SELECTION_LIMIT):
        cache.publish_corpus_manifest(
            scope,
            "model",
            "revision",
            selection=f"z-old-{index:02d}",
            units={"unit": f"old-key-{index}"},
            complete_scan=False,
        )

    cache.publish_corpus_manifest(
        scope,
        "model",
        "revision",
        selection="a-recent",
        units={"unit": "recent-key"},
        complete_scan=False,
    )
    cache.publish_corpus_manifest(
        scope,
        "model",
        "revision",
        selection="b-newest",
        units={"unit": "newest-key"},
        complete_scan=False,
    )

    manifest = cache.load_manifest(scope, "model", "revision")
    assert manifest is not None
    assert len(manifest.selections) == embedding_cache.MANIFEST_SELECTION_LIMIT
    assert "a-recent" in manifest.selections
    assert "b-newest" in manifest.selections
    assert "z-old-00" not in manifest.selections


def test_incomplete_manifest_matches_observed_files_by_exact_path(tmp_path) -> None:
    cache = EmbeddingCache()
    scope = tmp_path / "repo"
    scope.mkdir()
    first_path = "/repo/a.py"
    sibling_path = "/repo/a.py::sibling.py"
    first_uid = f"{first_path}::python::a.alpha::0"
    sibling_uid = f"{sibling_path}::python::sibling.beta::0"
    cache.put_many(
        scope,
        "model",
        "revision",
        [
            ("first-key", np.ones(2, dtype=np.float32)),
            ("sibling-key", np.zeros(2, dtype=np.float32)),
        ],
    )
    cache.publish_corpus_manifest(
        scope,
        "model",
        "revision",
        selection="selection",
        units={first_uid: "first-key", sibling_uid: "sibling-key"},
        unit_paths={first_uid: first_path, sibling_uid: sibling_path},
        complete_scan=True,
    )

    narrow = cache.publish_corpus_manifest(
        scope,
        "model",
        "revision",
        selection="selection",
        units={first_uid: "first-key"},
        unit_paths={first_uid: first_path},
        observed_files=(first_path,),
        complete_scan=False,
    )

    assert narrow is not None
    assert narrow.diff.deleted == []
    assert narrow.orphan_rows_retained == 0
    manifest = cache.load_manifest(scope, "model", "revision")
    assert manifest is not None
    assert manifest.selections["selection"].units == {
        first_uid: "first-key",
        sibling_uid: "sibling-key",
    }
