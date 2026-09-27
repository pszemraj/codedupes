"""Vector-space compatibility: family/profile/dtype changes and dimension-mismatched cache reuse."""

from __future__ import annotations

import numpy as np

from codedupes import embedding_cache, semantic
from codedupes.analyzer import AnalyzerConfig, CodeAnalyzer
from codedupes.embedding_cache import EmbeddingCache
from codedupes.semantic import (
    EmbeddingRunStats,
    compute_embeddings,
    compute_embeddings_with_identity,
)
from tests.embedding_cache_helpers import (
    CountingModel,
    five_units,
    patch_get_model,
    vector_for_text,
)


class _SimilarityModel(CountingModel):
    """Two-dimensional fake with a fixed similarity to matching inputs."""

    def __init__(self, matching_text: str, similarity: float) -> None:
        super().__init__(dim=2)
        self.matching_text = matching_text
        self.similarity = similarity

    def encode(self, texts, **kwargs):
        text_list = self._record_encode_call(texts, **kwargs)
        return np.array(
            [
                [1.0, 0.0]
                if self.matching_text in text
                else [self.similarity, np.sqrt(1 - self.similarity**2)]
                for text in text_list
            ],
            dtype=np.float32,
        )


def test_local_family_threshold_changes_reuse_embeddings(tmp_path, monkeypatch):
    model_dir = tmp_path / "approved-copy"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        '{"model_type": "gemma3_text", "use_bidirectional_attention": true}', encoding="utf-8"
    )
    (model_dir / "model.safetensors").write_bytes(b"weights")
    project = tmp_path / "project"
    project.mkdir()
    (project / "arithmetic.py").write_text(
        "def alpha(x):\n    return x + 1\n\ndef beta(x):\n    return x * 2\n", encoding="utf-8"
    )

    model = _SimilarityModel("alpha", 0.78)
    loads = patch_get_model(monkeypatch, model)
    settings = {
        "model_name": str(model_dir),
        "device": "cpu",
        "min_semantic_statements": 0,
        "run_traditional": False,
        "run_unused": False,
    }
    tuned = CodeAnalyzer(AnalyzerConfig(**settings)).analyze(project)
    assert len(tuned.semantic_duplicates) == 1
    calls = len(model.encode_calls)
    assert calls == 1
    (model_dir / "README.md").write_text("Approved workplace copy", encoding="utf-8")
    generic = CodeAnalyzer(AnalyzerConfig(threshold_profile="generic", **settings)).analyze(project)
    assert generic.semantic_duplicates == []
    assert generic.embedding_stats.cache_hit_rows == 2
    assert len(model.encode_calls) == calls
    assert loads["count"] == 1
    assert tuned.embedding_stats.cache_revision == generic.embedding_stats.cache_revision


def test_local_model_card_family_changes_split_only_prompt_sensitive_cache(tmp_path, monkeypatch):
    units = five_units(tmp_path)
    model_dir = tmp_path / "local-model-copy"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type": "unknown"}', encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"weights")
    readme = model_dir / "README.md"
    readme.write_text("# local checkpoint\n", encoding="utf-8")

    model = CountingModel()
    patch_get_model(monkeypatch, model)

    generic_stats = EmbeddingRunStats()
    _, generic_identity = compute_embeddings_with_identity(
        units,
        model_name=str(model_dir),
        device="cpu",
        cache_scope=tmp_path,
        stats=generic_stats,
    )

    # Model cards are excluded from the local content fingerprint, but this
    # heading changes the code encode plan from no prompt to EmbeddingGemma's
    # semantic-similarity prompt, so the vectors must not be reused.
    readme.write_text("# embeddinggemma-300m\n", encoding="utf-8")
    gemma_stats = EmbeddingRunStats()
    _, gemma_identity = compute_embeddings_with_identity(
        units,
        model_name=str(model_dir),
        device="cpu",
        cache_scope=tmp_path,
        stats=gemma_stats,
    )

    assert generic_identity.resolved_revision == gemma_identity.resolved_revision
    assert generic_identity != gemma_identity
    assert generic_identity.runtime_variant != gemma_identity.runtime_variant
    assert gemma_stats.cache_hit_rows == 0
    assert gemma_stats.encoded_inputs == len(units)
    assert len(model.encode_calls) == 2
    assert model.prompts_seen == [
        None,
        semantic.EMBEDDINGGEMMA_QUERY_PREFIXES["semantic-similarity"],
    ]

    # GTE and the original generic profile both encode code symmetrically
    # without a prompt, so their full embedding identities agree and the
    # original vectors can be reused despite the same README-only edit.
    readme.write_text("# gte-modernbert-base\n", encoding="utf-8")
    gte_stats = EmbeddingRunStats()
    _, gte_identity = compute_embeddings_with_identity(
        units,
        model_name=str(model_dir),
        device="cpu",
        cache_scope=tmp_path,
        stats=gte_stats,
    )

    assert gte_identity == generic_identity
    assert gte_identity.resolved_revision == gemma_identity.resolved_revision
    assert gte_stats.cache_hit_rows == len(units)
    assert gte_stats.encoded_inputs == 0
    assert gte_stats.model_loaded is False
    assert len(model.encode_calls) == 2


def test_search_profile_changes_reuse_corpus_and_query_vectors(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    (project / "arithmetic.py").write_text("def alpha(x):\n    return x + 1\n", encoding="utf-8")

    model = _SimilarityModel("def alpha", 0.60)
    loads = patch_get_model(monkeypatch, model)
    config = AnalyzerConfig(
        mode="search",
        device="cpu",
        min_semantic_statements=0,
        run_traditional=False,
        run_unused=False,
    )
    analyzer = CodeAnalyzer(config)
    analyzer.index(project)
    assert analyzer.search("addition") == []  # GTE search default is 0.68.
    config.threshold_profile = "embeddinggemma-300m"
    assert len(analyzer.search("addition")) == 1
    config.threshold_profile = "generic"
    analyzer.index(project)
    assert len(analyzer.search("addition")) == 1
    assert analyzer.search("addition", threshold=0.61) == []
    assert len(model.encode_calls) == 2
    assert loads["count"] == 2
    assert model.prompts_seen == [None, None]  # Threshold choices do not select Gemma prompts.


def test_dimension_change_warns_before_replacing_incompatible_shard(tmp_path, caplog):
    cache = EmbeddingCache()
    scope = tmp_path / "proj"
    scope.mkdir()
    cache.put_many(
        scope,
        "model-a",
        "rev1",
        [
            ("first", np.array([1.0, 2.0], dtype=np.float32)),
            ("second", np.array([3.0, 4.0], dtype=np.float32)),
        ],
    )
    replacement = np.array([5.0, 6.0, 7.0], dtype=np.float32)

    with caplog.at_level("WARNING", logger="codedupes.embedding_cache"):
        cache.put_many(scope, "model-a", "rev1", [("replacement", replacement)])

    hits = cache.get_many(scope, "model-a", "rev1", ["first", "second", "replacement"])
    assert set(hits) == {"replacement"}
    np.testing.assert_array_equal(hits["replacement"], replacement)
    assert "vector dimension changed from 2 to 3" in caplog.text
    assert "replacing all 2 entries" in caplog.text


def test_dim_mismatched_hits_after_revision_correction_recover(tmp_path, monkeypatch):
    units = five_units(tmp_path)
    model = CountingModel(dim=4)
    patch_get_model(monkeypatch, model)
    monkeypatch.setattr(semantic, "_resolve_hf_cached_revision", lambda _model: "rev-a")
    monkeypatch.setattr(semantic, "_get_loaded_model_commit_hash", lambda _model: "rev-b")

    stale_text = units[0].source.strip()
    stale_variant = semantic._cache_variant_for(
        semantic.resolve_model_profile("test-model"),
        "auto",
        semantic.resolve_encode_plan("test-model", mode="code"),
        mps_fallback=None,
    )
    stale_key = embedding_cache.compute_cache_key(
        "test-model", "rev-b", stale_text, variant=stale_variant
    )
    EmbeddingCache().put_many(
        tmp_path, "test-model", "rev-b", [(stale_key, np.array([9.0, 9.0], dtype=np.float32))]
    )

    result = compute_embeddings(units, model_name="test-model", cache_scope=tmp_path)
    assert result.shape == (5, 4)
    assert np.isfinite(result).all()
    np.testing.assert_array_equal(result[0], vector_for_text(stale_text))
