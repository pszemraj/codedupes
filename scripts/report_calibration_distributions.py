"""Report per-category embedding-similarity distributions for calibration corpora.

For each (language, model) pair this embeds the corpus once, then summarizes
cosine similarity per labeled category (exact through near_restructure),
for negative controls, and for the background of all other same-language unit
pairs. The distributions show where each model separates clones from
non-clones per language, independent of any threshold grid.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from codedupes.analyzer import AnalyzerConfig, CodeAnalyzer

try:
    from .sweep_common import resolve_label_unit
    from .validate_calibration_corpus import CATEGORY_NAMES
except ImportError:
    from sweep_common import resolve_label_unit
    from validate_calibration_corpus import CATEGORY_NAMES

DEFAULT_LANGUAGES = ("c", "rust", "javascript", "typescript", "python")
DEFAULT_MODELS = ("gte-modernbert-base", "embeddinggemma-300m")


def _summary(values: list[float]) -> dict[str, Any]:
    """Summarize one similarity sample.

    :param list[float] values: Cosine similarities.
    :return dict[str, Any]: Count and percentile summary.
    """
    if not values:
        return {"count": 0}
    array = np.array(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": round(float(array.min()), 4),
        "p25": round(float(np.percentile(array, 25)), 4),
        "median": round(float(np.median(array)), 4),
        "p75": round(float(np.percentile(array, 75)), 4),
        "max": round(float(array.max()), 4),
    }


def _pair_similarities(
    groups: list[list[str]],
    units: list[Any],
    uid_to_row: dict[str, int],
    embeddings: np.ndarray,
) -> tuple[list[float], int]:
    """Score labeled pairs against the embedding matrix.

    :param list[list[str]] groups: Label groups of unit specs.
    :param list[Any] units: Extracted corpus units for spec resolution.
    :param dict[str, int] uid_to_row: Embedding row index per unit uid.
    :param np.ndarray embeddings: Normalized embedding matrix.
    :return tuple[list[float], int]: Similarities and count of unembedded pairs.
    """
    similarities: list[float] = []
    missing = 0
    for group in groups:
        resolved = [resolve_label_unit(units, spec) for spec in group]
        for index_a in range(len(resolved)):
            for index_b in range(index_a + 1, len(resolved)):
                row_a = uid_to_row.get(resolved[index_a].uid)
                row_b = uid_to_row.get(resolved[index_b].uid)
                if row_a is None or row_b is None:
                    missing += 1
                    continue
                similarities.append(float(embeddings[row_a] @ embeddings[row_b]))
    return similarities, missing


def _analyze_language(
    *,
    language: str,
    model_name: str,
    corpus_root: Path,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    """Build the distribution report for one (language, model) pair.

    :param str language: Corpus language key and directory name.
    :param str model_name: Built-in model profile key.
    :param Path corpus_root: Calibration fixture root directory.
    :param str device: Embedding device.
    :param int batch_size: Embedding batch size.
    :return dict[str, Any]: Distribution summary per category.
    """
    config = AnalyzerConfig(
        run_traditional=False,
        run_semantic=True,
        run_unused=False,
        include_private=True,
        languages=(language,),
        model_name=model_name,
        min_semantic_statements=0,
        batch_size=batch_size,
        device=device,
    )
    analyzer = CodeAnalyzer(config)
    result = analyzer.analyze(corpus_root / language)
    embeddings = analyzer._embeddings
    semantic_units = analyzer._semantic_units
    assert embeddings is not None and semantic_units is not None
    uid_to_row = {unit.uid: row for row, unit in enumerate(semantic_units)}

    labels = json.loads((corpus_root / "labels" / f"{language}.json").read_text())
    categories: dict[str, list[list[str]]] = labels["categories"]

    report: dict[str, Any] = {"units": len(result.units), "embedded": len(semantic_units)}
    labeled_rows: set[tuple[int, int]] = set()
    for category in CATEGORY_NAMES:
        groups = categories.get(category, [])
        if not groups:
            continue
        similarities, missing = _pair_similarities(groups, result.units, uid_to_row, embeddings)
        summary = _summary(similarities)
        if missing:
            summary["unembedded_pairs"] = missing
        report[category] = summary
        for group in groups:
            rows = [uid_to_row.get(resolve_label_unit(result.units, spec).uid) for spec in group]
            for index_a in range(len(rows)):
                for index_b in range(index_a + 1, len(rows)):
                    if rows[index_a] is not None and rows[index_b] is not None:
                        labeled_rows.add(
                            (min(rows[index_a], rows[index_b]), max(rows[index_a], rows[index_b]))
                        )

    negative_similarities, negative_missing = _pair_similarities(
        labels.get("negative_controls", []), result.units, uid_to_row, embeddings
    )
    negative_summary = _summary(negative_similarities)
    if negative_missing:
        negative_summary["unembedded_pairs"] = negative_missing
    report["negative_controls"] = negative_summary

    # Background: every same-language candidate pair that is not labeled positive.
    matrix = embeddings @ embeddings.T
    background: list[float] = []
    for row_a in range(len(semantic_units)):
        for row_b in range(row_a + 1, len(semantic_units)):
            if (row_a, row_b) not in labeled_rows:
                background.append(float(matrix[row_a, row_b]))
    report["background"] = _summary(background)
    return report


def main() -> int:
    """Entry point.

    :return int: Process exit code.
    """
    parser = argparse.ArgumentParser(
        description="Summarize per-category embedding similarity for calibration corpora."
    )
    parser.add_argument(
        "--corpus-root", type=Path, default=Path("test_fixtures/polyglot_calibration")
    )
    parser.add_argument("--languages", nargs="*", default=list(DEFAULT_LANGUAGES))
    parser.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("test_fixtures/polyglot_calibration/reports/similarity_distributions.json"),
    )
    args = parser.parse_args()

    payload: dict[str, Any] = {}
    for model_name in args.models:
        payload[model_name] = {}
        for language in args.languages:
            report = _analyze_language(
                language=language,
                model_name=model_name,
                corpus_root=args.corpus_root,
                device=args.device,
                batch_size=args.batch_size,
            )
            payload[model_name][language] = report
            print(f"\n== {model_name} / {language} ==")
            for key, value in report.items():
                print(f"  {key}: {value}")

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote distribution report: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
