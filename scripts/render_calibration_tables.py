"""Render checked calibration metrics as documentation tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = REPO / "test_fixtures/calibration/calibration-results.json"
MODELS = (("gte-modernbert-base", "GTE"), ("embeddinggemma-300m", "Gemma"))
LANGUAGES = (
    ("python", "Python"),
    ("rust", "Rust"),
    ("c", "C"),
    ("javascript", "JavaScript"),
    ("typescript", "TypeScript"),
)
DIFFICULTIES = ("easy", "medium", "hard")


def _percent(value: float) -> str:
    return f"{value:.1%}"


def _threshold_models(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {entry["model"]: entry for entry in payload["threshold_selection"]["models"]}


def render_difficulty_table(payload: dict[str, Any]) -> str:
    """Render selected duplicate recall by language and difficulty."""
    models = _threshold_models(payload)
    recalls = {
        model: {
            entry["language"]: entry["selected_difficulty_recall"]
            for entry in models[model]["duplicate_by_language"]
        }
        for model, _label in MODELS
    }
    totals = {
        (model, difficulty): [0, 0] for model, _label in MODELS for difficulty in DIFFICULTIES
    }
    lines = [
        "| language | GTE easy | GTE medium | GTE hard | Gemma easy | Gemma medium | Gemma hard |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for language, label in LANGUAGES:
        cells = []
        for model, _model_label in MODELS:
            for difficulty in DIFFICULTIES:
                item = recalls[model][language][difficulty]
                detected, total = item["detected"], item["total"]
                totals[model, difficulty][0] += detected
                totals[model, difficulty][1] += total
                cells.append(f"{detected}/{total}" if total else "-")
        lines.append(f"| {label} | {' | '.join(cells)} |")

    counts = [f"{detected}/{total}" if total else "-" for detected, total in totals.values()]
    percentages = [
        _percent(detected / total) if total else "-" for detected, total in totals.values()
    ]
    lines.extend(
        (
            f"| **Overall** | **{'** | **'.join(counts)}** |",
            f"| **Recall** | **{'** | **'.join(percentages)}** |",
        )
    )
    return "\n".join(lines)


def _pooled_metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    tp = sum(row["tp"] for row in rows)
    fp = sum(row["fp"] for row in rows)
    fn = sum(row["fn"] for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def _metrics_row(output: str, model: str, metrics: dict[str, Any]) -> str:
    counts = f"{metrics['tp']} / {metrics['fp']} / {metrics['fn']}"
    ratios = " | ".join(_percent(metrics[field]) for field in ("precision", "recall", "f1"))
    return f"| {output} | {model} | {counts} | {ratios} |"


def render_pooled_metrics_table(payload: dict[str, Any]) -> str:
    """Render pooled admission, visibility, and search metrics."""
    threshold = _threshold_models(payload)
    hybrid = {entry["model"]: entry for entry in payload["hybrid_selection"]["models"]}
    lines = [
        "| output | model | TP / FP / FN | precision | recall | F1 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for model, label in MODELS:
        metrics = [entry["selected_metrics"] for entry in threshold[model]["duplicate_by_language"]]
        lines.append(_metrics_row("duplicate admission", label, _pooled_metrics(metrics)))
    for model, label in MODELS:
        lines.append(
            _metrics_row(
                "semantic-only default-visible", label, hybrid[model]["selected"]["metrics"]
            )
        )
    for model, label in MODELS:
        lines.append(_metrics_row("search", label, threshold[model]["search"]["selected_metrics"]))
    return "\n".join(lines)


def main() -> int:
    """Print checked documentation tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", nargs="?", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    payload = json.loads(args.report.read_text(encoding="utf-8"))
    print(render_difficulty_table(payload))
    print()
    print(render_pooled_metrics_table(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
