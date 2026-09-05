#!/usr/bin/env python3
"""Dependency-free integrity check for the planted source clone regions."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "fixtures" / "clone-ground-truth.json"


def extract_region(instance: dict[str, object]) -> tuple[str, int, int]:
    path = ROOT / str(instance["file"])
    source = path.read_text(encoding="utf-8")
    start_marker = str(instance["start_marker"])
    end_marker = str(instance["end_marker"])

    if source.count(start_marker) != 1 or source.count(end_marker) != 1:
        raise AssertionError(f"markers must be unique in {path.relative_to(ROOT)}")

    start = source.index(start_marker)
    end = source.index(end_marker, start) + len(end_marker)
    start_line = source.count("\n", 0, start) + 1
    end_line = source.count("\n", 0, end) + 1
    return source[start:end], start_line, end_line


def normalized(source: str) -> str:
    return "".join(source.split())


def levenshtein_similarity(left: str, right: str) -> float:
    denominator = max(len(left), len(right))
    if denominator == 0:
        return 1.0

    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            insertion = current[right_index - 1] + 1
            deletion = previous[right_index] + 1
            substitution = previous[right_index - 1] + (left_character != right_character)
            current.append(min(insertion, deletion, substitution))
        previous = current

    return 1.0 - previous[-1] / denominator


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    groups = {group["id"]: group for group in manifest["clone_groups"]}

    extracted: dict[str, list[str]] = {}
    for group_id, group in groups.items():
        extracted[group_id] = []
        for instance in group["instances"]:
            region, start_line, end_line = extract_region(instance)
            if start_line != instance["start_line"] or end_line != instance["end_line"]:
                raise AssertionError(
                    f"stale line span for {instance['file']}: "
                    f"manifest={instance['start_line']}-{instance['end_line']}, "
                    f"actual={start_line}-{end_line}"
                )
            extracted[group_id].append(region)

    exact_left, exact_right = extracted["exact-border-builder"]
    if exact_left != exact_right:
        raise AssertionError("exact-border-builder is no longer byte-identical")

    edit_left, edit_right = map(normalized, extracted["bubble-renderers"])
    if edit_left == edit_right:
        raise AssertionError("bubble-renderers unexpectedly became exact")
    edit_similarity = levenshtein_similarity(edit_left, edit_right)
    lower, upper = groups["bubble-renderers"]["oracle"]["expected_range"]
    if not lower <= edit_similarity <= upper:
        raise AssertionError(
            f"bubble-renderers similarity {edit_similarity:.3f} is outside {lower}..{upper}"
        )

    semantic_left, semantic_right = map(normalized, extracted["word-wrappers"])
    if semantic_left == semantic_right:
        raise AssertionError("word-wrappers source unexpectedly became identical")

    print("source fixture OK")
    print(f"  exact-border-builder: {len(exact_left)} identical bytes")
    print(f"  bubble-renderers:     {edit_similarity:.3f} normalized similarity")
    print("  word-wrappers:        distinct source regions; behavior is checked by cargo test")


if __name__ == "__main__":
    main()
