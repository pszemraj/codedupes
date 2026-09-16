"""Ground-truth contracts must remain independent of detector decisions."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from scripts.calibration_contract import (
    ProjectAnalyzer,
    analyzer_config,
    extract_project,
    load_projects,
    pair_key,
    resolve_annotations,
    resolve_unit,
    text_digest,
    validate_project,
    write_json,
)

pytestmark = pytest.mark.grammar


@pytest.fixture
def project(tmp_path: Path):
    root = tmp_path / "app"
    (root / "src").mkdir(parents=True)
    (root / "tests").mkdir()
    body = "    total = 0\n    for value in values:\n        total += value\n    return total\n"
    (root / "src/a.py").write_text("def first(values):\n" + body)
    (root / "src/b.py").write_text("def second(values):\n" + body)
    (root / "src/c.py").write_text("def third(values):\n" + body)
    (root / "src/__init__.py").write_text("")
    (root / "src/api.h").write_text("int declared(int value);\n")
    (root / "README.md").write_text("Application documentation")
    (root / "tests/test_app.py").write_text("def test_behavior():\n    assert True\n")
    annotations = {"schema_version": 2, "units": [], "pairs": [], "families": [], "probes": []}
    for stem, symbol in [("a", "first"), ("b", "second"), ("c", "third")]:
        annotations["units"].append(
            {
                "id": symbol,
                "selector": {
                    "path": f"src/{stem}.py",
                    "qualified_name": f"{stem}.{symbol}",
                    "kind": "function",
                },
            }
        )
    for a, b in [("first", "second"), ("first", "third"), ("second", "third")]:
        annotations["pairs"].append(
            {
                "id": f"{a}-{b}",
                "a": a,
                "b": b,
                "judgment": "negative" if b == "third" and a == "first" else "positive",
                "tags": ["exact"],
                "scope": "whole_unit",
                "behavior_equivalent": None,
                "contract": "Integer accumulation",
                "equivalence_domain": "integer lists",
                "rationale": "Explicit independent maintenance review",
                "evidence": ["tests/test_app.py::test_behavior"],
            }
        )
    annotations["families"] = [{"id": "family", "members": ["first", "second", "third"]}]
    write_json(tmp_path / "labels.json", annotations)
    write_json(
        tmp_path / "manifest.json",
        {
            "schema_version": 2,
            "policies": {
                "default": {},
                "public": {"include_private": False},
                "classes": {"semantic_unit_types": ["function", "method", "class"]},
                "tests": {"include_tests": True},
            },
            "projects": [
                {
                    "id": "app",
                    "root": "app",
                    "analysis_roots": ["src"],
                    "test_roots": ["tests"],
                    "support_files": ["README.md"],
                    "languages": ["python", "c"],
                    "behavior_tests": [],
                    "split": "development",
                    "split_group": "app",
                    "annotations": "labels.json",
                }
            ],
        },
    )
    return load_projects(tmp_path / "manifest.json")[0]


def test_nontransitive_family_and_deterministic_negative_are_legal(project):
    report = validate_project(project)
    assert report["pending_deterministic"] == []
    assert all(p["traditional_recovery"] for p in report["pairs"])
    assert all(p["exclusion_reason"] == "deterministic_exact_exclusion" for p in report["pairs"])


def test_missing_family_pair_is_not_inferred_positive(project):
    project.annotations["pairs"].pop()
    with pytest.raises(ValueError, match="unreviewed within-family"):
        validate_project(project)


@pytest.mark.parametrize("change", ["duplicate", "unresolved", "legacy", "bad_evidence"])
def test_invalid_annotations_fail_before_model_work(project, change):
    if change == "duplicate":
        project.annotations["pairs"].append(copy.deepcopy(project.annotations["pairs"][0]))
    elif change == "unresolved":
        project.annotations["units"][0]["selector"]["qualified_name"] = "missing"
    elif change == "legacy":
        project.annotations["schema_version"] = 1
        write_json(project.annotations_path, project.annotations)
        with pytest.raises(ValueError, match="schema_version"):
            load_projects(project.manifest_path)
        return
    else:
        project.annotations["pairs"][0]["evidence"] = ["tests/test_app.py::nonexistent"]
    with pytest.raises(ValueError):
        validate_project(project)


def test_search_relevance_is_independent_and_empty_results_are_legal(project):
    project.annotations["probes"] = [
        {
            "id": "related",
            "kind": "behavioral",
            "query": "accumulate values",
            "expected": ["first", "third"],
            "relevance_complete": True,
        },
        {
            "id": "noise",
            "kind": "no_result",
            "query": "encrypt a certificate",
            "expected": [],
            "relevance_complete": True,
        },
    ]
    assert validate_project(project)["probes"] == 2


def test_selectors_distinguish_paths_methods_and_redefinitions(project):
    for directory in ["one", "two"]:
        folder = project.root / "src" / directory
        folder.mkdir()
        (folder / "same.py").write_text(
            "class A:\n    def run(self):\n        return 1\n"
            "class B:\n    def run(self):\n        return 2\n"
        )
    repeated = project.root / "src/repeat.py"
    repeated.write_text("def again():\n    return 1\ndef again():\n    return 2\n")
    units, _ = extract_project(project, inventory=True)
    methods = [u for u in units if u.name == "run"]
    assert len(methods) == 4
    for unit in methods:
        selector = {
            "path": unit.file_path.relative_to(project.root).as_posix(),
            "qualified_name": unit.qualified_name,
            "kind": "method",
        }
        assert resolve_unit(project.root, units, selector) is unit
    repeated_units = [u for u in units if u.name == "again"]
    selector = {
        "path": "src/repeat.py",
        "qualified_name": repeated_units[0].qualified_name,
        "kind": "function",
    }
    with pytest.raises(ValueError, match="matched 2"):
        resolve_unit(project.root, units, selector)
    selector["start_line"] = repeated_units[1].lineno
    assert resolve_unit(project.root, units, selector) is repeated_units[1]
    # Qualified symbols are structured strings, not parsed with a :: delimiter.
    repeated_units[1].qualified_name = "Trait::<T>::again"
    selector["qualified_name"] = "Trait::<T>::again"
    assert resolve_unit(project.root, units, selector) is repeated_units[1]


def test_partial_spans_fail_when_stale(project):
    units, _ = extract_project(project, inventory=True)
    resolved = resolve_annotations(project, units)
    pair = project.annotations["pairs"][0]
    pair["scope"] = "partial"
    pair["regions"] = []
    for identifier in (pair["a"], pair["b"]):
        unit = resolved[identifier]
        source = "".join(unit.file_path.read_text().splitlines(keepends=True)[1:4])
        pair["regions"].append(
            {"unit": identifier, "start_line": 2, "end_line": 4, "sha256": text_digest(source)}
        )
    validate_project(project)
    pair["regions"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="stale region"):
        validate_project(project)


def test_named_policies_use_real_candidate_selection(project):
    (project.root / "src/extra.py").write_text(
        "def _private(x):\n    y = x + 1\n    z = y * 2\n    return z\n"
        "def tiny():\n    return 1\n"
        "class Record:\n    a = 1\n    b = 2\n    c = 3\n"
    )
    default_units, _ = extract_project(project)
    default = ProjectAnalyzer(project, analyzer_config(project))._select_semantic_candidates(
        default_units
    )
    assert "tiny" not in {u.name for u in default}
    assert "Record" not in {u.name for u in default}
    assert "_private" in {u.name for u in default}
    assert "test_behavior" not in {u.name for u in default_units}
    public = load_projects(project.manifest_path, policy="public")[0]
    assert "_private" not in {u.name for u in extract_project(public)[0]}
    classes = load_projects(project.manifest_path, policy="classes")[0]
    selected = ProjectAnalyzer(classes, analyzer_config(classes))._select_semantic_candidates(
        default_units
    )
    assert "Record" in {u.name for u in selected}
    tests = load_projects(project.manifest_path, policy="tests")[0]
    assert "test_behavior" in {u.name for u in extract_project(tests)[0]}


def test_group_split_leak_is_rejected(project):
    manifest = copy.deepcopy(project.manifest)
    duplicate = dict(manifest["projects"][0], id="other", split="evaluation")
    manifest["projects"].append(duplicate)
    write_json(project.manifest_path, manifest)
    with pytest.raises(ValueError, match="split leak"):
        load_projects(project.manifest_path)


def test_pair_key_is_unordered():
    assert pair_key("b", "a") == pair_key("a", "b")
