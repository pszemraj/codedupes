"""Versioned, model-independent corpus contracts and production extraction policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

from codedupes.analyzer import AnalyzerConfig, CodeAnalyzer
from codedupes.extractor import CodeExtractor
from codedupes.models import CodeUnit
from codedupes.pairs import ordered_pair_key
from codedupes.traditional import _block_kind, find_exact_pair_keys, jaccard_similarity

REPO = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO / "test_fixtures/calibration/manifest.json"
SCHEMA_VERSION = 2
POLICY_FIELDS = {
    "include_private",
    "include_tests",
    "min_semantic_statements",
    "semantic_unit_types",
    "filter_tiny_traditional",
    "tiny_unit_statement_cutoff",
    "suppress_test_semantic_matches",
}


def text_digest(value: str) -> str:
    """Hash source text without whitespace normalization."""
    return hashlib.sha256(value.encode()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object, rejecting non-object payloads."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")  # noqa: TRY004 -- one schema error type
    return value


def write_json(path: Path, value: Any) -> None:
    """Write reviewable JSON with a final newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def relative_file(root: Path, value: str) -> Path:
    """Resolve a project-relative reference without allowing it to escape the project."""
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError(f"expected a project-relative path: {value!r}")
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"path escapes project: {value}")
    return path


@dataclass
class Project:
    """One application and its authoritative annotations."""

    manifest_path: Path
    manifest: dict[str, Any]
    spec: dict[str, Any]
    annotations: dict[str, Any]
    policy_name: str
    policy: dict[str, Any]

    @property
    def root(self) -> Path:
        return (self.manifest_path.parent / self.spec["root"]).resolve()

    @property
    def id(self) -> str:
        return self.spec["id"]

    @property
    def annotations_path(self) -> Path:
        return self.manifest_path.parent / self.spec["annotations"]


def load_projects(
    manifest_path: Path = DEFAULT_MANIFEST,
    project_ids: list[str] | None = None,
    policy: str = "default",
) -> list[Project]:
    """Load only the current schema; legacy groups are deliberately unsupported."""
    manifest_path = manifest_path.resolve()
    manifest = read_json(manifest_path)
    if (
        type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != SCHEMA_VERSION
    ):
        raise ValueError(
            "expected calibration manifest schema_version=2; legacy formats are unsupported"
        )
    if not isinstance(manifest.get("policies"), dict) or policy not in manifest["policies"]:
        raise ValueError(f"unknown policy: {policy}")
    default_policy = manifest["policies"].get("default")
    selected_policy = manifest["policies"][policy]
    if not isinstance(default_policy, dict) or set(default_policy) - POLICY_FIELDS:
        raise ValueError("invalid policy fields: default")
    if not isinstance(selected_policy, dict) or set(selected_policy) - POLICY_FIELDS:
        raise ValueError(f"invalid policy fields: {policy}")
    chosen = default_policy | selected_policy
    specs = manifest.get("projects")
    if not isinstance(specs, list) or not specs:
        raise ValueError("manifest requires nonempty projects")
    ids = [spec["id"] for spec in specs]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate project ID")
    if project_ids and set(project_ids) - set(ids):
        raise ValueError(f"unknown project IDs: {set(project_ids) - set(ids)}")
    # Group assignment is corpus-wide, even when only one project is requested.
    groups = {}
    for spec in specs:
        for required in (
            "root",
            "analysis_roots",
            "test_roots",
            "support_files",
            "languages",
            "behavior_tests",
            "split",
            "split_group",
            "annotations",
        ):
            if required not in spec:
                raise ValueError(f"{spec['id']}: missing {required}")
        if spec["split"] not in {"development", "evaluation"}:
            raise ValueError("split must be development or evaluation")
        group = spec["split_group"]
        if group in groups and groups[group] != spec["split"]:
            raise ValueError(f"split leak through group {group}")
        groups[group] = spec["split"]
    projects = []
    # A source tree cannot be assigned to both splits under different project IDs.
    roots_by_split = []
    for spec in specs:
        root = (manifest_path.parent / spec["root"]).resolve()
        for old_root, split in roots_by_split:
            if spec["split"] != split and (
                root.is_relative_to(old_root) or old_root.is_relative_to(root)
            ):
                raise ValueError("split leak through shared project source")
        roots_by_split.append((root, spec["split"]))
        annotations = read_json(manifest_path.parent / spec["annotations"])
        if annotations.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"{spec['id']}: annotations must use schema_version=2")
        if "positive_groups" in annotations or "negative_controls" in annotations:
            raise ValueError("legacy group labels are unsupported")
        provenance = annotations.get("provenance")
        if (
            not isinstance(provenance, dict)
            or not isinstance(provenance.get("origin"), str)
            or not provenance["origin"].strip()
            or provenance.get("split") != spec["split"]
            or provenance.get("split_group") != spec["split_group"]
        ):
            raise ValueError(
                f"{spec['id']}: annotation provenance must match manifest split and group"
            )
        if not root.is_dir():
            raise ValueError(f"missing project root: {root}")
        if not project_ids or spec["id"] in project_ids:
            projects.append(Project(manifest_path, manifest, spec, annotations, policy, chosen))
    return projects


def extract_project(
    project: Project, *, inventory: bool = False
) -> tuple[list[CodeUnit], list[Any]]:
    """Extract declared roots, retaining normal production discovery rules."""
    roots = list(project.spec["analysis_roots"])
    include_tests = inventory or project.policy.get("include_tests", False)
    if include_tests:
        roots += project.spec["test_roots"]
    units = {}
    diagnostics = []
    for relative in roots:
        root = relative_file(project.root, relative)
        if not root.exists():
            raise ValueError(f"missing analysis/test root: {relative}")
        extractor = CodeExtractor(
            root if root.is_dir() else root.parent,
            include_private=True if inventory else project.policy.get("include_private", True),
            exclude_patterns=[] if include_tests else None,
            languages=project.spec["languages"],
        )
        found = (
            extractor.extract_all() if root.is_dir() else list(extractor.extract_from_file(root))
        )
        units.update((unit.uid, unit) for unit in found)
        diagnostics.extend(extractor.diagnostics)
    bad = [d for d in diagnostics if d.severity == "error" or "parse" in d.code]
    if bad:
        raise ValueError("extraction diagnostics: " + "; ".join(d.message for d in bad))
    return sorted(units.values(), key=lambda u: (str(u.file_path), u.start_byte)), diagnostics


class ProjectAnalyzer(CodeAnalyzer):
    """Use production analysis with manifest-governed file discovery only."""

    def __init__(self, project: Project, config: AnalyzerConfig) -> None:
        super().__init__(config)
        self.project = project

    def _extract_corpus_units(self, path: Path) -> list[CodeUnit]:
        units, diagnostics = extract_project(self.project)
        self._extraction_diagnostics = diagnostics
        self._python_files = sorted({u.file_path for u in units if u.language == "python"})
        return units


def analyzer_config(project: Project, *, semantic: bool = True, **kwargs: Any) -> AnalyzerConfig:
    """Build production configuration from a named extraction/candidate policy."""
    settings = {k: v for k, v in project.policy.items() if k != "include_tests"}
    if not semantic:
        for key in (
            "min_semantic_statements",
            "semantic_unit_types",
            "suppress_test_semantic_matches",
        ):
            settings.pop(key, None)
    elif "semantic_unit_types" in settings:
        settings["semantic_unit_types"] = tuple(settings["semantic_unit_types"])
    settings.update(kwargs)
    return AnalyzerConfig(
        run_semantic=semantic,
        run_unused=False,
        languages=tuple(project.spec["languages"]),
        **settings,
    )


def resolve_unit(root: Path, units: list[CodeUnit], selector: dict[str, Any]) -> CodeUnit:
    """Resolve a structured selector exactly once, never selecting the first match."""
    for field in ("path", "qualified_name", "kind"):
        if not isinstance(selector.get(field), str) or not selector[field]:
            raise ValueError(f"unit selector requires {field}")
    path = relative_file(root, selector["path"])
    matches = [
        u
        for u in units
        if u.file_path.resolve() == path
        and u.qualified_name == selector["qualified_name"]
        and u.unit_type.name.lower() == selector["kind"]
        and ("start_line" not in selector or u.lineno == selector["start_line"])
    ]
    if len(matches) != 1:
        raise ValueError(f"selector {selector} matched {len(matches)} units; expected exactly one")
    return matches[0]


def pair_key(a: str, b: str) -> tuple[str, str]:
    """Canonicalize an annotation-level unordered pair."""
    return tuple(sorted((a, b)))


def resolve_annotations(project: Project, units: list[CodeUnit]) -> dict[str, CodeUnit]:
    """Validate judgments, selectors, evidence, family closure, and checked regions."""
    data = project.annotations
    for field in ("units", "pairs", "families", "probes"):
        if not isinstance(data.get(field), list):
            raise ValueError(f"annotations require a {field} list")  # noqa: TRY004
    resolved = {}
    used_uids = set()
    for record in data["units"]:
        identifier = record["id"]
        if identifier in resolved:
            raise ValueError(f"duplicate unit ID: {identifier}")
        unit = resolve_unit(project.root, units, record["selector"])
        if unit.uid in used_uids:
            raise ValueError(f"multiple annotation IDs select the same unit: {identifier}")
        resolved[identifier] = unit
        used_uids.add(unit.uid)
    pairs = set()
    pair_ids = set()
    for pair in data["pairs"]:
        key = pair_key(pair["a"], pair["b"])
        if key[0] == key[1] or not set(key) <= resolved.keys():
            raise ValueError(f"invalid pair endpoints: {key}")
        if pair["id"] in pair_ids or key in pairs:
            raise ValueError(f"duplicate/conflicting pair: {key}")
        pair_ids.add(pair["id"])
        pairs.add(key)
        if pair.get("judgment") not in {"positive", "negative", "ambiguous"}:
            raise ValueError(f"{key}: missing explicit maintenance judgment")
        if pair["judgment"] == "positive" and pair.get("difficulty") not in {
            "easy",
            "medium",
            "hard",
        }:
            raise ValueError(f"{key}: positive pair requires easy/medium/hard difficulty")
        for field in ("rationale", "contract", "equivalence_domain"):
            if not isinstance(pair.get(field), str) or not pair[field].strip():
                raise ValueError(f"{key}: missing {field}")
        if (
            pair.get("behavior_equivalent") is not None
            and type(pair["behavior_equivalent"]) is not bool
        ):
            raise ValueError(f"{key}: behavior_equivalent must be boolean or null")
        if not isinstance(pair.get("tags"), list) or not pair["tags"]:
            raise ValueError(f"{key}: transformation tags required")
        if pair.get("scope") not in {"whole_unit", "partial"}:
            raise ValueError(f"{key}: invalid duplication scope")
        if pair.get("behavior_equivalent") is False and not pair.get("difference_witness"):
            raise ValueError(f"{key}: observable difference witness required")
        evidence = pair.get("evidence", [])
        if not evidence:
            raise ValueError(f"{key}: behavior evidence required")
        for reference in evidence:
            file, separator, symbol = reference.partition("::")
            evidence_path = relative_file(project.root, file)
            if (
                not separator
                or not evidence_path.is_file()
                or symbol not in evidence_path.read_text()
            ):
                raise ValueError(f"{key}: invalid evidence reference {reference}")
        regions = pair.get("regions", [])
        if pair["scope"] == "partial" and {r["unit"] for r in regions} != set(key):
            raise ValueError(f"{key}: partial pair needs regions in both enclosing units")
        for region in regions:
            if region["unit"] not in key:
                raise ValueError(f"{key}: region references a different unit")
            unit = resolved[region["unit"]]
            start, end = region["start_line"], region["end_line"]
            if not unit.lineno <= start <= end <= unit.end_lineno:
                raise ValueError(f"{key}: region outside enclosing unit")
            source = "".join(unit.file_path.read_text().splitlines(keepends=True)[start - 1 : end])
            if text_digest(source) != region["sha256"]:
                raise ValueError(f"{key}: stale region fingerprint")
        for signal, actual in (
            (
                "identifier_jaccard",
                jaccard_similarity(resolved[key[0]].identifiers, resolved[key[1]].identifiers),
            ),
        ):
            band = pair.get("expected_bands", {}).get(signal)
            if band is not None and not band[0] <= actual <= band[1]:
                raise ValueError(f"{key}: {signal} outside declared band")
    family_ids = set()
    for family in data["families"]:
        if family["id"] in family_ids:
            raise ValueError("duplicate family ID")
        family_ids.add(family["id"])
        members = family["members"]
        if len(members) != len(set(members)) or not set(members) <= resolved.keys():
            raise ValueError(f"invalid family members: {family['id']}")
        missing = {pair_key(*p) for p in combinations(members, 2)} - pairs
        if missing:
            raise ValueError(f"unreviewed within-family pairs in {family['id']}: {sorted(missing)}")
    probe_ids = set()
    for probe in data["probes"]:
        if (
            probe["id"] in probe_ids
            or not isinstance(probe.get("query"), str)
            or not probe["query"].strip()
        ):
            raise ValueError("invalid or duplicate search probe")
        probe_ids.add(probe["id"])
        if probe.get("kind") not in {"behavioral", "partial_symbol", "exact_symbol", "no_result"}:
            raise ValueError("invalid probe kind")
        expected = probe.get("expected")
        if (
            not isinstance(expected, list)
            or not set(expected) <= resolved.keys()
            or len(expected) != len(set(expected))
        ):
            raise ValueError("invalid search expected set")
        if probe["kind"] == "no_result" and expected:
            raise ValueError("no-result probe has expected targets")
        if probe.get("relevance_complete") is not True:
            raise ValueError("search relevance must be reviewed over the declared project scope")
    return resolved


def unit_ids(
    project: Project, units: list[CodeUnit], resolved: dict[str, CodeUnit]
) -> dict[str, str]:
    """Map runtime UIDs to portable annotation IDs or explicit inventory references."""
    mapping = {u.uid: identifier for identifier, u in resolved.items()}
    for unit in units:
        mapping.setdefault(
            unit.uid,
            f"{project.id}/{unit.file_path.relative_to(project.root).as_posix()}"
            f"::{unit.qualified_name}::{unit.unit_type.name.lower()}::{unit.lineno}",
        )
    return mapping


def eligibility_reason(
    a: CodeUnit,
    b: CodeUnit,
    candidate_uids: set[str],
    exact_pairs: set[tuple[str, str]],
    *,
    cross_language: bool = False,
    suppress_tests: bool = False,
) -> str | None:
    """Explain pair exclusion using the existing production comparison contract."""
    if a.uid not in candidate_uids or b.uid not in candidate_uids:
        return "endpoint_not_embedded"
    if not cross_language and a.language != b.language:
        return "different_language"
    if _block_kind(a.unit_type) != _block_kind(b.unit_type):
        return "incompatible_kinds"
    if a.overlaps(b):
        return "overlapping_units"
    if ordered_pair_key(a, b) in exact_pairs:
        return "deterministic_exact_exclusion"
    if suppress_tests and (a.name.startswith("test_") or b.name.startswith("test_")):
        return "test_match_suppressed"
    return None


def validate_project(project: Project, *, require_adjudicated: bool = True) -> dict[str, Any]:
    """Validate annotations and report policy-specific coverage without loading models."""
    inventory, _ = extract_project(project, inventory=True)
    resolved = resolve_annotations(project, inventory)
    analyzer = ProjectAnalyzer(project, analyzer_config(project, semantic=False))
    result = analyzer.analyze(project.root)
    candidates = ProjectAnalyzer(project, analyzer_config(project))._select_semantic_candidates(
        result.units
    )
    candidate_uids = {u.uid for u in candidates}
    exact = find_exact_pair_keys(candidates)
    ids = unit_ids(project, inventory, resolved)
    judged = {pair_key(p["a"], p["b"]) for p in project.annotations["pairs"]}
    deterministic = {
        pair_key(ids[d.unit_a.uid], ids[d.unit_b.uid]) for d in result.traditional_duplicates
    }
    missing = sorted(deterministic - judged)
    if require_adjudicated and missing:
        raise ValueError(f"unjudged deterministic findings: {missing}")
    coverage = []
    for pair in project.annotations["pairs"]:
        a, b = resolved[pair["a"]], resolved[pair["b"]]
        reason = eligibility_reason(
            a,
            b,
            candidate_uids,
            exact,
            suppress_tests=project.policy.get("suppress_test_semantic_matches", False),
        )
        embedded = a.uid in candidate_uids and b.uid in candidate_uids
        expected = pair.get("expected_eligibility", {}).get(project.policy_name)
        if expected is not None and (
            expected["embedded"] != embedded or expected["comparable"] != (reason is None)
        ):
            raise ValueError(f"{pair['id']}: unexpected eligibility under {project.policy_name}")
        coverage.append(
            {
                "pair": pair["id"],
                "embedded": embedded,
                "comparable": reason is None,
                "exclusion_reason": reason,
                "traditional_recovery": pair_key(pair["a"], pair["b"]) in deterministic,
            }
        )
    for probe in project.annotations["probes"]:
        excluded = [i for i in probe["expected"] if resolved[i].uid not in candidate_uids]
        if excluded and not probe.get("policy_exclusion"):
            raise ValueError(f"probe {probe['id']} has unreachable targets: {excluded}")
    ineligible = [item for item in coverage if not item["embedded"] or not item["comparable"]]
    return {
        "project": project.id,
        "policy": project.policy_name,
        "units": len(result.units),
        "semantic_candidates": len(candidates),
        "judgments": len(coverage),
        "comparable_judgments": sum(item["comparable"] for item in coverage),
        "traditional_judgments": sum(item["traditional_recovery"] for item in coverage),
        "ineligible_judgments": ineligible,
        "pending_deterministic": missing,
        "probes": len(project.annotations["probes"]),
    }


def run_behavior(project: Project) -> dict[str, Any]:
    """Execute every declared fixture test and entry point."""
    runs = []
    for command in project.spec["behavior_tests"]:
        argv = [arg.replace("{python}", sys.executable) for arg in command["argv"]]
        env = os.environ.copy()
        env.update(command.get("env", {}))
        result = subprocess.run(
            argv,
            cwd=project.root,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        runs.append(
            {
                "id": command["id"],
                "returncode": result.returncode,
            }
        )
        if result.returncode:
            raise ValueError(
                f"behavior command failed: {command['id']}\n{result.stdout}\n{result.stderr}"
            )
    return {"project": project.id, "runs": runs}


def add_contract_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the common manifest/project/policy selectors."""
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--project", action="append", dest="projects")
    parser.add_argument("--policy", default="default")
