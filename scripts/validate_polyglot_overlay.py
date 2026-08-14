"""Validate the codedupes polyglot overlay and its pinned parser contract."""

from __future__ import annotations

import argparse
import sys
import tempfile
import tomllib
from pathlib import Path

REQUIRED_DEPENDENCIES = {
    "tree-sitter": "0.25.2",
    "tree-sitter-c": "0.24.2",
    "tree-sitter-rust": "0.24.2",
    "tree-sitter-javascript": "0.25.0",
    "tree-sitter-typescript": "0.23.2",
}

FIXTURES = {
    "sample.c": (
        "c",
        "int add(int left, int right) { return left + right; }\n",
        {"sample.add"},
    ),
    "sample.rs": (
        "rust",
        "pub fn add(left: i32, right: i32) -> i32 { left + right }\n",
        {"sample.add"},
    ),
    "sample.js": (
        "javascript",
        "export const add = (left, right) => left + right;\n",
        {"sample.add"},
    ),
    "sample.ts": (
        "typescript",
        "export function add(left: number, right: number): number { return left + right; }\n",
        {"sample.add"},
    ),
    "component.tsx": (
        "typescript",
        "export const Card = (props: { title: string }) => <h1>{props.title}</h1>;\n",
        {"component.Card"},
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root containing pyproject.toml and src/.",
    )
    parser.add_argument(
        "--require-grammars",
        action="store_true",
        help="Fail instead of skipping extraction when parser wheels are unavailable.",
    )
    return parser.parse_args()


def _assert_dependency_pins(source_root: Path) -> None:
    with (source_root / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)["project"]
    dependencies = project["dependencies"]
    for package, version in REQUIRED_DEPENDENCIES.items():
        required = f"{package}=={version}"
        if required not in dependencies:
            raise AssertionError(f"Missing exact dependency pin: {required}")


def _grammar_errors() -> list[str]:
    from codedupes.languages.registry import get_grammar_statuses

    return [status.error for status in get_grammar_statuses() if status.error]


def _run_extraction_fixtures() -> None:
    from codedupes.extractor import CodeExtractor

    with tempfile.TemporaryDirectory(prefix="codedupes-polyglot-") as temp_dir:
        root = Path(temp_dir)
        for filename, (language, source, expected_names) in FIXTURES.items():
            file_path = root / filename
            file_path.write_text(source, encoding="utf-8")
            extractor = CodeExtractor(root, languages=(language,), include_private=True)
            units = list(extractor.extract_from_file(file_path))
            actual_names = {unit.qualified_name for unit in units}
            if not expected_names <= actual_names:
                raise AssertionError(
                    f"{filename}: expected {sorted(expected_names)}, got {sorted(actual_names)}"
                )
            for unit in units:
                source_bytes = file_path.read_bytes()
                if source_bytes[unit.start_byte : unit.end_byte].decode("utf-8") != unit.source:
                    raise AssertionError(f"{filename}: byte range does not reproduce unit source")


def main() -> int:
    args = _parse_args()
    source_root = args.source_root.resolve()
    sys.path.insert(0, str(source_root / "src"))

    _assert_dependency_pins(source_root)
    errors = _grammar_errors()
    if errors:
        print("PASS: exact Tree-sitter dependency pins are present")
        for error in dict.fromkeys(errors):
            print(f"UNAVAILABLE: {error}")
        if args.require_grammars:
            print("FAIL: --require-grammars was requested", file=sys.stderr)
            return 1
        print("SKIP: parser-backed extraction fixtures (install the pinned wheels to run them)")
        return 0

    _run_extraction_fixtures()
    print("PASS: exact Tree-sitter dependency pins are present")
    print("PASS: C, Rust, JavaScript, TypeScript, and TSX extraction fixtures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
