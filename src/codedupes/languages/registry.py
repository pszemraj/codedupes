"""Language detection, grammar metadata, and backend construction."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from importlib import metadata
from pathlib import Path
from typing import Final

from codedupes.constants import is_default_excluded_dir
from codedupes.languages.base import LanguageBackend

SUPPORTED_LANGUAGES: Final[tuple[str, ...]] = (
    "python",
    "c",
    "rust",
    "javascript",
    "typescript",
)

_LANGUAGE_ALIASES: Final[dict[str, str]] = {
    "py": "python",
    "python": "python",
    "c": "c",
    "rs": "rust",
    "rust": "rust",
    "js": "javascript",
    "jsx": "javascript",
    "javascript": "javascript",
    "ts": "typescript",
    "tsx": "typescript",
    "typescript": "typescript",
}

DECLARATION_FILE_SUFFIXES: Final[tuple[str, ...]] = (".d.ts", ".d.mts", ".d.cts")
CPP_SUFFIXES: Final[frozenset[str]] = frozenset(
    {".C", ".H", ".cc", ".cpp", ".cxx", ".c++", ".hh", ".hpp", ".hxx", ".h++"}
)
TREE_SITTER_PACKAGE: Final[tuple[str, str]] = ("tree-sitter", "0.25.2")


@dataclass(frozen=True)
class LanguageSelection:
    """Canonical language and parser dialect chosen for one file."""

    language: str
    dialect: str


@dataclass(frozen=True)
class GrammarStatus:
    """Installed-package status for one parser dialect."""

    language: str
    dialect: str
    package: str
    pinned_version: str
    installed_version: str | None
    available: bool
    error: str | None = None


GRAMMAR_PACKAGES: Final[dict[str, tuple[str, str]]] = {
    "c": ("tree-sitter-c", "0.24.2"),
    "rust": ("tree-sitter-rust", "0.24.2"),
    "javascript": ("tree-sitter-javascript", "0.25.0"),
    "typescript": ("tree-sitter-typescript", "0.23.2"),
    "tsx": ("tree-sitter-typescript", "0.23.2"),
}

# Single source of truth for every exact parser pin. pyproject.toml must match
# this mapping; tests enforce the invariant so a pin bump is always a reviewed,
# two-location change instead of a transitive surprise.
REQUIRED_PARSER_PACKAGES: Final[dict[str, str]] = {
    TREE_SITTER_PACKAGE[0]: TREE_SITTER_PACKAGE[1],
    **{package: version for package, version in GRAMMAR_PACKAGES.values()},
}


_EXTENSION_SELECTIONS: Final[dict[str, LanguageSelection]] = {
    ".py": LanguageSelection("python", "python"),
    ".pyi": LanguageSelection("python", "python"),
    ".c": LanguageSelection("c", "c"),
    ".rs": LanguageSelection("rust", "rust"),
    ".js": LanguageSelection("javascript", "javascript"),
    ".jsx": LanguageSelection("javascript", "jsx"),
    ".mjs": LanguageSelection("javascript", "javascript"),
    ".cjs": LanguageSelection("javascript", "javascript"),
    ".ts": LanguageSelection("typescript", "typescript"),
    ".mts": LanguageSelection("typescript", "typescript"),
    ".cts": LanguageSelection("typescript", "typescript"),
    ".tsx": LanguageSelection("typescript", "tsx"),
}


def _is_cpp_suffix(suffix: str) -> bool:
    """Return whether a raw suffix conventionally identifies C++ source.

    ``.C`` and ``.H`` are case-sensitive C++ spellings, while the remaining
    supported C++ signals are matched case-insensitively.

    :param suffix: Raw suffix from :attr:`pathlib.Path.suffix`.
    :return: ``True`` when the suffix should prevent C parsing.
    """
    return suffix in CPP_SUFFIXES or suffix.lower() in CPP_SUFFIXES


def normalize_languages(languages: tuple[str, ...] | list[str] | None) -> tuple[str, ...] | None:
    """Canonicalize a requested language filter while preserving order.

    :param languages: Requested language names or aliases, if any.
    :raises ValueError: If any requested value is not a supported language.
    :return: Canonical language names, or ``None`` when no filter was requested.
    """
    if languages is None or len(languages) == 0:
        return None

    normalized: list[str] = []
    invalid: list[str] = []
    for language in languages:
        key = language.strip().lower()
        canonical = _LANGUAGE_ALIASES.get(key)
        if canonical is None:
            invalid.append(language)
            continue
        if canonical not in normalized:
            normalized.append(canonical)

    if invalid:
        allowed = ", ".join(SUPPORTED_LANGUAGES)
        raise ValueError(
            f"Unsupported language selection: {', '.join(invalid)}. Allowed values: {allowed}"
        )
    return tuple(normalized)


def language_for_path(
    path: Path,
    *,
    include_stubs: bool,
    selected_languages: tuple[str, ...] | None,
    allow_c_header: bool,
) -> LanguageSelection | None:
    """Resolve a supported language/dialect for ``path``.

    TypeScript declaration files are intentionally excluded because they do not
    contain executable implementation bodies.  ``.h`` is accepted only when the
    caller has resolved its C/C++ ambiguity.

    :param path: File whose language is being resolved.
    :param include_stubs: Whether ``.pyi`` stub files are eligible.
    :param selected_languages: Canonical language filter, or ``None`` for no filter.
    :param allow_c_header: Whether ambiguous ``.h`` files may be treated as C.
    :return: Language and dialect for the file, or ``None`` when unsupported.
    """
    name = path.name.lower()
    if name.endswith(DECLARATION_FILE_SUFFIXES):
        return None

    suffix = path.suffix
    if _is_cpp_suffix(suffix):
        return None

    normalized_suffix = suffix.lower()
    if normalized_suffix == ".h":
        selection = LanguageSelection("c", "c") if allow_c_header else None
    else:
        selection = _EXTENSION_SELECTIONS.get(normalized_suffix)

    if selection is None:
        return None
    if selection.language == "python" and normalized_suffix == ".pyi" and not include_stubs:
        return None
    if selected_languages is not None and selection.language not in selected_languages:
        return None
    return selection


def repository_allows_c_headers(
    root: Path,
    selected_languages: tuple[str, ...] | None,
    *,
    should_exclude: Callable[[Path], bool] | None = None,
) -> bool:
    """Return whether ambiguous ``.h`` files can safely be treated as C.

    Explicit ``--language c`` selection wins.  Automatic detection accepts headers only
    when the scanned tree contains C source and no C++ source.  The scan mirrors the
    extraction walk, pruning artifact directories and applying the extractor's
    exclusion predicate when supplied.

    :param root: Scan root for the analysis.
    :param selected_languages: Canonical language filter, or ``None`` for auto-detection.
    :param should_exclude: Optional extractor predicate for ignored files/directories.
    :return: ``True`` when ``.h`` files may be parsed as C.
    """
    if selected_languages is not None:
        return "c" in selected_languages

    scan_root = root if root.is_dir() else root.parent
    saw_c_source = False
    # One predicate, shared with the extraction walk. A divergent second list let a
    # vendored C++ tree flip `.h` handling for files extraction never visits, and let
    # `*.egg-info` C++ flip it for files extraction always skips.
    for directory, dirnames, filenames in os.walk(scan_root):
        current = Path(directory)
        dirnames[:] = [
            name
            for name in dirnames
            if not is_default_excluded_dir(name)
            and (should_exclude is None or not should_exclude(current / name))
        ]
        for filename in filenames:
            if should_exclude is not None and should_exclude(current / filename):
                continue
            suffix = Path(filename).suffix
            if _is_cpp_suffix(suffix):
                return False
            if suffix.lower() == ".c":
                saw_c_source = True
    return saw_c_source


def get_backend(
    *,
    root: Path,
    selection: LanguageSelection,
    include_private: bool,
) -> LanguageBackend:
    """Construct a Tree-sitter backend without importing parser dependencies eagerly.

    :param root: Extraction root used for qualified naming.
    :param selection: Language and dialect chosen for the file.
    :param include_private: Whether non-public units are extracted.
    :raises ValueError: If ``selection`` names Python, which has a dedicated extractor.
    :return: Backend for the selected language and dialect.
    """
    if selection.language == "python":
        raise ValueError("Python extraction is implemented by codedupes.extractor.CodeExtractor")

    from codedupes.languages.tree_sitter_backend import create_backend

    return create_backend(
        root=root,
        language=selection.language,
        dialect=selection.dialect,
        include_private=include_private,
    )


@cache
def _probe_dialect(dialect: str) -> str | None:
    """Build a parser and run an empty parse; return an error message or ``None``.

    Version metadata alone cannot prove a native wheel is loadable: a wrong
    platform wheel, missing shared library, bad capsule, or Tree-sitter ABI
    mismatch all pass the version check and still fail at parser construction.

    :param dialect: Parser dialect to probe.
    :return: Error message describing the failure, or ``None`` when the parser loads.
    """
    from codedupes.languages.tree_sitter_backend import (
        GrammarProvider,
        GrammarUnavailableError,
    )

    try:
        GrammarProvider.parser(dialect).parse(b"")
    except GrammarUnavailableError as exc:
        return str(exc)
    except Exception as exc:  # noqa: BLE001 - native loader faults must become status text
        return f"parser construction failed: {type(exc).__name__}: {exc}"
    return None


def get_grammar_statuses() -> tuple[GrammarStatus, ...]:
    """Report whether every required parser package is installed and loadable.

    Exact version pins are checked first; when they match, each dialect's
    parser is actually constructed (memoized per process) so a broken wheel is
    reported instead of surfacing later mid-analysis.

    :return: Status for the core package and every pinned grammar dialect.
    """
    core_package, core_pinned = TREE_SITTER_PACKAGE
    try:
        tree_sitter_version = metadata.version(core_package)
        tree_sitter_error = (
            None
            if tree_sitter_version == core_pinned
            else f"{core_package}=={core_pinned} is required; found {tree_sitter_version}"
        )
    except metadata.PackageNotFoundError:
        tree_sitter_version = None
        tree_sitter_error = f"{core_package}=={core_pinned} is not installed"

    statuses: list[GrammarStatus] = []
    for dialect, (package, pinned) in GRAMMAR_PACKAGES.items():
        try:
            installed = metadata.version(package)
            package_error = (
                None
                if installed == pinned
                else f"{package}=={pinned} is required; found {installed}"
            )
        except metadata.PackageNotFoundError:
            installed = None
            package_error = f"{package}=={pinned} is not installed"

        language = "typescript" if dialect == "tsx" else dialect
        error_parts = [part for part in (tree_sitter_error, package_error) if part]
        if not error_parts:
            probe_error = _probe_dialect(dialect)
            if probe_error:
                error_parts.append(probe_error)
        statuses.append(
            GrammarStatus(
                language=language,
                dialect=dialect,
                package=package,
                pinned_version=pinned,
                installed_version=installed,
                available=not error_parts,
                error="; ".join(error_parts) or None,
            )
        )
    return tuple(statuses)
