"""Module-prefix and qualified-name construction shared by every language backend."""

from __future__ import annotations

from pathlib import Path

# Multi-part ECMAScript suffixes that ``Path.stem`` would only half-strip.
_ECMASCRIPT_SUFFIXES = (".d.ts", ".d.mts", ".d.cts", ".tsx", ".mts", ".cts", ".jsx", ".mjs", ".cjs")


def module_prefix(root: Path, file_path: Path, language: str) -> str:
    """Build the dotted module prefix that qualifies every unit in one file.

    Python follows import semantics: ``pkg/__init__.py`` names ``pkg`` and a
    root-level ``__init__.py`` names nothing. The other languages collapse their
    conventional entry-point stems (``index``, plus Rust's ``mod``/``lib``/``main``)
    into the enclosing directory.

    :param root: Extraction root the file path is made relative to.
    :param file_path: File being extracted.
    :param language: Canonical language name.
    :return: Dotted prefix, possibly empty for a root-level Python package initializer.
    """
    try:
        rel = file_path.relative_to(root)
    except ValueError:
        rel = Path(file_path.name)

    parts = list(rel.parts[:-1])
    stem = rel.name
    for suffix in _ECMASCRIPT_SUFFIXES:
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    else:
        stem = Path(stem).stem

    if language == "python":
        if stem != "__init__":
            parts.append(stem)
        return ".".join(part for part in parts if part)

    conventional = {"index"}
    if language == "rust":
        conventional |= {"mod", "lib", "main"}
    if stem not in conventional or not parts:
        parts.append(stem)
    if not parts:
        parts.append(stem or file_path.stem)
    return ".".join(part for part in parts if part)


def qualified(prefix: str, *parts: str) -> str:
    """Join a module prefix and name segments into one dotted name.

    :param prefix: Module prefix, possibly empty.
    :param parts: Name segments in outermost-first order.
    :return: Dotted qualified name with empty segments dropped.
    """
    clean = [part for part in (prefix, *parts) if part]
    return ".".join(clean)
