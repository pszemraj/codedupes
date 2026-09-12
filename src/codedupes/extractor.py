"""Language-aware extraction of functions, methods, and classes."""

from __future__ import annotations

import fnmatch
import logging
import os
import re
from collections.abc import Iterator
from pathlib import Path

from codedupes.constants import is_default_excluded_dir
from codedupes.languages.registry import (
    DECLARATION_FILE_SUFFIXES,
    get_backend,
    language_for_path,
    normalize_languages,
    repository_allows_c_headers,
)
from codedupes.models import CodeUnit, ExtractionDiagnostic

logger = logging.getLogger(__name__)

DEFAULT_EXCLUDE_PATTERNS = [
    "**/test_*",
    "**/*_test.*",
    "**/*_tests.*",
    "**/*.test.*",
    "**/*.spec.*",
    "**/tests/**",
    "**/__tests__/**",
]


class CodeExtractor:
    """Extract supported code units from a source tree or individual file."""

    def __init__(
        self,
        root: Path,
        exclude_patterns: list[str] | None = None,
        include_private: bool = True,
        include_stubs: bool = False,
        languages: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        """Construct an extractor for a project root.

        :param root: Root path to scan.
        :param exclude_patterns: Path/name globs; ``None`` uses test defaults for
            directory discovery, while an empty list disables those defaults.
            Directly named files bypass only the implicit test defaults.
        :param include_private: Include private names when true.
        :param include_stubs: Include ``.pyi`` files.
        :param languages: Optional canonical/alias language filter. Auto-detects
            supported source files when omitted.
        """
        self.root = root.resolve()
        self._uses_default_exclude_patterns = exclude_patterns is None
        self.exclude_patterns = (
            DEFAULT_EXCLUDE_PATTERNS.copy() if exclude_patterns is None else exclude_patterns
        )
        self._exclude_matchers: list[
            tuple[bool, bool, re.Pattern[str], re.Pattern[str] | None]
        ] = []
        for pattern in self.exclude_patterns:
            anchored = pattern.startswith(("./", "/"))
            directory_only = pattern.endswith("/")
            pattern = pattern.removeprefix("./").lstrip("/").rstrip("/")
            matcher = re.compile(fnmatch.translate(os.path.normcase(pattern)))
            zero_depth = (
                re.compile(fnmatch.translate(os.path.normcase(pattern[3:])))
                if pattern.startswith("**/")
                else None
            )
            self._exclude_matchers.append(
                (anchored or "/" in pattern, directory_only, matcher, zero_depth)
            )
        self.include_private = include_private
        self.include_stubs = include_stubs
        self.languages = normalize_languages(languages)
        self.diagnostics: list[ExtractionDiagnostic] = []
        self._c_headers_allowed: bool | None = None

    @staticmethod
    def _is_excluded_dir_name(name: str) -> bool:
        """Return ``True`` when a directory name should be skipped by default.

        :param name: Directory name.
        :return: Whether the directory is excluded.
        """
        return is_default_excluded_dir(name)

    def _should_exclude(
        self,
        path: Path,
        *,
        check_ancestors: bool = True,
        match_patterns: bool = True,
    ) -> bool:
        """Check exclusions for a path and its resolved in-tree symlink target.

        :param path: Candidate path.
        :param check_ancestors: Check parents unless the walk already pruned them.
        :param match_patterns: Apply configured path/name globs when true.
        :return: ``True`` when extraction should skip this file or directory.
        """
        if self._matches_exclude(
            path,
            check_ancestors=check_ancestors,
            match_patterns=match_patterns,
        ):
            return True
        if not path.is_symlink():
            return False
        try:
            resolved = path.resolve()
        except (OSError, RuntimeError):
            # Leave broken/looping links for the normal read-error diagnostic.
            return False
        return resolved.is_relative_to(self.root) and self._matches_exclude(
            resolved,
            match_patterns=match_patterns,
        )

    def _matches_exclude(
        self,
        path: Path,
        *,
        check_ancestors: bool = True,
        match_patterns: bool = True,
    ) -> bool:
        """Match a path's in-tree name against the configured exclusions.

        :param path: Candidate path under the extraction root.
        :param check_ancestors: Include parent directories in the match candidates.
        :param match_patterns: Apply configured path/name globs when true.
        :return: Whether the name or an ancestor matches an exclusion.
        """
        rel = path.relative_to(self.root)
        path_is_directory = path.is_dir()
        directory_parts = rel.parts if path_is_directory else rel.parts[:-1]
        if not check_ancestors:
            directory_parts = (rel.name,) if path_is_directory else ()
        if any(self._is_excluded_dir_name(part) for part in directory_parts):
            return True
        if not match_patterns:
            return False

        # Match ancestors too: excluding a directory excludes its whole subtree.
        candidates = [rel]
        if check_ancestors:
            candidates.extend(parent for parent in rel.parents if parent != Path("."))
        for candidate in candidates:
            is_directory = candidate != rel or path_is_directory
            relative_name = os.path.normcase(candidate.as_posix())
            basename = os.path.normcase(candidate.name)
            for use_path, directory_only, matcher, zero_depth in self._exclude_matchers:
                if directory_only and not is_directory:
                    continue
                value = relative_name if use_path else basename
                if matcher.match(value) or (is_directory and matcher.match(value + os.sep)):
                    return True
                # ``**/`` also matches zero directory levels.
                if zero_depth is not None and (
                    zero_depth.match(relative_name)
                    or (is_directory and zero_depth.match(relative_name + os.sep))
                ):
                    return True
        return False

    def _allow_c_headers(self) -> bool:
        """Resolve the repository-level C-header ambiguity policy once.

        :return: ``True`` when ambiguous ``.h`` files may be parsed as C.
        """
        if self._c_headers_allowed is None:
            self._c_headers_allowed = repository_allows_c_headers(
                self.root, self.languages, should_exclude=self._should_exclude
            )
        return self._c_headers_allowed

    def extract_from_file(self, file_path: Path) -> Iterator[CodeUnit]:
        """Yield all supported code units from a single file.

        Every supported language, Python included, is routed to its pinned
        Tree-sitter grammar package. Missing grammars are a hard configuration
        error; codedupes never silently falls back to line chunking.

        :param file_path: File to extract code units from.
        :return: Iterator over the code units found in the file.
        """
        # Normalize relative caller paths, but keep the in-tree name for files
        # that are symlinks to targets outside the root: exclusion and module
        # naming are computed relative to the root, and the symlink is the
        # file's identity within the analyzed tree.
        file_path = file_path.absolute()
        match_patterns = not self._uses_default_exclude_patterns
        if file_path.is_relative_to(self.root) and self._should_exclude(
            file_path,
            match_patterns=match_patterns,
        ):
            return
        try:
            resolved = file_path.resolve()
        except (OSError, RuntimeError):
            # pathlib translates ELOOP into RuntimeError on supported Python
            # versions; leave the original path for the read-error diagnostic.
            resolved = file_path
        if resolved.is_relative_to(self.root):
            file_path = resolved
        if self._should_exclude(file_path, match_patterns=match_patterns):
            logger.debug(f"Skipping excluded file {file_path}")
            return

        allow_c_header = file_path.suffix == ".h" and self._allow_c_headers()
        selection = language_for_path(
            file_path,
            include_stubs=self.include_stubs,
            selected_languages=self.languages,
            allow_c_header=allow_c_header,
        )
        if selection is None:
            self._diagnose_unsupported_file(file_path)
            return

        backend = get_backend(
            root=self.root,
            selection=selection,
            include_private=self.include_private,
        )
        result = backend.extract_file(file_path)
        self.diagnostics.extend(result.diagnostics)
        yield from result.units

    def _diagnose_unsupported_file(self, file_path: Path) -> None:
        """Record why an explicitly requested file resolves to no language.

        :param file_path: File that no extraction backend accepts.
        """
        suffix = file_path.suffix.lower()
        if file_path.name.lower().endswith(DECLARATION_FILE_SUFFIXES):
            language = "typescript"
            message = "TypeScript declaration files contain no implementation bodies."
            code = "declaration-file"
        elif file_path.suffix == ".h" and not self._allow_c_headers():
            language = "c"
            message = (
                "Skipped by the conservative C-header policy; pass --language c "
                "to parse .h files as C."
            )
            code = "c-header-policy"
        elif suffix == ".pyi" and not self.include_stubs:
            language = "python"
            message = "Skipped stub file; pass --include-stubs to analyze .pyi files."
            code = "stub-policy"
        else:
            unfiltered = language_for_path(
                file_path,
                include_stubs=True,
                selected_languages=None,
                allow_c_header=True,
            )
            if unfiltered is not None:
                language = unfiltered.language
                message = f"Excluded by the --language filter ({unfiltered.language})."
                code = "language-filter"
            else:
                language = "unknown"
                message = "Unsupported file type; no extraction backend accepts it."
                code = "unsupported-file"
        logger.warning(f"{file_path}: {message}")
        self.diagnostics.append(
            ExtractionDiagnostic(
                file_path=file_path,
                language=language,
                message=message,
                severity="warning",
                code=code,
            )
        )

    def _report_walk_error(self, error: OSError) -> None:
        """Record a directory traversal failure as incomplete extraction.

        :param error: Filesystem error raised while scanning a directory.
        :return: ``None``.
        """
        directory = Path(error.filename) if error.filename is not None else self.root
        message = f"Could not scan directory {directory}: {error}"
        logger.warning(message)
        self.diagnostics.append(
            ExtractionDiagnostic(
                file_path=directory,
                language="unknown",
                message=message,
                severity="warning",
                code="walk-error",
            )
        )

    def extract_all(self) -> list[CodeUnit]:
        """Extract all supported code units from the configured directory tree.

        :return: Every code unit extracted from the tree, in sorted walk order.
        """
        units: list[CodeUnit] = []
        seen: set[Path] = set()
        allow_c_header: bool | None = None
        skipped_headers: list[Path] = []
        skipped_test_files = 0
        skipped_test_dirs = 0
        default_test_patterns = set(DEFAULT_EXCLUDE_PATTERNS).intersection(self.exclude_patterns)

        def matches_default_tests(path: Path) -> bool:
            """Identify active default test globs on an already excluded path.

            :param path: File or directory skipped by the current walk.
            :return: Whether active default test globs match the path.
            """
            if self._should_exclude(path, match_patterns=False):
                return False
            relative = path.relative_to(self.root).as_posix()
            if path.is_dir():
                relative += "/"
            return any(
                fnmatch.fnmatch(relative, pattern)
                or fnmatch.fnmatch(relative, pattern.removeprefix("**/"))
                for pattern in default_test_patterns
            )

        for dirpath, dirnames, filenames in os.walk(
            self.root, followlinks=False, onerror=self._report_walk_error
        ):
            # Sorted in place so the walk descends deterministically: raw ``os.walk``
            # order is filesystem-dependent and would reorder the reported units.
            current_dir = Path(dirpath)
            included_dirs = []
            for name in sorted(dirnames):
                directory = current_dir / name
                if self._should_exclude(directory, check_ancestors=False):
                    skipped_test_dirs += matches_default_tests(directory)
                else:
                    included_dirs.append(name)
            dirnames[:] = included_dirs

            for filename in sorted(filenames):
                source_file = current_dir / filename
                is_c_header = source_file.suffix == ".h"
                if is_c_header and allow_c_header is None:
                    allow_c_header = self._allow_c_headers()
                header_allowed = bool(allow_c_header) if is_c_header else False
                selection = language_for_path(
                    source_file,
                    include_stubs=self.include_stubs,
                    selected_languages=self.languages,
                    allow_c_header=header_allowed,
                )
                report_skipped_header = (
                    is_c_header and not header_allowed and self.languages is None
                )
                if selection is None and not report_skipped_header:
                    continue

                # Ancestors were pruned above; symlink targets still need a full check.
                if self._should_exclude(source_file, check_ancestors=False):
                    skipped_test_files += matches_default_tests(source_file)
                    continue
                if selection is None:
                    skipped_headers.append(source_file)
                    continue

                try:
                    resolved = source_file.resolve()
                except (OSError, RuntimeError):
                    resolved = source_file
                if resolved in seen:
                    continue
                seen.add(resolved)

                units.extend(self.extract_from_file(source_file))

        if skipped_test_files or skipped_test_dirs:
            logger.info(
                f"Skipped {skipped_test_files} files and {skipped_test_dirs} directories "
                "matching default test exclusions; use --no-default-excludes to include them."
            )

        if skipped_headers:
            message = (
                f"{len(skipped_headers)} .h file(s) skipped by the conservative "
                "C-header policy; pass --language c to parse them as C."
            )
            logger.warning(message)
            self.diagnostics.append(
                ExtractionDiagnostic(
                    file_path=skipped_headers[0],
                    language="c",
                    message=message,
                    severity="warning",
                    code="c-header-policy",
                )
            )

        return units
