"""Language-aware extraction of functions, methods, and classes."""

from __future__ import annotations

import fnmatch
import logging
import os
import re
import subprocess
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

# (use path or basename, anchored, directory-only, matcher, zero-depth matcher)
_ExcludeMatcher = tuple[bool, bool, bool, re.Pattern[str], re.Pattern[str] | None]

DEFAULT_EXCLUDE_PATTERNS = [
    "**/test_*",
    "**/*_test.*",
    "**/*_tests.*",
    "**/*.test.*",
    "**/*.spec.*",
    "**/tests/**",
    "**/__tests__/**",
]


def git_ignored_paths(root: Path) -> frozenset[Path]:
    """Return the root-relative paths git ignores beneath a scan root.

    Git is the authority on its own ignore rules (nested ``.gitignore`` files,
    ``.git/info/exclude``, the global excludes file, negations, and tracked
    files that match a pattern but are not ignored), so the answer comes from
    ``git ls-files`` rather than a reimplementation. A wholly ignored directory
    is one entry, which lets the walk prune it without listing its contents.

    :param root: Resolved scan root.
    :return: Ignored paths relative to ``root``; empty when ``root`` is not
        inside a git work tree, ``git`` is unavailable, or ``root`` itself is
        ignored (an explicitly selected ignored directory is scanned in full).
    """
    command = [
        "git",
        "-C",
        os.fspath(root),
        "ls-files",
        "--others",
        "--ignored",
        "--exclude-standard",
        "--directory",
        "-z",
    ]
    try:
        completed = subprocess.run(
            command, stdin=subprocess.DEVNULL, capture_output=True, check=False
        )
    except OSError as error:
        logger.debug(f"git is unavailable; .gitignore is not applied to {root}: {error}")
        return frozenset()
    if completed.returncode != 0:
        detail = completed.stderr.decode(errors="replace").strip().splitlines()
        logger.debug(
            f".gitignore is not applied to {root}: {detail[0] if detail else 'git ls-files failed'}"
        )
        return frozenset()
    entries = [os.fsdecode(entry) for entry in completed.stdout.split(b"\0") if entry]
    # Git can emit "./" for the selected root alongside ignored child files;
    # the marker itself is not a path to prune.
    return frozenset(
        Path(entry.rstrip("/")) for entry in entries if entry.rstrip("/") not in {"", "."}
    )


def git_work_tree(path: Path) -> Path | None:
    """Return the git work tree containing a directory, if any.

    :param path: Directory to query, passed to ``git`` as ``-C``.
    :return: Resolved work-tree root, or ``None`` outside a git work tree or
        when ``git`` is unavailable.
    """
    try:
        completed = subprocess.run(
            ["git", "-C", os.fspath(path), "rev-parse", "--show-toplevel"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    toplevel = completed.stdout.decode(errors="replace").strip()
    if not toplevel:
        return None
    # git prints the realpath, which can differ from Path.resolve() on macOS
    # (/private/var vs /var); resolve again so both compare equal.
    return Path(toplevel).resolve()


class CodeExtractor:
    """Extract supported code units from a source tree or individual file."""

    def __init__(
        self,
        root: Path,
        exclude_patterns: list[str] | None = None,
        include_private: bool = True,
        include_stubs: bool = False,
        languages: tuple[str, ...] | list[str] | None = None,
        respect_gitignore: bool = True,
        implicit_default_excludes: bool = False,
        pattern_root: Path | None = None,
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
        :param respect_gitignore: Skip paths git ignores when the root is inside a
            git work tree. Directly named files are analyzed regardless.
        :param implicit_default_excludes: Whether an explicit pattern list starts
            with CLI-added defaults rather than caller-supplied exclusions.
        :param pattern_root: Root for anchored exclusions when walking a larger
            tree for a directly selected file's references. Unanchored patterns
            still apply throughout the tree.
        """
        self.root = root.resolve()
        self.pattern_root = pattern_root.resolve() if pattern_root is not None else self.root
        self.respect_gitignore = respect_gitignore
        self._ignored_paths: frozenset[Path] | None = None
        self._uses_default_exclude_patterns = exclude_patterns is None
        self.exclude_patterns = (
            DEFAULT_EXCLUDE_PATTERNS.copy() if exclude_patterns is None else exclude_patterns
        )
        self._exclude_matchers: list[_ExcludeMatcher] = []
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
                (anchored or "/" in pattern, anchored, directory_only, matcher, zero_depth)
            )
        # Pattern values cannot reveal whether the caller supplied a default
        # shape explicitly. The CLI marks its own prefix so only that prefix
        # is ignored by the reference-only walk.
        if implicit_default_excludes and (
            self.exclude_patterns[: len(DEFAULT_EXCLUDE_PATTERNS)] != DEFAULT_EXCLUDE_PATTERNS
        ):
            raise ValueError("implicit_default_excludes requires the default pattern prefix")
        default_count = (
            len(DEFAULT_EXCLUDE_PATTERNS)
            if exclude_patterns is None or implicit_default_excludes
            else 0
        )
        self._default_exclude_patterns = self.exclude_patterns[:default_count]
        self._user_exclude_matchers = self._exclude_matchers[default_count:]
        self.include_private = include_private
        self.include_stubs = include_stubs
        self.languages = normalize_languages(languages)
        self.diagnostics: list[ExtractionDiagnostic] = []
        # Every file handed to a backend, by canonical language, whether or not
        # it yielded units: the unused analysis parses each Python file for
        # references, and a re-export module or script has none to yield.
        self.extracted_files: dict[str, list[Path]] = {}
        # Python files skipped by a default test-file shape alone (not a user
        # exclusion, git ignore rule, or artifact directory): still parsed for
        # unused-code references even though they are not extracted as units.
        self.reference_only_files: list[Path] = []
        self._c_headers_allowed: bool | None = None

    @staticmethod
    def _is_excluded_dir_name(name: str) -> bool:
        """Return ``True`` when a directory name should be skipped by default.

        :param name: Directory name.
        :return: Whether the directory is excluded.
        """
        return is_default_excluded_dir(name)

    def _is_gitignored(self, path: Path, *, check_ancestors: bool = True) -> bool:
        """Return whether git ignores an in-tree path or one of its ancestors.

        The ignored set is read once per extractor, on first use, so a
        single-file target never runs git.

        :param path: Candidate path under the extraction root.
        :param check_ancestors: Include parent directories in the lookup.
        :return: Whether the path is skipped by the ignore rules.
        """
        if not self.respect_gitignore:
            return False
        if self._ignored_paths is None:
            self._ignored_paths = git_ignored_paths(self.root)
        if not self._ignored_paths:
            return False
        rel = path.relative_to(self.root)
        if rel in self._ignored_paths:
            return True
        return check_ancestors and any(parent in self._ignored_paths for parent in rel.parents)

    def _should_exclude(
        self,
        path: Path,
        *,
        check_ancestors: bool = True,
        match_patterns: bool = True,
        match_ignored: bool = True,
        matchers: list[_ExcludeMatcher] | None = None,
    ) -> bool:
        """Check exclusions for a path and its resolved in-tree symlink target.

        :param path: Candidate path.
        :param check_ancestors: Check parents unless the walk already pruned them.
        :param match_patterns: Apply configured path/name globs when true.
        :param match_ignored: Apply git ignore rules when true.
        :param matchers: Pattern matchers to use instead of the full configured set;
            artifact directories and git ignore rules still apply either way.
        :return: ``True`` when extraction should skip this file or directory.
        """
        if self._matches_exclude(
            path,
            check_ancestors=check_ancestors,
            match_patterns=match_patterns,
            match_ignored=match_ignored,
            matchers=matchers,
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
            match_ignored=match_ignored,
            matchers=matchers,
        )

    def _matches_exclude(
        self,
        path: Path,
        *,
        check_ancestors: bool = True,
        match_patterns: bool = True,
        match_ignored: bool = True,
        matchers: list[_ExcludeMatcher] | None = None,
    ) -> bool:
        """Match a path's in-tree name against the configured exclusions.

        :param path: Candidate path under the extraction root.
        :param check_ancestors: Include parent directories in the match candidates.
        :param match_patterns: Apply configured path/name globs when true.
        :param match_ignored: Apply git ignore rules when true.
        :param matchers: Pattern matchers to use instead of the full configured set;
            artifact directories and git ignore rules still apply either way.
        :return: Whether the name or an ancestor matches an exclusion.
        """
        rel = path.relative_to(self.root)
        path_is_directory = path.is_dir()
        directory_parts = rel.parts if path_is_directory else rel.parts[:-1]
        if not check_ancestors:
            directory_parts = (rel.name,) if path_is_directory else ()
        if any(self._is_excluded_dir_name(part) for part in directory_parts):
            return True
        if match_ignored and self._is_gitignored(path, check_ancestors=check_ancestors):
            return True
        if not match_patterns:
            return False

        # Anchored patterns keep a single-file target's parent as their scope
        # during the wider reference walk. Unanchored patterns still apply to
        # project files outside that parent.
        pattern_rel = (
            path.relative_to(self.pattern_root) if path.is_relative_to(self.pattern_root) else None
        )
        match_rel = pattern_rel if pattern_rel is not None else rel
        # Match ancestors too: excluding a directory excludes its whole subtree.
        candidates = [match_rel]
        if check_ancestors:
            candidates.extend(parent for parent in match_rel.parents if parent != Path("."))
        active_matchers = self._exclude_matchers if matchers is None else matchers
        for candidate in candidates:
            is_directory = candidate != match_rel or path_is_directory
            relative_name = os.path.normcase(candidate.as_posix())
            basename = os.path.normcase(candidate.name)
            for use_path, anchored, directory_only, matcher, zero_depth in active_matchers:
                if anchored and pattern_rel is None:
                    continue
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

    def _default_only_exclusion(self, path: Path) -> bool:
        """Return whether a path is excluded only by a default test-file shape.

        :param path: Candidate path, already known to be excluded (ancestors pruned).
        :return: ``True`` when the full exclusion rules skip the path but the
            user's own patterns, git ignore rules, and artifact directories do not.
        """
        return self._should_exclude(path, check_ancestors=False) and not self._should_exclude(
            path, check_ancestors=False, matchers=self._user_exclude_matchers
        )

    def _collect_reference_files(self, directory: Path) -> list[Path]:
        """Walk a default-excluded directory for Python files to parse for references.

        The caller has already established that ``directory`` is skipped only by a
        default test-file shape, so this walk prunes with the user's own patterns
        alone; git ignore rules and artifact directories still apply through
        :meth:`_should_exclude`.

        :param directory: Directory the main walk pruned for matching a default test shape.
        :return: Python file paths under ``directory`` not otherwise excluded.
        """
        # os.walk does not descend into symlink directories it encounters, but
        # it does walk one supplied as the starting path.
        if directory.is_symlink():
            return []
        collected: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(
            directory, followlinks=False, onerror=self._report_walk_error
        ):
            current_dir = Path(dirpath)
            included_dirs = []
            for name in sorted(dirnames):
                subdirectory = current_dir / name
                if not self._should_exclude(
                    subdirectory, check_ancestors=False, matchers=self._user_exclude_matchers
                ):
                    included_dirs.append(name)
            dirnames[:] = included_dirs

            for filename in sorted(filenames):
                if not filename.endswith(".py"):
                    continue
                source_file = current_dir / filename
                if self._should_exclude(
                    source_file, check_ancestors=False, matchers=self._user_exclude_matchers
                ):
                    continue
                collected.append(source_file)
        return collected

    def reference_files(self) -> list[Path]:
        """Return Python files under the root for unused-code reference parsing only.

        Walks the whole tree with the user's own exclusion patterns (git ignore
        rules and artifact directories still apply), so test files the default
        shapes would otherwise drop from duplicate detection are included. Used
        by a single-file scan target to seed the reference graph project-wide.

        :return: Every Python file under the root the user's own patterns admit.
        """
        return self._collect_reference_files(self.root)

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
        # A named file is a deliberate request: implicit test globs and git
        # ignore rules gate directory discovery, and the walk applied both.
        match_patterns = not self._uses_default_exclude_patterns
        if file_path.is_relative_to(self.root) and self._should_exclude(
            file_path,
            match_patterns=match_patterns,
            match_ignored=False,
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
        if self._should_exclude(file_path, match_patterns=match_patterns, match_ignored=False):
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
        self.extracted_files.setdefault(selection.language, []).append(file_path)
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

    def extract_all(self, *, collect_reference_files: bool = True) -> list[CodeUnit]:
        """Extract all supported code units from the configured directory tree.

        When requested, populates :attr:`reference_only_files` with Python files
        skipped only by a default test-file shape, for unused-code reference parsing.

        :param collect_reference_files: Discover default-excluded Python files
            for unused-code references; disable when unused analysis will not run.
        :return: Every code unit extracted from the tree, in sorted walk order.
        """
        units: list[CodeUnit] = []
        self.reference_only_files = []
        seen: set[Path] = set()
        allow_c_header: bool | None = None
        skipped_headers: list[Path] = []
        skipped_test_files = 0
        skipped_test_dirs = 0
        skipped_ignored_files = 0
        skipped_ignored_dirs = 0
        default_test_patterns = set(self._default_exclude_patterns)

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

        def skipped_by_gitignore(path: Path) -> bool:
            """Identify paths the walk skipped for git ignore rules alone.

            :param path: File or directory skipped by the current walk.
            :return: Whether git ignore rules, not built-in directory names, skipped it.
            """
            return self._should_exclude(path, match_patterns=False) and not self._should_exclude(
                path, match_patterns=False, match_ignored=False
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
                    skipped_ignored_dirs += skipped_by_gitignore(directory)
                    if collect_reference_files and self._default_only_exclusion(directory):
                        self.reference_only_files.extend(self._collect_reference_files(directory))
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
                    skipped_ignored_files += skipped_by_gitignore(source_file)
                    if (
                        collect_reference_files
                        and source_file.suffix == ".py"
                        and self._default_only_exclusion(source_file)
                    ):
                        self.reference_only_files.append(source_file)
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
            message = (
                "their Python files still count as references for unused-code analysis. "
                if collect_reference_files
                else ""
            )
            hint = (
                "Use --no-default-excludes to include them in duplicate detection too."
                if collect_reference_files
                else "Use --no-default-excludes to include them in duplicate detection."
            )
            logger.info(
                f"Skipped {skipped_test_files} files and {skipped_test_dirs} directories "
                f"matching default test exclusions; {message}{hint}"
            )
        if skipped_ignored_files or skipped_ignored_dirs:
            logger.info(
                f"Skipped {skipped_ignored_files} files and {skipped_ignored_dirs} directories "
                "ignored by git; use --no-gitignore to include them."
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
