# codedupes

`codedupes` detects duplicate code across Python, C, Rust, JavaScript, JSX, TypeScript, and TSX. It combines deterministic structural/token matching with semantic code embeddings, while retaining conservative Python-only unused-code analysis.

Core capabilities:

- Language-aware extraction of functions, methods, and classes
- Exact structural and token fingerprints, plus identifier-Jaccard near matching
- Semantic matching and search with model-profile embeddings (default `gte-modernbert-base`)
- Persistent, content-addressed embedding cache so unchanged code is not re-embedded
- Explicit CPU, CUDA, and Apple Silicon MPS execution
- Recoverable parse diagnostics instead of silent line-chunk fallback

## Install

```bash
pip install "codedupes @ git+https://github.com/pszemraj/codedupes.git"
```

The normal installation includes exact-pinned, precompiled Tree-sitter packages for C, Rust, JavaScript/JSX, TypeScript, and TSX. codedupes does not download grammars while analyzing a repository.

See [Installation](https://github.com/pszemraj/codedupes/blob/main/docs/install.md) for Python and runtime requirements.

## Quick start

```bash
codedupes check ./src
codedupes search ./src "normalize request payload"
codedupes info

# Restrict a mixed tree to selected languages
codedupes check ./src --language rust --language typescript

# Apple Silicon
codedupes check ./src --device mps
```

Language aliases such as `py`, `rs`, `js`, `jsx`, `ts`, and `tsx` are accepted. Omit `--language` to auto-detect every supported source type.

See [Output and exit codes](https://github.com/pszemraj/codedupes/blob/main/docs/output.md) for report modes and CI behavior.

## Documentation

- [Installation](https://github.com/pszemraj/codedupes/blob/main/docs/install.md)
- [Polyglot language support](https://github.com/pszemraj/codedupes/blob/main/docs/polyglot-languages.md)
- [CLI reference](https://github.com/pszemraj/codedupes/blob/main/docs/cli.md)
- [Usage guide](https://github.com/pszemraj/codedupes/blob/main/docs/usage.md)
- [Python API](https://github.com/pszemraj/codedupes/blob/main/docs/python-api.md)
- [Output and exit codes](https://github.com/pszemraj/codedupes/blob/main/docs/output.md)
- [Analysis defaults and heuristics](https://github.com/pszemraj/codedupes/blob/main/docs/analysis-defaults.md)
- [Semantic model profiles and tasks](https://github.com/pszemraj/codedupes/blob/main/docs/model-profiles.md)
- [Embedding cache](https://github.com/pszemraj/codedupes/blob/main/docs/caching.md)
- [Accelerators and Apple Silicon](https://github.com/pszemraj/codedupes/blob/main/docs/accelerators.md)
- [Hybrid gate tuning](https://github.com/pszemraj/codedupes/blob/main/docs/hybrid-tuning.md)

## Scope and limits

- Duplicate checking is same-language by default. Semantic search can retrieve implementations across languages.
- Unused-code analysis is Python-only. Non-Python units are counted and explicitly excluded rather than evaluated with Python heuristics.
- C preprocessing, Rust macro expansion, and JavaScript/TypeScript project-wide name resolution are outside the syntax-only extraction layer.
- `.h` files are treated as C only when C is explicitly selected or the scanned tree contains C sources and no detected C++ sources.
- TypeScript declaration files (`.d.ts`, `.d.mts`, and `.d.cts`) are skipped because they contain declarations rather than executable implementations.
