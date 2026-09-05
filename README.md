# codedupes

`codedupes` detects duplicate code across Python, C, Rust, JavaScript, JSX, TypeScript, and TSX. It combines deterministic structural/token matching with semantic code embeddings, while retaining conservative Python-only unused-code analysis.

See [Installation](docs/install.md) for setup and runtime requirements.

## Quick start

```bash
codedupes check ./src
codedupes search ./src "normalize request payload"
codedupes info
```

See [Output and exit codes](docs/output.md) for report modes and CI behavior.

## Documentation

- [Installation](docs/install.md)
- [Polyglot language support](docs/polyglot-languages.md)
- [CLI reference](docs/cli.md)
- [Python API](docs/python-api.md)
- [Output and exit codes](docs/output.md)
- [Analysis defaults and heuristics](docs/analysis-defaults.md)
- [Semantic model profiles and tasks](docs/model-profiles.md)
- [Embedding cache](docs/caching.md)
- [Accelerators and Apple Silicon](docs/accelerators.md)
- [Hybrid gate tuning](docs/hybrid-tuning.md)
- [Next release changes](docs/release-notes-next.md)

## Scope and limits

Extraction is syntax-only; compiler preprocessing, macro expansion, and project-wide name resolution are outside its scope. See [language support](docs/polyglot-languages.md) for supported files, comparison boundaries, and parser limits, and [unused-code analysis](docs/analysis-defaults.md#potentially-unused-defaults) for the Python reference graph and its limitations.
