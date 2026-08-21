# Polyglot Language Support

codedupes supports Python, C, Rust, JavaScript, JSX, TypeScript, and TSX without turning its duplicate engine into a collection of language-specific special cases. The language backend owns parsing and feature extraction. The duplicate, embedding, ranking, and reporting stages consume the same `CodeUnit` model regardless of source language.

## Architecture

The analysis path is:

```text
source discovery
  -> language and dialect selection
  -> one parse per file
  -> executable code-unit extraction
  -> precomputed structural/token/identifier features
  -> traditional and semantic comparison
  -> language-aware reporting
```

Python keeps the CPython `ast` backend. C, Rust, JavaScript/JSX, TypeScript, and TSX use Tree-sitter. The registry in `src/codedupes/languages/registry.py` is the single authority for extensions, aliases, dialects, grammar package pins, and ambiguous C-header handling. `CodeExtractor` remains the public facade.

The Tree-sitter packages are ordinary mandatory dependencies:

```text
tree-sitter==0.25.2
tree-sitter-c==0.24.2
tree-sitter-rust==0.24.2
tree-sitter-javascript==0.25.0
tree-sitter-typescript==0.23.2
```

They are exact-pinned because grammar node kinds and field layouts are part of codedupes' behavior. Each upstream package ships a precompiled parser and exposes a Python capsule consumed by `tree_sitter.Language`. There is no runtime grammar download, compiler invocation, or fallback to line chunking.

The operational idea is versioned, locally available parsers. codedupes supports a small, deliberately tested language set and owns its extraction semantics. Pulling in a generic bundle containing many unused grammars would add packaging and compatibility surface without improving these four backends.

## Supported files

| Language | Extensions | Dialect behavior |
|---|---|---|
| Python | `.py`, optional `.pyi` | CPython AST |
| C | `.c`, conditionally `.h` | C grammar |
| Rust | `.rs` | Rust grammar |
| JavaScript | `.js`, `.mjs`, `.cjs` | JavaScript grammar |
| JSX | `.jsx` | JavaScript grammar with JSX syntax |
| TypeScript | `.ts`, `.mts`, `.cts` | TypeScript grammar |
| TSX | `.tsx` | Separate TSX grammar |

TypeScript declaration files ending in `.d.ts`, `.d.mts`, or `.d.cts` are skipped. They contain API declarations rather than implementation bodies and would produce misleading duplicate candidates.

### C headers

A `.h` file is ambiguous because both C and C++ use that extension. Automatic detection treats headers as C only when the scanned tree contains at least one `.c` file and no detected C++ source/header extension. Explicitly selecting C also accepts headers:

```bash
codedupes check ./include --language c --traditional-only
```

This is stricter than a retrieval tool's extension map. Duplicate fingerprints must not be generated from a plausibly wrong grammar.

## What becomes a code unit

### Python

The existing behavior remains the compatibility baseline: functions, async functions, methods, nested functions, classes, and nested classes. Source snippets remain complete source lines. Byte ranges now describe those exact emitted bytes, including Unicode-safe offsets.

### C

Only `function_definition` nodes are emitted. Prototypes, typedefs, variable declarations, function-pointer declarations, and macros are not functions. `static` functions are marked non-public; other definitions are marked public.

The declarator walker handles nested pointer, parenthesized, attributed, and function declarators rather than assuming the function name is a direct child.

### Rust

Body-bearing `function_item` nodes are emitted. Free and nested functions are `FUNCTION` units. Functions inside `impl` or trait bodies are `METHOD` units. Trait methods with default bodies are included; required signatures without bodies are skipped.

Inline test code is excluded by default: functions under a `#[cfg(test)]` (including `#[cfg(all(test, ...))]`) module or attribute, and free `#[test]` functions, are skipped. File-glob test exclusion cannot catch these because Rust inline test modules share source files with production code. `#[cfg(not(test))]` and `#[cfg(any(test, ...))]` gate real production configurations and stay extracted.

Lexical qualification includes modules, enclosing functions, implementation targets, and traits where available. Structs, enums, traits, and `impl` blocks are not flattened into fake classes. Their methods remain independently analyzable.

### JavaScript and JSX

The backend covers common modern executable forms:

- Function and generator declarations
- Named function expressions
- Arrow functions with stable bindings
- Class declarations and named class expressions
- Constructors, methods, getters, setters, async methods, and generators
- Class fields bound to functions or arrows
- Methods and bound function values in stably named objects
- ESM default exports
- Stable assignment targets such as `exports.run` and `module.exports.run`

Anonymous callbacks passed directly into calls are intentionally skipped. Their names and identities are unstable, and treating every callback as a top-level duplicate unit creates noise. An anonymous default export receives the deterministic name `default`.

Export marking stops at function-body boundaries: a unit nested inside an exported function is local scope, not a module export. A class body is not a boundary, so members of an exported class stay exported.

### TypeScript and TSX

TypeScript reuses the ECMAScript extraction rules while adding TypeScript-specific exclusions and wrappers. Implementations with bodies are emitted. Overload signatures, abstract method signatures, interface members, ambient declarations, and other bodyless declarations are skipped. For an overloaded function, the implementation is one unit; the preceding signatures are not.

TypeScript and TSX are distinct parser dialects even though both report the canonical language `typescript`.

## Source ranges and parse recovery

Tree-sitter is byte-addressed. The backend reads files as bytes, parses those bytes, and slices snippets with `start_byte:end_byte`; decoding happens only after the slice. This prevents Unicode text before a function from corrupting its range.

A missing or incompatible grammar is a configuration error and stops analysis. codedupes never substitutes arbitrary line chunks.

A file containing Tree-sitter recovery nodes is different: unaffected units can still be useful. codedupes emits a file-level `partial-parse` diagnostic, skips any extracted unit whose own syntax subtree contains an error, and emits a `unit-parse-error` diagnostic for that skipped unit. Diagnostics are available in terminal and JSON output.

## Fingerprints and comparison boundaries

Each backend computes features while its original syntax tree is in memory:

- A structural fingerprint with a schema version, canonical language, and unit type
- A token fingerprint retaining literal token text while ignoring comments and whitespace
- An identifier set for Jaccard near matching
- A statement count for semantic eligibility and hybrid scoring
- Direct call names for future language-specific reference analyzers

The structural stream includes significant anonymous operator tokens. `a + b` and `a - b` therefore cannot collapse merely because both expressions have the same named syntax nodes. Local identifiers are normalized by encounter order, while field/property/type names are preserved where they carry API or behavioral meaning. String values are normalized for structural matching; numeric values are preserved. Token matching remains stricter and retains literal text.

Traditional exact and Jaccard comparisons are blocked by canonical language and public unit type before pair generation. Overlapping units in the same file, such as a parent function and its nested function, are not reported as duplicates of each other.

Semantic duplicate checking is also same-language by default, and each language is gated by its own calibrated duplicate threshold from the model profile (see [Analysis defaults](analysis-defaults.md#semantic-duplicate-gate-defaults)). `--cross-language` opts into cross-language semantic pairs; those claims are uncalibrated, so a mixed pair is held to the looser of its two language gates. Semantic `search` remains cross-language because retrieval is exactly where that shared embedding space is useful.

## Unused-code analysis

Unused-code analysis remains Python-only in this release. The current reference graph understands Python imports, aliases, package entry points, `__all__`, `__main__`, pytest conventions, and Python public-name behavior.

Equivalent correctness elsewhere requires build-system and module resolution:

- C needs translation-unit and preprocessor context, ideally from `compile_commands.json`.
- Rust needs Cargo modules, traits, macros, `cfg`, generated code, and build scripts.
- JavaScript needs ESM/CommonJS resolution, package exports, re-exports, and dynamic imports.
- TypeScript additionally needs `tsconfig` path mappings, project references, and declaration semantics.

Running Python heuristics over those languages would create false dead-code claims. codedupes instead reports how many non-Python units were explicitly excluded.

## CLI use

Auto-detect all supported languages:

```bash
codedupes check .
codedupes search . "retry with exponential backoff"
```

Restrict a mixed repository with a repeatable filter:

```bash
codedupes check . --language rust --language typescript
codedupes search . "parse authorization header" --language js --language ts
```

Inspect parser package readiness:

```bash
codedupes info
```

`info` reports each parser dialect, its exact required package version, the installed version, and whether it is ready. Readiness is verified by actually constructing a parser and running an empty parse, so a wrong-platform or ABI-broken wheel is reported here instead of failing mid-analysis.

## Grammar upgrade procedure

Treat every grammar update as a behavioral change:

1. Change one exact package pin in `codedupes.languages.registry` and `pyproject.toml` (tests enforce that both match).
2. Construct its parser and run every extraction fixture (`pytest -m grammar`), including the golden structural-hash values.
3. Review changes in unit names, ranges, native kinds, statement counts, and fingerprints.
4. Run parser-independent normalization tests.
5. Run the per-language validator, sweep, and distribution report over `test_fixtures/polyglot_calibration/` and diff against the recorded tables in its README. The corpora measure each language's similarity scale under both built-in models, and the shipped per-language duplicate gates in `codedupes.semantic_profiles` are derived from these measurements — if a pin bump moves a language's recorded numbers, decide explicitly whether its gate must move with them.
6. Update the pin only after every difference is understood.

A semver-compatible grammar update can still rename a node or field. Broad version ranges would let an ordinary dependency refresh silently change duplicate reports.

## Known limits

Tree-sitter parses source syntax; it does not run the language toolchain. C macro expansion and conditional preprocessing, Rust macro expansion and `cfg` evaluation, and JavaScript/TypeScript runtime module resolution are outside this extraction layer. Code generated entirely by those mechanisms cannot be discovered without integrating the corresponding compiler or build graph.
