# Polyglot language support

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

Python keeps the CPython `ast` backend. C, Rust, JavaScript/JSX, TypeScript, and TSX use Tree-sitter. The registry in [registry](../src/codedupes/languages/registry.py) defines extensions, aliases, dialects, grammar package pins, and ambiguous C-header handling. `CodeExtractor` remains the public facade.

The [parser packages](install.md#polyglot-parser-dependencies) are exact-pinned because grammar node kinds and field layouts affect extraction.

## Supported files

`--language` accepts canonical names `python`, `c`, `rust`, `javascript`, and `typescript`, plus aliases `py`, `rs`, `js`, `jsx`, `ts`, and `tsx`. JSX and TSX aliases select their whole canonical language, not only that dialect.

| Language | Extensions | Dialect behavior |
| --- | --- | --- |
| Python | `.py`, optional `.pyi` | CPython AST |
| C | `.c`, conditionally `.h` | C grammar |
| Rust | `.rs` | Rust grammar |
| JavaScript | `.js`, `.mjs`, `.cjs` | JavaScript grammar |
| JSX | `.jsx` | JavaScript grammar with JSX syntax |
| TypeScript | `.ts`, `.mts`, `.cts` | TypeScript grammar |
| TSX | `.tsx` | Separate TSX grammar |

TypeScript declaration files ending in `.d.ts`, `.d.mts`, or `.d.cts` are skipped. They contain API declarations rather than implementation bodies and would produce misleading duplicate candidates.

### C headers

A lowercase `.h` file is ambiguous because both C and C++ use that extension. Automatic detection treats headers as C only when the scanned tree contains at least one `.c` file and no detected C++ source/header extension. Case-sensitive `.C` and `.H` suffixes are conventional C++ spellings and are never normalized into supported C files. The repository-wide ambiguity probe runs only when a lowercase `.h` candidate is encountered, so analyzing an explicit non-header file does not scan its sibling tree for an irrelevant header decision. The detection scan honors both default and custom extraction exclusions: C++ inside an excluded directory such as `node_modules` cannot flip the decision, while C++ in a directory the walk analyzes (including `vendor/`) disables header parsing rather than letting C++ headers be parsed with the C grammar. Explicitly selecting C also accepts lowercase `.h` headers:

```bash
codedupes check ./include --language c --traditional-only
```

This is stricter than a retrieval tool's extension map. Duplicate fingerprints must not be generated from a plausibly wrong grammar.

Skipped headers are reported rather than silently dropped: a directory scan emits one summary `c-header-policy` extraction diagnostic naming how many `.h` files it passed over and suggesting `--language c`. An explicitly named `.h` file gets its own diagnostic.

## What becomes a code unit

### Python

Python emits functions, async functions, methods, nested functions, classes, and nested classes. Source snippets retain complete source lines, and byte ranges describe those emitted bytes, including Unicode, BOM, and CRLF offsets.

### C

Only `function_definition` nodes are emitted. Prototypes, typedefs, variable declarations, function-pointer declarations, and macros are not functions. Internal linkage is read from the definition's storage-class specifier, so `static` functions are marked non-public while `int dst[static 4]` parameters and interleaved comments do not confuse the check; other definitions are marked public.

The declarator walker handles nested pointer, parenthesized, attributed, and function declarators rather than assuming the function name is a direct child.

### Rust

Body-bearing `function_item` nodes are emitted. Free and nested functions are `FUNCTION` units. Functions inside `impl` or trait bodies are `METHOD` units. Trait methods with default bodies are included; required signatures without bodies are skipped.

Inline test code is excluded by default: functions under a `#[cfg(test)]` (including `#[cfg(all(..., test, ...))]` regardless of predicate order) module or attribute, and free `#[test]` functions, are skipped. File-glob test exclusion cannot catch these because Rust inline test modules share source files with production code. `#[cfg(not(test))]` and `#[cfg(any(test, ...))]` gate real production configurations and stay extracted.

Lexical qualification includes modules, enclosing functions, implementation targets, and traits where available. Structs, enums, traits, and `impl` blocks are not flattened into fake classes. Their methods remain independently analyzable.

A default trait method inherits the enclosing trait's visibility. Methods in `impl Trait for Type` cannot carry `pub` themselves: when the trait is a bare name declared in the same file, its visibility determines whether those methods are public. Path-qualified and unresolved traits are treated as public because cross-file trait resolution is outside the extractor's scope.

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

A bound class expression uses its external binding as the unit identity even when the class also declares an internal name: `const Worker = class Implementation {}` is reported as `Worker`. The internal name is used only for an unbound named expression. This keeps deferred export clauses and field accessibility attached to the name through which surrounding code actually reaches the class.

Export marking stops at function-body boundaries: a unit nested inside an exported function is local scope, not a module export. A class body is not a boundary, so members of an exported class stay exported.

Export clauses mark units declared elsewhere in the same file: `export { name }`, `export { name as alias }`, and `export default name` mark that local unit exported even though it has no export ancestor of its own. A re-export clause naming another module (`export { name } from "./other"`) refers to that module's units and never marks a local unit exported.

### TypeScript and TSX

TypeScript reuses the ECMAScript extraction rules while adding TypeScript-specific exclusions and wrappers. Implementations with bodies are emitted. Overload signatures, abstract method signatures, interface members, ambient declarations, and other bodyless declarations are skipped. For an overloaded function, the implementation is one unit; the preceding signatures are not.

TypeScript and TSX are distinct parser dialects even though both report the canonical language `typescript`.

### Visibility filtering

`--no-private` (`include_private=False`) filters on each backend's computed visibility rather than a name-prefix subset of it: C internal linkage, Rust `pub` including the trait rules above, TypeScript `private`/`protected` accessibility modifiers, and `_`/`#`-prefixed member names. A filtered-out private class takes its members with it, matching the Python extractor - emitting a method whose owner was never reported would leak the container's internals under an unreachable name.

## Source ranges and parse recovery

Tree-sitter is byte-addressed. The backend reads files as bytes, parses those bytes, and slices snippets with `start_byte:end_byte`; decoding happens only after the slice. This prevents Unicode text before a function from corrupting its range.

A missing or incompatible grammar is a configuration error and stops analysis. codedupes never substitutes arbitrary line chunks.

A file containing Tree-sitter recovery nodes is different: unaffected units can still be useful. codedupes emits a file-level `partial-parse` diagnostic, skips any extracted unit whose own syntax subtree contains an error, and emits a `unit-parse-error` diagnostic for that skipped unit. See [diagnostic output](output.md#diagnostics) for how commands expose these records. Unreadable files emit `read-error` and are skipped; non-UTF-8 Tree-sitter files emit `invalid-utf8` while continuing with replacement characters.

## Fingerprints and comparison boundaries

Each backend computes features while its original syntax tree is in memory:

- A structural fingerprint with a schema version, canonical language, and unit type
- A token fingerprint retaining literal token text while ignoring comments and whitespace
- An identifier set for Jaccard near matching
- A statement count for semantic eligibility and hybrid scoring
- Direct call names for future language-specific reference analyzers

The structural stream includes significant anonymous operator tokens. `a + b` and `a - b` therefore cannot collapse merely because both expressions have the same named syntax nodes. Local identifiers are normalized by encounter order, while field/property/type names are preserved where they carry API or behavioral meaning. String values are normalized for structural matching; numeric values are preserved. JSX text is display copy, not structure, so it normalizes with the other string forms and two otherwise identical React components do not fingerprint apart on their labels. Token matching remains stricter and retains literal text.

Identifier matching is Unicode-aware. ECMAScript identifiers are Unicode from ES2015 on and Rust accepts non-ASCII identifiers, so a non-ASCII name yields a unit and identifier-set entries instead of being dropped by an ASCII-only pattern.

Traditional exact and Jaccard comparisons are blocked by canonical language and blocking kind before pair generation. Functions and methods share one `callable` kind, so a function copied into a class body stays comparable with its module-level original, matching how semantic pairing treats them; classes block separately. Exact matching stays same-language: a C and a Rust function cannot become exact duplicates because their canonical token streams align. Overlapping units in the same file, such as a parent function and its nested function, are not reported as duplicates of each other.

Semantic comparison follows the [per-language gates and cross-language policy](analysis-defaults.md#semantic-duplicate-gate-defaults). Semantic search retrieves across the selected languages.

## Unused-code analysis

The [unused-code heuristic](analysis-defaults.md#potentially-unused-defaults) evaluates Python only. Extending it requires translation-unit and preprocessor context for C, Cargo/module/trait resolution for Rust, and project-wide module resolution for JavaScript/TypeScript. Syntax extraction alone cannot establish those references.

## Parser readiness

Run `codedupes info` to inspect each parser dialect's required and installed package versions. Readiness checks construct a parser and run an empty parse, so a wrong-platform or ABI-broken wheel is reported before analysis.

## Grammar upgrade procedure

Treat every grammar update as a behavioral change:

1. Change one exact package pin in `codedupes.languages.registry` and `pyproject.toml` (tests enforce that both match).
2. Construct its parser and run every extraction fixture (`pytest -m grammar`), including the golden structural-hash values.
3. Review changes in unit names, ranges, native kinds, statement counts, and fingerprints.
4. Run parser-independent normalization tests.
5. Run the [calibration validator](../test_fixtures/polyglot_calibration/README.md#validation), [sweep, and distribution report](../test_fixtures/polyglot_calibration/README.md#re-running). Compare results with the recorded tables and reassess the [duplicate gates](analysis-defaults.md#semantic-duplicate-gate-defaults) if measurements change.
6. Update the pin only after every difference is understood.

A semver-compatible grammar update can still rename a node or field. Broad version ranges would let an ordinary dependency refresh silently change duplicate reports.

## Known limits

Tree-sitter parses source syntax; it does not run the language toolchain. C macro expansion and conditional preprocessing, Rust macro expansion and `cfg` evaluation, and JavaScript/TypeScript runtime module resolution are outside this extraction layer. Code generated entirely by those mechanisms cannot be discovered without integrating the corresponding compiler or build graph.
