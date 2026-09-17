# Cowsay dupe fixture

A working Rust application that targets both a native CLI and browser WebAssembly. Its independently supported rendering paths are reviewed through the shared calibration contract; none is dead sample code.

## Reviewed maintenance families

| Fixture group | Kind | Live behavior |
| --- | --- | --- |
| `exact-border-builder` | exact / Type 1 | Both speech and thought bubbles call byte-identical border-pair builders in separate modules. |
| `bubble-renderers` | edit-distance / Type 3 | The speech and thought renderers retain the same construction skeleton, with changed control flow and delimiters. |
| `word-wrappers` | semantic / Type 4 | Scanner, iterator/fold, and FIFO queue implementations produce the same wrapped lines. The native CLI can select any implementation. |
| `render-assembly` | translated pipeline | Format-based and mutable-buffer assembly produce the same bubble-plus-cow result. The native CLI can select either renderer. |
| `worker-selection` | translated algorithm | Sorting eligible jobs and retaining a bounded priority heap produce the same ranked selection. The native CLI exercises the heap path. |

Labels, search relevance, contracts, and evidence references live in the shared [`calibration/annotations/cowsay.json`](../calibration/annotations/cowsay.json). The duplicated implementations remain supported application paths. `cargo test` checks that the exact pair remains exact and that the wrapping implementations remain behaviorally equivalent.

## Analyze the fixture

To inspect the deterministic clone with codedupes, run this from the repository root:

```sh
codedupes check test_fixtures/cowsay_wasm --language rust --traditional-only --no-unused --fail-on none
```

It reports the intentional `make_borders` exact pair. `--fail-on none` keeps that expected finding from failing the command. A normal combined `codedupes check` also evaluates the semantic wrapper pair after the embedding model is available.

## Native use

With Rust installed, run these commands from `test_fixtures/cowsay_wasm/` in the codedupes checkout:

```sh
cargo test
cargo run -- "Rust cows are memory safe."
cargo run -- --think --width 24 --wrapper queue --renderer composed "I am considering ownership."
cargo run -- --scheduler-demo
printf 'stdin works too\n' | cargo run -- --width 16
```

The CLI supports `--think`, `--width`, `--wrapper scanner|fold|queue`,
`--renderer pipeline|composed`, and `--scheduler-demo`. The browser-facing API
retains its two wrapper choices because its boolean argument is part of the
simple WebAssembly demo.

## Browser/WebAssembly use

With `wasm-pack` installed, run from the same fixture directory:

```sh
wasm-pack build --target web --out-dir web/pkg --release
make serve
```

Open `http://localhost:8080/web/`. Serve the project over HTTP rather than opening `web/index.html` directly; the generated JavaScript module needs to fetch its `.wasm` file.

The no-bundler browser path is deliberately plain: `wasm-pack` emits the ES module and Wasm binary into `web/pkg`, while `web/app.js` imports the generated module directly.

## Fixture checks

```sh
cargo test --test fixture_integrity
conda run --name inf python ../../scripts/validate_calibration_corpus.py --project cowsay
```

The shared validator checks selectors, judgments, evidence, search relevance, and detector eligibility. Rust tests preserve the exact/non-exact distinction and validate behavior; they do not require fixture source to hit a similarity range.

## Project layout

```text
src/
  bubble/          exact and edit-distance clone groups
  wrapping/        semantic clone group
  scheduler.rs      independently implemented worker selection paths
  lib.rs           native API and wasm-bindgen exports
  main.rs          native CLI
web/
  index.html
  app.js
  styles.css
scripts/
  build-web.sh
```

The wrapping width counts Unicode scalar values rather than terminal display cells. That is intentional here: bringing in a display-width dependency would add noise to a fixture whose target is source-clone detection.
