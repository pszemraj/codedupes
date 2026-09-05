# Cowsay dupe fixture

A working Rust application that targets both a native CLI and browser WebAssembly, with intentional code clones planted as detector ground truth. The application is small enough to inspect, but the clones sit on live paths: none of them are dead sample code.

## What is planted

| Fixture group | Kind | Live behavior |
| --- | --- | --- |
| `exact-border-builder` | exact / Type 1 | Both speech and thought bubbles call byte-identical border-pair builders in separate modules. |
| `bubble-renderers` | edit-distance / Type 3 | The speech and thought renderers retain the same construction skeleton, with changed control flow and delimiters. |
| `word-wrappers` | semantic / Type 4 | A stateful scanner and an iterator/fold implementation produce the same wrapped lines. The browser and CLI can select either implementation. |

Labels are stored in [`fixtures/clone-ground-truth.json`](fixtures/clone-ground-truth.json). Stable region markers are included in the Rust files, along with one-based line spans for tools that require coordinates. Prefer markers when source changes. The duplicated code is deliberate; do not deduplicate these implementations before evaluating a detector. `cargo test` checks that the exact clone remains exact and that the semantic implementations remain behaviorally equivalent.

Run the following commands from `test_fixtures/cowsay_wasm/` in the codedupes checkout.

## Native use

With Rust installed, run:

```sh
cargo test
cargo run -- "Rust cows are memory safe."
cargo run -- --think --width 24 --wrapper fold "I am considering ownership."
printf 'stdin works too\n' | cargo run -- --width 16
```

The CLI supports `--think`, `--width`, and `--wrapper scanner|fold`.

## Browser/WebAssembly use

With `wasm-pack` installed, run from the same fixture directory:

```sh
wasm-pack build --target web --out-dir web/pkg --release
python -m http.server 8080
```

Open `http://localhost:8080/web/`. Serve the project over HTTP rather than opening `web/index.html` directly; the generated JavaScript module needs to fetch its `.wasm` file.

The no-bundler browser path is deliberately plain: `wasm-pack` emits the ES module and Wasm binary into `web/pkg`, while `web/app.js` imports the generated module directly.

## Fixture checks

```sh
cargo test --test fixture_integrity
python scripts/verify_source_fixture.py
```

The Python check is dependency-free and validates the source markers, current line spans, exact-clone equality, and edit-clone similarity. The Rust tests add behavioral validation for the semantic clone.

## Project layout

```text
src/
  bubble/          exact and edit-distance clone groups
  wrapping/        semantic clone group
  lib.rs           native API and wasm-bindgen exports
  main.rs          native CLI
fixtures/
  clone-ground-truth.json
web/
  index.html
  app.js
  styles.css
scripts/
  build-web.sh
  verify_source_fixture.py
```

The wrapping width counts Unicode scalar values rather than terminal display cells. That is intentional here: bringing in a display-width dependency would add noise to a fixture whose target is source-clone detection.
