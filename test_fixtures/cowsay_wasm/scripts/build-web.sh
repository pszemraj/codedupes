#!/usr/bin/env sh
set -eu

wasm-pack build --target web --out-dir web/pkg --release
printf '%s\n' 'Built web/pkg. Serve the repository root and open /web/.'
