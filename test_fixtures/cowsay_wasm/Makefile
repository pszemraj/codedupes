.PHONY: test fixture cli web serve

PYTHON ?= python

test:
	cargo test

fixture:
	$(PYTHON) ../../scripts/validate_calibration_corpus.py --project cowsay

cli:
	cargo run -- "Rust cows are memory safe."

web:
	wasm-pack build --target web --out-dir web/pkg --release

serve:
	$(PYTHON) -m http.server 8080
