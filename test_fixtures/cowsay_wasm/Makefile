.PHONY: test fixture cli web serve

test:
	cargo test

fixture:
	python3 scripts/verify_source_fixture.py

cli:
	cargo run -- "Rust cows are memory safe."

web:
	wasm-pack build --target web --out-dir web/pkg --release

serve:
	python3 -m http.server 8080
