.PHONY: test fixture cli web serve

test:
	cargo test

fixture:
	conda run --name inf python ../../scripts/validate_calibration_corpus.py --project cowsay

cli:
	cargo run -- "Rust cows are memory safe."

web:
	wasm-pack build --target web --out-dir web/pkg --release

serve:
	python3 -m http.server 8080
