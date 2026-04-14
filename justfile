# set shell := ["sh", "-cu"]

default:
    @just --list

build:
    cargo build

check:
    cargo check

clippy:
    cargo clippy --all-targets --all-features -- -D warnings

test:
    cargo test

test-fixtures:
    cargo test fixture_regression -- --ignored --nocapture

fmt:
    cargo fmt

run text:
    cargo run -- "{{ text }}"

run-json text:
    cargo run -- --json "{{ text }}"

run-debug-json text:
    cargo run -- --debug-json "{{ text }}"

run-tokens text:
    RUST_LOG=debug cargo run -- --show-tokens "{{ text }}"

run-stdin:
    cargo run -- --stdin

run-json-stdin:
    cargo run -- --stdin --json

run-debug-json-stdin:
    cargo run -- --stdin --debug-json

sample:
    cargo run -- "Maria Garcia from OpenAI visited Berlin on 2024-05-01."

download:
    cargo run -- download

package:
    cargo build --release
    test -f models/profiles.toml || (echo "missing models/profiles.toml" >&2; exit 1)
    test -f models/distilbert-base-multilingual-cased-ner-hrl/tokenizer.json || (echo "missing model assets; run 'just download'" >&2; exit 1)
    test -f models/distilbert-base-multilingual-cased-ner-hrl/config.json || (echo "missing model assets; run 'just download'" >&2; exit 1)
    test -f models/distilbert-base-multilingual-cased-ner-hrl/onnx/model_quantized.onnx || (echo "missing model assets; run 'just download'" >&2; exit 1)
    rm -rf dist/tiktag
    mkdir -p dist/tiktag/models/distilbert-base-multilingual-cased-ner-hrl/onnx
    cp target/release/tiktag dist/tiktag/tiktag
    cp models/profiles.toml dist/tiktag/models/profiles.toml
    cp models/distilbert-base-multilingual-cased-ner-hrl/tokenizer.json dist/tiktag/models/distilbert-base-multilingual-cased-ner-hrl/tokenizer.json
    cp models/distilbert-base-multilingual-cased-ner-hrl/config.json dist/tiktag/models/distilbert-base-multilingual-cased-ner-hrl/config.json
    cp models/distilbert-base-multilingual-cased-ner-hrl/onnx/model_quantized.onnx dist/tiktag/models/distilbert-base-multilingual-cased-ner-hrl/onnx/model_quantized.onnx

smoke-package:
    just package
    (cd dist/tiktag && PATH=/usr/bin:/bin; printf "Contact Maria at maria@example.com\n" | ./tiktag --stdin | grep "\[PERSON_1\]" >/dev/null)
    (cd dist/tiktag && PATH=/usr/bin:/bin; printf "Contact Maria at maria@example.com\n" | ./tiktag --stdin --json | grep "\"schema_version\": 1" >/dev/null)
