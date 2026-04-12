set shell := ["zsh", "-cu"]

default:
    @just --list

build:
    cargo build

check:
    cargo check

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

run-tokens text:
    RUST_LOG=debug cargo run -- --show-tokens "{{ text }}"

run-stdin:
    cargo run -- --stdin

run-json-stdin:
    cargo run -- --stdin --json

sample:
    cargo run -- "Contact Maria Rossi at maria.rossi@example.it or +39 347 123 4567 in Milan."

download:
    cargo run -- download
