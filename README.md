# tiktag

[![CI](https://github.com/agmxyz/tiktag/actions/workflows/ci.yml/badge.svg)](https://github.com/agmxyz/tiktag/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/tiktag.svg)](https://crates.io/crates/tiktag)
[![docs.rs](https://docs.rs/tiktag/badge.svg)](https://docs.rs/tiktag)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Rust library + CLI for text anonymization.

`tiktag` uses a built-in ONNX NER model for `PERSON`, `ORG`, and `LOCATION`, then applies additive regex recognizers such as email.

## Install

```bash
cargo install tiktag
```

Or from source:

```bash
cargo install --path .
```

## Quickstart

Download bundled model assets first:

```bash
tiktag download
```

CLI:

```bash
tiktag "Maria Garcia from OpenAI visited Berlin. Contact maria@example.com."
tiktag "Me llamo Máximo Décimo Meridio. Comandante de los Ejércitos de Roma, contáctame maximo@example.com."
echo "Maria Garcia from OpenAI visited Berlin." | tiktag --stdin --json
```

Library:

```rust
use std::path::Path;
use tiktag::Tiktag;

let profiles_path = Path::new("/path/to/downloaded/models/profiles.toml");
let mut tiktag = Tiktag::new(profiles_path)?;
let out = tiktag.anonymize("Maria Garcia from OpenAI visited Berlin.")?;
println!("{}", out.anonymization.anonymized_text);
```

`Tiktag::new` takes an explicit `profiles_path`; `model_dir` resolves relative to that file's parent.

## CLI

- `tiktag "<text>"` prints anonymized text
- `tiktag --stdin` reads input from stdin
- `tiktag --json` emits safe machine-readable output
- `tiktag --debug-json` emits reversible replacement metadata for local debugging only
- `tiktag --show-tokens` prints per-token predictions to stderr
- `tiktag download` fetches bundled model assets

## JSON

- `--json` fields: `schema_version`, `provenance`, `profile`, `anonymized_text`, `stats`
- `stats.timings` is machine-dependent; content-hash pipelines must ignore it
- additive field changes keep `schema_version`; breaking changes bump it

## Development

```bash
just verify         # clippy + test
just test-fixtures  # manual regression fixtures; requires downloaded assets
just bench          # manual performance checks
just smoke-package  # release packaging smoke check
```

Contributions are very welcome.

## Caveat

Model-based anonymization can miss entities. Treat `tiktag` as an assistive control, not a sole compliance or safety gate.

See [AGENTS.md](AGENTS.md) for project contract and invariants.
