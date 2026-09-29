# tiktag

[![CI](https://github.com/agmxyz/tiktag/actions/workflows/ci.yml/badge.svg)](https://github.com/agmxyz/tiktag/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/tiktag.svg)](https://crates.io/crates/tiktag)
[![docs.rs](https://docs.rs/tiktag/badge.svg)](https://docs.rs/tiktag)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Rust library + CLI for text anonymization. Use it to redact logs, prompts, tickets, and exports.

`tiktag` uses a built-in ONNX NER model for `PERSON`, `ORG`, and `LOCATION`, then applies additive regex recognizers such as email.

Built-in model: [Xenova/distilbert-base-multilingual-cased-ner-hrl](https://huggingface.co/Xenova/distilbert-base-multilingual-cased-ner-hrl) (quantized ONNX)

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
tiktag "Me llamo Máximo Décimo Meridio. Comandante de los Ejércitos de Roma, contáctame maximo@example.com."
echo "Maria Garcia from OpenAI visited Berlin." | tiktag --stdin --json
```

Output example (excerpt):

```bash
$ tiktag --json "Me llamo Máximo Décimo Meridio. Comandante de los Ejércitos de Roma, contáctame maximo@example.com."
```

```json
{
  "anonymized_text": "Me llamo [PERSON_1]. Comandante de los Ejércitos de [LOCATION_1], contáctame [EMAIL_ADDRESS_1].",
  "stats": {
    "detected_entity_count": 3,
    "accepted_replacement_count": 3
  }
}
```

Library:

```rust
use std::path::Path;
use tiktag::Tiktag;

let profiles_path = Path::new("/path/to/downloaded/models/profiles.toml");
let mut tiktag = Tiktag::new(profiles_path)?;
let out = tiktag.anonymize("Text to anonymize.")?;
println!("{}", out.anonymization.anonymized_text);
```

`Tiktag::new` takes an explicit `profiles_path`; `model_dir` resolves relative to that file's parent.

## Performance

Baseline warm `anonymize` latency with the built-in quantized model on Apple
M2 Pro (macOS 26.6.2, arm64), Rust 1.88 release build, native ONNX Runtime
1.24.2. Each synthetic fixture ran in three fresh processes with 30 warm calls
per process after five warmups; values pool 90 calls and report p50/p95 in ms.

| Fixture | Input tokens (including special) | Windows | p50 / p95 (ms) |
| --- | ---: | ---: | ---: |
| short | 10 | 1 | 3.720 / 4.098 |
| near limit | 503 | 1 | 67.250 / 70.735 |
| multi-window | 1,586 | 4 | 280.222 / 314.346 |

`Tiktag::new` initialization medians: 340.411 ms (short), 376.830 ms (near
limit), 333.992 ms (four-window); excluded above. Reuse one `Tiktag` in
long-lived hosts; each CLI invocation pays initialization. The macOS build
registered CoreML with CPU fallback; provider placement can vary. Linux is
unmeasured. These synthetic results are not a performance guarantee. See the
[full report](docs/performance-report.md) for limitations, correctness findings,
raw results, and reproduction steps.

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
