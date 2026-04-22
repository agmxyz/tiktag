# tiktag

Rust library + CLI for text anonymization. Ships a built-in multilingual NER model (`Xenova/distilbert-base-multilingual-cased-ner-hrl`, quantized ONNX).

## Install

```bash
cargo install tiktag
```

Or build from source:

```bash
cargo install --path .
```

## Quickstart

```bash
tiktag download
tiktag "Maria Garcia from OpenAI visited Berlin."
echo "Maria Garcia from OpenAI visited Berlin." | tiktag --stdin --json
```

## Library

```rust
use std::path::Path;
use tiktag::Tiktag;

let mut tiktag = Tiktag::new(Path::new("models/profiles.toml"))?;
let out = tiktag.anonymize("Maria Garcia from OpenAI visited Berlin.")?;
println!("{}", out.anonymization.anonymized_text);
```

- `Tiktag::new` loads tokenizer + ONNX session once (~350 ms).
- `Tiktag::anonymize(&mut self, text)` reuses that state (ms-to-tens-of-ms). Wrap in a mutex to share across threads.
- Both return `Result<_, TiktagError>`.
- Placeholder numbering is stable within a single call; no cross-document identity.

## CLI

```bash
tiktag "Maria Garcia from OpenAI visited Berlin."
echo "Maria Garcia from OpenAI visited Berlin." | tiktag --stdin
tiktag --stdin --json < file.txt
tiktag --stdin --debug-json < file.txt  # reversible map; debug only
tiktag download                         # fetch model assets
```

Flags: `--stdin`, `--json`, `--debug-json`, `--show-tokens`.

The CLI resolves `models/profiles.toml` from an OS app-data directory first (`.../tiktag/models/profiles.toml`), then falls back to legacy locations (next to binary, then current working directory).

## JSON

`--json` fields: `schema_version`, `provenance`, `profile`, `anonymized_text`, `stats`.
`stats.timings` varies by machine; pipelines that hash output should ignore it.
Additive field changes keep `schema_version`; breaking changes bump it.

## Dev

```bash
just download         # fetch model assets
just test             # unit + integration
just test-fixtures    # end-to-end (requires assets)
just bench            # criterion throughput harness
just smoke-package    # release build + dist smoke
```

## Built-in profile

Fixed config path: `models/profiles.toml`. `model_dir` resolves relative to the config directory.
Required files under `model_dir`: `tokenizer.json`, `config.json`, `onnx/model_quantized.onnx`.

## Caveat

Model-based anonymization can miss entities. Use `tiktag` as an assistive control, not your only compliance/safety gate.

## Model attribution

- Model source: [`Xenova/distilbert-base-multilingual-cased-ner-hrl`](https://huggingface.co/Xenova/distilbert-base-multilingual-cased-ner-hrl)
- Model license/terms: see model card on Hugging Face.

See `AGENTS.md` for the authoritative contract and known footguns.
