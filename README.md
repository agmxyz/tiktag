# tiktag

Rust library + CLI for text anonymization. Ships a built-in multilingual NER model (`Xenova/distilbert-base-multilingual-cased-ner-hrl`, quantized ONNX).

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
echo "Contact maria@example.com" | tiktag --stdin
tiktag --stdin --json        < file.txt
tiktag --stdin --debug-json  < file.txt   # reversible map; debug only
tiktag download                           # fetch model assets
```

Flags: `--stdin`, `--json`, `--debug-json`, `--show-tokens`.

The CLI loads `models/profiles.toml` **relative to the current working directory** — run from the install root.

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

See `AGENTS.md` for the authoritative contract and known footguns.
