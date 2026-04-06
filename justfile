set shell := ["zsh", "-cu"]

default_profiles := "models/profiles.toml"

default:
    @just --list

build:
    cargo build

check:
    cargo check

test:
    cargo test

fmt:
    cargo fmt

run text profile='' profiles=default_profiles:
    #!/usr/bin/env bash
    set -euo pipefail
    args=(--profiles "{{ profiles }}")
    if [[ -n "{{ profile }}" ]]; then
      args+=(--profile "{{ profile }}")
    fi
    cargo run -- "${args[@]}" "{{ text }}"

run-json text profile='' profiles=default_profiles:
    #!/usr/bin/env bash
    set -euo pipefail
    args=(--profiles "{{ profiles }}")
    if [[ -n "{{ profile }}" ]]; then
      args+=(--profile "{{ profile }}")
    fi
    cargo run -- "${args[@]}" --json "{{ text }}"

run-tokens text profile='' profiles=default_profiles:
    #!/usr/bin/env bash
    set -euo pipefail
    args=(--profiles "{{ profiles }}")
    if [[ -n "{{ profile }}" ]]; then
      args+=(--profile "{{ profile }}")
    fi
    RUST_LOG=debug cargo run -- "${args[@]}" --show-tokens "{{ text }}"

sample profile='' profiles=default_profiles:
    #!/usr/bin/env bash
    set -euo pipefail
    args=(--profiles "{{ profiles }}")
    if [[ -n "{{ profile }}" ]]; then
      args+=(--profile "{{ profile }}")
    fi
    cargo run -- "${args[@]}" "Contact Maria Rossi at maria.rossi@example.it or +39 347 123 4567 in Milan."

download-profile profile profiles=default_profiles:
    #!/usr/bin/env bash
    set -euo pipefail

    profile="{{ profile }}"
    profiles="{{ profiles }}"
    profiles_dir="$(cd "$(dirname "$profiles")" && pwd)"

    repo="$(
      awk -v profile="$profile" '
        $0 ~ "^\\[profiles\\." profile "\\]$" { in_profile=1; next }
        in_profile && /^\[profiles\./ { exit }
        in_profile && $1 == "hf_repo" { gsub(/"/, "", $3); print $3; exit }
      ' "$profiles"
    )"

    model_dir="$(
      awk -v profile="$profile" '
        $0 ~ "^\\[profiles\\." profile "\\]$" { in_profile=1; next }
        in_profile && /^\[profiles\./ { exit }
        in_profile && $1 == "model_dir" { gsub(/"/, "", $3); print $3; exit }
      ' "$profiles"
    )"

    if [[ -z "$repo" || -z "$model_dir" ]]; then
      echo "failed to resolve hf_repo/model_dir for profile '$profile' from $profiles" >&2
      exit 1
    fi

    dest_dir="$profiles_dir/$model_dir"
    if [[ "$model_dir" = /* ]]; then
      dest_dir="$model_dir"
    fi

    hf download \
      "$repo" \
      config.json \
      tokenizer.json \
      onnx/model_quantized.onnx \
      --local-dir "$dest_dir"

download-default profiles=default_profiles:
    #!/usr/bin/env bash
    set -euo pipefail

    profiles="{{ profiles }}"
    default_profile="$(
      awk -F'"' '/^default_profile[[:space:]]*=/ { print $2; exit }' "$profiles"
    )"

    if [[ -z "$default_profile" ]]; then
      echo "failed to resolve default_profile from $profiles" >&2
      exit 1
    fi

    just download-profile "$default_profile" "$profiles"

search-supported query="multilingual ner onnx" limit="100":
    @hf models ls --search "{{ query }}" --limit {{ limit }} --expand siblings,pipeline_tag,library_name,downloads --format json | jq -r '.[] | select(.pipeline_tag=="token-classification") | . as $m | [(.siblings[]?.rfilename)] as $files | select(($files|index("config.json")) and ($files|index("tokenizer.json")) and ($files|index("onnx/model_quantized.onnx"))) | "\($m.id)\tdownloads=\($m.downloads // 0)\tlibrary=\($m.library_name // "unknown")"'
