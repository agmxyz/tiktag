// CLI `download` subcommand. Fetches the built-in model bundle via hf-hub.
//
// hf-hub caches under ~/.cache/huggingface; we then `fs::copy` each required
// file into the profile's model_dir so the provenance hash (see main.rs
// build_json_provenance) reads from a stable in-tree location instead of the
// cache. Re-running `download` is cheap: hf-hub skips unchanged files.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Context;
use hf_hub::api::sync::{Api, ApiBuilder};
use log::info;

const INTERNAL_PROFILES_PATH: &str = "models/profiles.toml";
const REQUIRED_FILES: &[&str] = &["config.json", "tokenizer.json", "onnx/model_quantized.onnx"];

pub fn download() -> anyhow::Result<()> {
    let profiles_path = resolve_download_profiles_path()?;
    let profile = tiktag::Profiles::load(&profiles_path)?.resolve_default();
    let api = ApiBuilder::new()
        .build()
        .context("failed to init hf-hub api")?;
    fetch_bundle(&api, &profile)
}

/// Pick a destination for the bundle. Prefers `<exe_dir>/models/profiles.toml`
/// next to the installed binary; falls back to `<cwd>/models/profiles.toml` for
/// the `cargo run -- download` dev workflow (exe lives under `target/debug` or
/// `target/release`). Creates the destination if missing — this is the bootstrap
/// path, so neither candidate may exist yet.
fn resolve_download_profiles_path() -> anyhow::Result<PathBuf> {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(Path::to_path_buf));
    let dest_dir = match exe_dir {
        Some(dir) if is_cargo_target_dir(&dir) => PathBuf::from("."),
        Some(dir) => dir,
        None => PathBuf::from("."),
    };
    Ok(dest_dir.join(INTERNAL_PROFILES_PATH))
}

/// Dumb check: is this the `target/debug` or `target/release` dir from
/// `cargo build`? We want those to redirect to cwd so `cargo run -- download`
/// keeps working without shipping models under `target/`.
fn is_cargo_target_dir(dir: &Path) -> bool {
    let Some(name) = dir.file_name().and_then(|s| s.to_str()) else {
        return false;
    };
    if name != "debug" && name != "release" {
        return false;
    }
    dir.parent()
        .and_then(|p| p.file_name())
        .and_then(|s| s.to_str())
        == Some("target")
}

fn fetch_bundle(api: &Api, profile: &tiktag::ResolvedProfile) -> anyhow::Result<()> {
    info!(
        "downloading assets for model='{}' repo='{}' into '{}'",
        profile.name,
        profile.hf_repo,
        profile.model_dir.display()
    );

    fs::create_dir_all(&profile.model_dir)
        .with_context(|| format!("failed to create model dir {}", profile.model_dir.display()))?;

    let repo = api.model(profile.hf_repo.clone());

    for file in REQUIRED_FILES {
        let cached = repo
            .get(file)
            .with_context(|| format!("failed to download '{file}' from '{}'", profile.hf_repo))?;
        let dest = profile.model_dir.join(file);
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create subdir {}", parent.display()))?;
        }
        fs::copy(&cached, &dest).with_context(|| {
            format!("failed to copy {} -> {}", cached.display(), dest.display())
        })?;
    }

    tiktag::validate_model_bundle(&profile.model_dir)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{ApiBuilder, fetch_bundle};

    #[test]
    #[ignore = "network: hits huggingface.co"]
    fn downloads_real_bundle_from_hf() {
        let temp = tempfile::tempdir().expect("temp dir");
        let profile = tiktag::ResolvedProfile {
            name: "distilbert_ner_hrl".to_owned(),
            hf_repo: "Xenova/distilbert-base-multilingual-cased-ner-hrl".to_owned(),
            model_dir: temp.path().to_path_buf(),
            max_tokens: 512,
            overlap_tokens: 128,
        };
        let api = ApiBuilder::new().build().expect("api");
        fetch_bundle(&api, &profile).expect("fetch");
    }
}
