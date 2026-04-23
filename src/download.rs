// CLI `download` subcommand. Fetches the built-in model bundle via hf-hub.
//
// hf-hub caches under ~/.cache/huggingface; we then `fs::copy` each required
// file into the profile's model_dir so the provenance hash (see main.rs
// build_json_provenance) reads from a stable in-tree location instead of the
// cache. Re-running `download` is cheap: hf-hub skips unchanged files.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, anyhow};
use hf_hub::api::sync::{Api, ApiBuilder};
use log::info;

const INTERNAL_PROFILES_PATH: &str = "models/profiles.toml";
const DEFAULT_PROFILES_TOML: &str = r#"hf_repo = "Xenova/distilbert-base-multilingual-cased-ner-hrl"
model_dir = "distilbert-base-multilingual-cased-ner-hrl"
max_tokens = 512
overlap_tokens = 128

[recognizers]
date_time = true
"#;
const REQUIRED_FILES: &[&str] = &["config.json", "tokenizer.json", "onnx/model_quantized.onnx"];

pub fn download() -> anyhow::Result<()> {
    let profiles_path = resolve_download_profiles_path()?;
    ensure_profiles_file(&profiles_path)?;
    let profile = tiktag::Profiles::load(&profiles_path)?.resolve_default();
    let api = ApiBuilder::new()
        .build()
        .context("failed to init hf-hub api")?;
    fetch_bundle(&api, &profile)
}

/// Pick a destination for the bundle.
/// - Dev (`cargo run`): use `<cwd>/models/profiles.toml`.
/// - Installed binary: use app-data path (`.../tiktag/models/profiles.toml`).
/// - If app-data file is absent but legacy exe-dir file exists, keep legacy path.
fn resolve_download_profiles_path() -> anyhow::Result<PathBuf> {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(Path::to_path_buf));
    if exe_dir.as_deref().is_some_and(is_cargo_target_dir) {
        return Ok(PathBuf::from(INTERNAL_PROFILES_PATH));
    }

    let app_candidate = crate::app_profiles_path();
    let legacy_exe_candidate = exe_dir.map(|dir| dir.join(INTERNAL_PROFILES_PATH));

    if let Some(path) = app_candidate.as_ref()
        && path.exists()
    {
        return Ok(path.to_path_buf());
    }
    if let Some(path) = legacy_exe_candidate.as_ref()
        && path.exists()
    {
        return Ok(path.to_path_buf());
    }

    if let Some(path) = app_candidate {
        return Ok(path);
    }
    if let Some(path) = legacy_exe_candidate {
        return Ok(path);
    }
    Err(anyhow!(
        "failed to resolve profile destination (no app-data dir and no executable path)"
    ))
}

fn ensure_profiles_file(path: &Path) -> anyhow::Result<()> {
    if path.exists() {
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create profiles dir {}", parent.display()))?;
    }
    fs::write(path, DEFAULT_PROFILES_TOML)
        .with_context(|| format!("failed to write bootstrap profile {}", path.display()))?;
    Ok(())
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
    use std::fs;

    use super::{ApiBuilder, DEFAULT_PROFILES_TOML, ensure_profiles_file, fetch_bundle};

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
            date_time_recognizer: true,
        };
        let api = ApiBuilder::new().build().expect("api");
        fetch_bundle(&api, &profile).expect("fetch");
    }

    #[test]
    fn bootstraps_profiles_file_when_missing() {
        let temp = tempfile::tempdir().expect("temp dir");
        let profiles_path = temp.path().join("models/profiles.toml");

        ensure_profiles_file(&profiles_path).expect("bootstrap profile");

        let written = fs::read_to_string(&profiles_path).expect("read profile");
        assert_eq!(written, DEFAULT_PROFILES_TOML);
    }

    #[test]
    fn does_not_overwrite_existing_profiles_file() {
        let temp = tempfile::tempdir().expect("temp dir");
        let profiles_path = temp.path().join("models/profiles.toml");
        let custom = r#"hf_repo = "custom/repo"
model_dir = "custom-dir"
max_tokens = 256
overlap_tokens = 64
"#;
        fs::create_dir_all(profiles_path.parent().expect("parent")).expect("mkdir");
        fs::write(&profiles_path, custom).expect("write custom");

        ensure_profiles_file(&profiles_path).expect("preserve custom");

        let written = fs::read_to_string(&profiles_path).expect("read profile");
        assert_eq!(written, custom);
    }
}
