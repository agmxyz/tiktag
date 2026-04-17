use std::fs;
use std::path::Path;

use anyhow::Context;
use hf_hub::api::sync::{Api, ApiBuilder};
use log::info;

const INTERNAL_PROFILES_PATH: &str = "models/profiles.toml";
const REQUIRED_FILES: &[&str] = &["config.json", "tokenizer.json", "onnx/model_quantized.onnx"];

pub fn download() -> anyhow::Result<()> {
    let profile = tiktag::Profiles::load(Path::new(INTERNAL_PROFILES_PATH))?.resolve_default();
    let api = ApiBuilder::new()
        .build()
        .context("failed to init hf-hub api")?;
    fetch_bundle(&api, &profile)
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
