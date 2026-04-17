// Internal model config loading for built-in multilingual DistilBERT NER model.
// The repo still keeps its model path and token limits in models/profiles.toml
// using default_profile + [profiles.<name>] TOML shape.

use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::error::TiktagError;

pub const BUILTIN_PROFILE_NAME: &str = "distilbert_ner_hrl";

/// The built-in model config after validation and path resolution.
#[derive(Debug, Clone)]
pub struct ResolvedProfile {
    pub name: String,
    pub hf_repo: String,
    pub model_dir: PathBuf,
    pub max_tokens: usize,
    pub overlap_tokens: usize,
}

/// Parsed internal config. Holds the base directory for resolving relative model_dir paths.
#[derive(Debug)]
pub struct Profiles {
    base_dir: PathBuf,
    profile: ProfileSpec,
}

#[derive(Debug, Clone)]
struct ProfileSpec {
    hf_repo: String,
    model_dir: PathBuf,
    max_tokens: usize,
    overlap_tokens: usize,
}

#[derive(Debug, Deserialize)]
struct ProfilesFileRaw {
    default_profile: String,
    profiles: std::collections::BTreeMap<String, ProfileRaw>,
}

#[derive(Debug, Deserialize)]
struct ProfileRaw {
    hf_repo: String,
    model_dir: PathBuf,
    max_tokens: usize,
    overlap_tokens: usize,
}

impl Profiles {
    pub fn load(path: &Path) -> Result<Self, TiktagError> {
        let raw_text = fs::read_to_string(path).map_err(|source| TiktagError::ProfileRead {
            path: path.to_path_buf(),
            source,
        })?;
        let base_dir = path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));

        // Parse errors get a dedicated typed variant so callers can distinguish
        // TOML decoding failures from semantic validation.
        let raw: ProfilesFileRaw =
            toml::from_str(&raw_text).map_err(|source| TiktagError::ProfileParse {
                path: path.to_path_buf(),
                source,
            })?;

        Self::validate_raw(&base_dir, raw).map_err(TiktagError::ProfileInvalid)
    }

    pub fn resolve_default(&self) -> ResolvedProfile {
        ResolvedProfile {
            name: BUILTIN_PROFILE_NAME.to_owned(),
            hf_repo: self.profile.hf_repo.clone(),
            model_dir: resolve_profile_model_dir(&self.base_dir, &self.profile.model_dir),
            max_tokens: self.profile.max_tokens,
            overlap_tokens: self.profile.overlap_tokens,
        }
    }

    fn validate_raw(base_dir: &Path, raw: ProfilesFileRaw) -> Result<Self, String> {
        if raw.default_profile != BUILTIN_PROFILE_NAME {
            return Err(format!("default_profile must be '{BUILTIN_PROFILE_NAME}'"));
        }

        if raw.profiles.len() != 1 || !raw.profiles.contains_key(BUILTIN_PROFILE_NAME) {
            return Err(format!(
                "profiles file must contain only [profiles.{BUILTIN_PROFILE_NAME}]"
            ));
        }

        let spec = raw
            .profiles
            .get(BUILTIN_PROFILE_NAME)
            .expect("built-in profile must exist after validation");

        if spec.hf_repo.trim().is_empty() {
            return Err(format!(
                "profile '{BUILTIN_PROFILE_NAME}' has empty hf_repo"
            ));
        }
        if spec.max_tokens == 0 {
            return Err(format!(
                "profile '{BUILTIN_PROFILE_NAME}' has invalid max_tokens=0"
            ));
        }
        // Reserve 2 tokens for [CLS] and [SEP]; overlap must fit inside the
        // remaining content budget or sliding-window stride makes no progress.
        let content_tokens = spec.max_tokens.saturating_sub(2);
        if spec.overlap_tokens >= content_tokens {
            return Err(format!(
                "profile '{BUILTIN_PROFILE_NAME}' has overlap_tokens={} which must be less than max_tokens - 2 ({})",
                spec.overlap_tokens, content_tokens
            ));
        }

        Ok(Self {
            base_dir: base_dir.to_path_buf(),
            profile: ProfileSpec {
                hf_repo: spec.hf_repo.clone(),
                model_dir: spec.model_dir.clone(),
                max_tokens: spec.max_tokens,
                overlap_tokens: spec.overlap_tokens,
            },
        })
    }

    #[cfg(test)]
    fn from_raw(base_dir: &Path, raw_text: &str) -> anyhow::Result<Self> {
        let raw: ProfilesFileRaw = toml::from_str(raw_text)?;
        Self::validate_raw(base_dir, raw).map_err(|msg| anyhow::anyhow!(msg))
    }
}

/// Resolve model_dir: absolute paths pass through, relative ones are joined to base_dir.
fn resolve_profile_model_dir(base_dir: &Path, model_dir: &Path) -> PathBuf {
    if model_dir.is_absolute() {
        model_dir.to_path_buf()
    } else {
        base_dir.join(model_dir)
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{BUILTIN_PROFILE_NAME, Profiles};

    #[test]
    fn loads_and_resolves_builtin_profile() {
        let profiles = Profiles::from_raw(
            &PathBuf::from("models"),
            r#"
default_profile = "distilbert_ner_hrl"

[profiles.distilbert_ner_hrl]
hf_repo = "Xenova/distilbert-base-multilingual-cased-ner-hrl"
model_dir = "distilbert-base-multilingual-cased-ner-hrl"
max_tokens = 512
overlap_tokens = 128
"#,
        )
        .expect("profiles should parse");

        let resolved = profiles.resolve_default();
        assert_eq!(resolved.name, BUILTIN_PROFILE_NAME);
        assert_eq!(
            resolved.hf_repo,
            "Xenova/distilbert-base-multilingual-cased-ner-hrl"
        );
        assert_eq!(
            resolved.model_dir,
            PathBuf::from("models/distilbert-base-multilingual-cased-ner-hrl")
        );
        assert_eq!(resolved.max_tokens, 512);
        assert_eq!(resolved.overlap_tokens, 128);
    }

    #[test]
    fn rejects_non_builtin_default_profile() {
        let err = Profiles::from_raw(
            &PathBuf::from("models"),
            r#"
default_profile = "missing"

[profiles.distilbert_ner_hrl]
hf_repo = "Xenova/distilbert-base-multilingual-cased-ner-hrl"
model_dir = "distilbert-base-multilingual-cased-ner-hrl"
max_tokens = 512
overlap_tokens = 128
"#,
        )
        .expect_err("non-built-in default profile should fail");

        assert!(
            err.to_string()
                .contains("default_profile must be 'distilbert_ner_hrl'")
        );
    }

    #[test]
    fn rejects_zero_max_tokens() {
        let err = Profiles::from_raw(
            &PathBuf::from("models"),
            r#"
default_profile = "distilbert_ner_hrl"

[profiles.distilbert_ner_hrl]
hf_repo = "Xenova/distilbert-base-multilingual-cased-ner-hrl"
model_dir = "distilbert-base-multilingual-cased-ner-hrl"
max_tokens = 0
overlap_tokens = 0
"#,
        )
        .expect_err("zero max_tokens should fail");

        assert!(err.to_string().contains("invalid max_tokens=0"));
    }

    #[test]
    fn rejects_additional_profiles() {
        let err = Profiles::from_raw(
            &PathBuf::from("models"),
            r#"
default_profile = "distilbert_ner_hrl"

[profiles.distilbert_ner_hrl]
hf_repo = "Xenova/distilbert-base-multilingual-cased-ner-hrl"
model_dir = "distilbert-base-multilingual-cased-ner-hrl"
max_tokens = 512
overlap_tokens = 128

[profiles.secondary]
hf_repo = "example/secondary"
model_dir = "secondary-model"
max_tokens = 512
overlap_tokens = 128
"#,
        )
        .expect_err("extra profiles should fail");

        assert!(
            err.to_string()
                .contains("must contain only [profiles.distilbert_ner_hrl]")
        );
    }

    #[test]
    fn rejects_missing_hf_repo() {
        let err = Profiles::from_raw(
            &PathBuf::from("models"),
            r#"
default_profile = "distilbert_ner_hrl"

[profiles.distilbert_ner_hrl]
model_dir = "distilbert-base-multilingual-cased-ner-hrl"
max_tokens = 512
overlap_tokens = 128
"#,
        )
        .expect_err("missing hf_repo should fail");

        assert!(err.to_string().contains("missing field `hf_repo`"));
    }

    #[test]
    fn rejects_overlap_exceeding_limit() {
        let err = Profiles::from_raw(
            &PathBuf::from("models"),
            r#"
default_profile = "distilbert_ner_hrl"

[profiles.distilbert_ner_hrl]
hf_repo = "Xenova/distilbert-base-multilingual-cased-ner-hrl"
model_dir = "distilbert-base-multilingual-cased-ner-hrl"
max_tokens = 512
overlap_tokens = 510
"#,
        )
        .expect_err("overlap >= max_tokens - 2 should fail");

        assert!(err.to_string().contains("overlap_tokens=510"));
    }
}
