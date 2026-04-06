// Profile loading and resolution. Profiles live in a TOML file (models/profiles.toml)
// and define which model to load, its token limit, and decode strategy.
// The file has a default_profile key plus a [profiles.<name>] table per model.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, anyhow, bail};
use serde::Deserialize;

use crate::decode::DecodeStrategy;

/// A profile after validation and path resolution — ready for the runtime.
#[derive(Debug, Clone)]
pub struct ResolvedProfile {
    pub name: String,
    pub hf_repo: String,
    pub model_dir: PathBuf,
    pub max_tokens: usize,
    pub decode_strategy: DecodeStrategy,
}

/// Parsed profiles file. Holds the base directory (for resolving relative model_dir paths),
/// the default profile name, and all parsed profile specs.
#[derive(Debug)]
pub struct Profiles {
    base_dir: PathBuf,
    default_profile: String,
    profiles: BTreeMap<String, ProfileSpec>,
}

/// Internal validated spec — mirrors ProfileRaw but lives past parsing.
#[derive(Debug, Clone)]
struct ProfileSpec {
    hf_repo: String,
    model_dir: PathBuf,
    max_tokens: usize,
    decode_strategy: DecodeStrategy,
}

/// Raw serde shape — maps 1:1 to the TOML on disk. No defaults allowed.
#[derive(Debug, Deserialize)]
struct ProfilesFileRaw {
    default_profile: String,
    profiles: BTreeMap<String, ProfileRaw>,
}

#[derive(Debug, Deserialize)]
struct ProfileRaw {
    hf_repo: String,
    model_dir: PathBuf,
    max_tokens: usize,
    decode_strategy: DecodeStrategy,
}

impl Profiles {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let raw_text = fs::read_to_string(path)
            .with_context(|| format!("failed to read profiles file {}", path.display()))?;
        let base_dir = path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        Self::from_raw(&base_dir, &raw_text)
            .with_context(|| format!("failed to parse TOML {}", path.display()))
    }

    pub fn resolve(&self, requested_profile: Option<&str>) -> anyhow::Result<ResolvedProfile> {
        let profile_name = requested_profile.unwrap_or(&self.default_profile);
        let spec = self.profiles.get(profile_name).ok_or_else(|| {
            let available = self.profiles.keys().cloned().collect::<Vec<_>>().join(", ");
            anyhow!("unknown profile '{profile_name}'. available profiles: {available}")
        })?;

        Ok(ResolvedProfile {
            name: profile_name.to_owned(),
            hf_repo: spec.hf_repo.clone(),
            model_dir: resolve_profile_model_dir(&self.base_dir, &spec.model_dir),
            max_tokens: spec.max_tokens,
            decode_strategy: spec.decode_strategy,
        })
    }

    fn from_raw(base_dir: &Path, raw_text: &str) -> anyhow::Result<Self> {
        let raw: ProfilesFileRaw = toml::from_str(raw_text)?;

        if raw.profiles.is_empty() {
            bail!("profiles file has no profiles");
        }

        if !raw.profiles.contains_key(raw.default_profile.as_str()) {
            bail!(
                "default_profile '{}' does not exist in profiles",
                raw.default_profile
            );
        }

        let mut profiles = BTreeMap::new();
        for (name, spec) in raw.profiles {
            if spec.hf_repo.trim().is_empty() {
                bail!("profile '{name}' has empty hf_repo");
            }
            if spec.max_tokens == 0 {
                bail!("profile '{name}' has invalid max_tokens=0");
            }

            profiles.insert(
                name,
                ProfileSpec {
                    hf_repo: spec.hf_repo,
                    model_dir: spec.model_dir,
                    max_tokens: spec.max_tokens,
                    decode_strategy: spec.decode_strategy,
                },
            );
        }

        Ok(Self {
            base_dir: base_dir.to_path_buf(),
            default_profile: raw.default_profile,
            profiles,
        })
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

    use super::Profiles;
    use crate::decode::DecodeStrategy;

    #[test]
    fn loads_and_resolves_default_profile() {
        let profiles = Profiles::from_raw(
            &PathBuf::from("models"),
            r#"
default_profile = "eu_pii"

[profiles.eu_pii]
hf_repo = "bardsai/eu-pii-anonimization-multilang"
model_dir = "eu-pii-anonimization-multilang"
max_tokens = 512
decode_strategy = "pii_relaxed"
"#,
        )
        .expect("profiles should parse");

        let resolved = profiles
            .resolve(None)
            .expect("default profile should resolve");
        assert_eq!(resolved.name, "eu_pii");
        assert_eq!(resolved.hf_repo, "bardsai/eu-pii-anonimization-multilang");
        assert_eq!(
            resolved.model_dir,
            PathBuf::from("models/eu-pii-anonimization-multilang")
        );
        assert_eq!(resolved.max_tokens, 512);
        assert_eq!(resolved.decode_strategy, DecodeStrategy::PiiRelaxed);
    }

    #[test]
    fn rejects_unknown_default_profile() {
        let err = Profiles::from_raw(
            &PathBuf::from("models"),
            r#"
default_profile = "missing"

[profiles.eu_pii]
hf_repo = "bardsai/eu-pii-anonimization-multilang"
model_dir = "eu-pii-anonimization-multilang"
max_tokens = 512
decode_strategy = "pii_relaxed"
"#,
        )
        .expect_err("missing default profile should fail");

        assert!(err.to_string().contains("default_profile 'missing'"));
    }

    #[test]
    fn rejects_zero_max_tokens() {
        let err = Profiles::from_raw(
            &PathBuf::from("models"),
            r#"
default_profile = "eu_pii"

[profiles.eu_pii]
hf_repo = "bardsai/eu-pii-anonimization-multilang"
model_dir = "eu-pii-anonimization-multilang"
max_tokens = 0
decode_strategy = "pii_relaxed"
"#,
        )
        .expect_err("zero max_tokens should fail");

        assert!(err.to_string().contains("invalid max_tokens=0"));
    }

    #[test]
    fn rejects_unknown_requested_profile() {
        let profiles = Profiles::from_raw(
            &PathBuf::from("models"),
            r#"
default_profile = "eu_pii"

[profiles.eu_pii]
hf_repo = "bardsai/eu-pii-anonimization-multilang"
model_dir = "eu-pii-anonimization-multilang"
max_tokens = 512
decode_strategy = "pii_relaxed"
"#,
        )
        .expect("profiles should parse");

        let err = profiles
            .resolve(Some("missing"))
            .expect_err("unknown profile should fail");

        assert!(err.to_string().contains("unknown profile 'missing'"));
    }

    #[test]
    fn rejects_missing_hf_repo() {
        let err = Profiles::from_raw(
            &PathBuf::from("models"),
            r#"
default_profile = "eu_pii"

[profiles.eu_pii]
model_dir = "eu-pii-anonimization-multilang"
max_tokens = 512
decode_strategy = "pii_relaxed"
"#,
        )
        .expect_err("missing hf_repo should fail");

        assert!(err.to_string().contains("missing field `hf_repo`"));
    }
}
