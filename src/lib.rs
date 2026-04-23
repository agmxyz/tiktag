// Public library surface.
//
// Lifecycle: `Tiktag::new` loads profile + tokenizer + ONNX session once (hundreds
// of ms); `Tiktag::anonymize` reuses that state per call (ms-to-tens-of-ms).
// Hosts that serve many documents should construct one `Tiktag` and reuse it;
// locking and threading are the host's responsibility.
// macOS/CoreML footgun: compile cache is process-local, so CLI invocations pay
// compile repeatedly while long-lived library hosts amortize cost.
//
// Pipeline per call (see module docs for details):
//   runtime::infer  → tokenize, optional sliding window, ONNX forward, decode entities
//   anonymize::anonymize → filter/merge entities into stable placeholders, rewrite text

mod anonymize;
mod decode;
mod error;
mod model_bundle;
mod profiles;
mod recognizers;
mod runtime;
mod window;

#[cfg(test)]
mod fixture_tests;

use std::path::Path;

pub use anonymize::{AnonymizationResult, PlaceholderFamily, Replacement};
pub use error::{Result as TiktagResult, TiktagError};
pub use model_bundle::{REQUIRED_MODEL_FILES, missing_model_files, validate_model_bundle};
pub use profiles::{BUILTIN_PROFILE_NAME, Profiles, ResolvedProfile};

#[derive(Debug)]
pub struct Tiktag {
    profile: profiles::ResolvedProfile,
    runtime: runtime::ModelRuntime,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TiktagOutput {
    pub anonymization: AnonymizationResult,
    pub sequence_len: usize,
    pub window_count: usize,
}

impl Tiktag {
    pub fn new(profiles_path: &Path) -> Result<Self, TiktagError> {
        let profiles = profiles::Profiles::load(profiles_path)?;
        let profile = profiles.resolve_default();
        let runtime = runtime::ModelRuntime::load(&profile)?;

        Ok(Self { profile, runtime })
    }

    pub fn anonymize(&mut self, text: &str) -> Result<TiktagOutput, TiktagError> {
        let inference = self.runtime.infer(text)?;
        let mut entities = inference.entities;
        if self.profile.date_time_recognizer {
            entities.extend(recognizers::date_time::detect(text));
        }
        let anonymization = anonymize::anonymize(text, &entities)?;

        Ok(TiktagOutput {
            anonymization,
            sequence_len: inference.sequence_len,
            window_count: inference.window_count,
        })
    }

    pub fn profile_name(&self) -> &str {
        &self.profile.name
    }

    pub fn hf_repo(&self) -> &str {
        &self.profile.hf_repo
    }

    pub fn model_dir(&self) -> &Path {
        &self.profile.model_dir
    }
}

#[cfg(test)]
mod tests {
    use super::Tiktag;
    use std::path::PathBuf;

    #[test]
    fn constructor_surfaces_missing_profiles_path() {
        let err = Tiktag::new(&PathBuf::from("missing/profiles.toml"))
            .expect_err("missing profiles file should fail");

        assert!(err.to_string().contains("failed to read profile file"));
        assert!(err.to_string().contains("missing/profiles.toml"));
    }
}
