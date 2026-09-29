//! Text anonymization library backed by a built-in multilingual NER model.
//!
//! `tiktag` loads one bundled profile and exposes a small library surface:
//! construct [`Tiktag`] once, then call [`Tiktag::anonymize`] for each input.
//! The built-in pipeline uses the Xenova
//! `distilbert-base-multilingual-cased-ner-hrl` model for person, org, and
//! location entities, then applies additive regex recognizers such as email.
//!
//! # Example
//!
//! ```no_run
//! use std::path::Path;
//!
//! use tiktag::Tiktag;
//!
//! # fn main() -> Result<(), tiktag::TiktagError> {
//! let mut tiktag = Tiktag::new(Path::new("models/profiles.toml"))?;
//! let out = tiktag.anonymize("Maria Garcia from OpenAI visited Berlin.")?;
//! println!("{}", out.anonymization.anonymized_text);
//! # Ok(())
//! # }
//! ```
//!
//! # Notes
//!
//! - [`Tiktag::new`] is the expensive step: it loads the profile, tokenizer,
//!   and ONNX runtime session.
//! - [`Tiktag::anonymize`] takes `&mut self` and reuses that loaded runtime.
//! - Placeholder numbering is stable within one call only.
//! - Model-based anonymization can miss entities; treat `tiktag` as an
//!   assistive control, not a sole safety boundary.

mod anonymize;
mod decode;
mod error;
mod measurement;
mod model_bundle;

#[cfg(feature = "profiling")]
pub use measurement::{InitializationTimings, PipelineTimings};
use measurement::{Measurement, add_elapsed, elapsed};
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

/// Reusable anonymizer instance backed by one loaded profile and ONNX session.
#[derive(Debug)]
pub struct Tiktag {
    profile: profiles::ResolvedProfile,
    runtime: runtime::ModelRuntime,
}

/// Result of one [`Tiktag::anonymize`] call.
#[derive(Debug, Clone, PartialEq)]
pub struct TiktagOutput {
    /// Final anonymization payload for the input text.
    pub anonymization: AnonymizationResult,
    /// Tokenized sequence length used for inference before any windowing split.
    pub sequence_len: usize,
    /// Number of inference windows processed for this input.
    pub window_count: usize,
}

impl Tiktag {
    /// Loads the configured profile, tokenizer, and ONNX runtime session.
    ///
    /// This is the expensive constructor. Reuse one instance across many calls
    /// when possible.
    pub fn new(profiles_path: &Path) -> Result<Self, TiktagError> {
        let profiles = profiles::Profiles::load(profiles_path)?;
        let profile = profiles.resolve_default();
        let runtime = runtime::ModelRuntime::load(&profile, None, &mut Measurement::default())?;

        Ok(Self { profile, runtime })
    }

    /// Runs model inference plus enabled regex recognizers and rewrites the
    /// input into stable placeholders.
    ///
    /// Placeholder numbering is stable within this call only. The same exact
    /// normalized value in the same family reuses one placeholder.
    pub fn anonymize(&mut self, text: &str) -> Result<TiktagOutput, TiktagError> {
        self.anonymize_inner(text, &mut Measurement::default())
    }

    fn anonymize_inner(
        &mut self,
        text: &str,
        measurement: &mut Measurement,
    ) -> Result<TiktagOutput, TiktagError> {
        let total = measurement.start();
        let inference = self.runtime.infer(text, measurement)?;
        let start = measurement.start();
        let mut entities = inference.entities;
        // Keep this sequential for simplicity; if profiling shows need, regex recognizers can
        // run in parallel with inference prep and merge here with the same overlap rules.
        if self.profile.email_recognizer {
            entities.extend(recognizers::email::detect(text));
        }
        add_elapsed(&mut measurement.pipeline.recognizers_ms, start);
        let start = measurement.start();
        let anonymization = anonymize::anonymize(text, &entities)?;
        add_elapsed(&mut measurement.pipeline.masking_ms, start);
        measurement.pipeline.total_ms = elapsed(total);

        Ok(TiktagOutput {
            anonymization,
            sequence_len: inference.sequence_len,
            window_count: inference.window_count,
        })
    }

    /// Creates an instrumented session. An optional ORT trace prefix enables native
    /// profiling; use a separate run because tracing changes timings and memory.
    #[cfg(feature = "profiling")]
    pub fn new_measured(
        profiles_path: &Path,
        ort_trace_prefix: Option<&Path>,
    ) -> Result<(Self, InitializationTimings), TiktagError> {
        let mut measurement = Measurement {
            enabled: true,
            ..Default::default()
        };
        let start = measurement.start();
        let profile = profiles::Profiles::load(profiles_path)?.resolve_default();
        let runtime = runtime::ModelRuntime::load(&profile, ort_trace_prefix, &mut measurement)?;
        measurement.initialization.total_ms = elapsed(start);
        Ok((Self { profile, runtime }, measurement.initialization))
    }

    /// Runs the normal pipeline with numeric stage timings, without logging data.
    #[cfg(feature = "profiling")]
    pub fn anonymize_measured(
        &mut self,
        text: &str,
    ) -> Result<(TiktagOutput, PipelineTimings), TiktagError> {
        let mut measurement = Measurement {
            enabled: true,
            ..Default::default()
        };
        let output = self.anonymize_inner(text, &mut measurement)?;
        Ok((output, measurement.pipeline))
    }

    /// Flushes a native ORT trace and returns its filename.
    #[cfg(feature = "profiling")]
    pub fn end_profiling(&mut self) -> Result<String, TiktagError> {
        self.runtime.end_profiling()
    }

    /// Returns the logical name of the resolved built-in profile.
    pub fn profile_name(&self) -> &str {
        &self.profile.name
    }

    /// Returns the Hugging Face repo identifier for the loaded model bundle.
    pub fn hf_repo(&self) -> &str {
        &self.profile.hf_repo
    }

    /// Returns the resolved filesystem path to the model directory.
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
