//! Opt-in wall-clock measurements. No text, token values, or entity values.
use std::time::Instant;

use serde::Serialize;

/// Milliseconds accumulated across all windows in one call.
#[derive(Debug, Default, Clone, Serialize)]
pub struct PipelineTimings {
    pub tokenization_ms: f64,
    pub window_preparation_ms: f64,
    pub tensor_preparation_ms: f64,
    pub model_execution_ms: f64,
    pub decoding_ms: f64,
    pub stitching_ms: f64,
    pub recognizers_ms: f64,
    pub masking_ms: f64,
    pub total_ms: f64,
}

/// Constructor wall times; total includes profile loading and validation.
#[derive(Debug, Default, Clone, Serialize)]
pub struct InitializationTimings {
    pub tokenizer_ms: f64,
    pub session_ms: f64,
    pub total_ms: f64,
}

#[derive(Default)]
pub(crate) struct Measurement {
    pub enabled: bool,
    pub pipeline: PipelineTimings,
    pub initialization: InitializationTimings,
}

impl Measurement {
    pub fn start(&self) -> Option<Instant> {
        self.enabled.then(Instant::now)
    }
}

pub(crate) fn elapsed(start: Option<Instant>) -> f64 {
    start.map_or(0.0, |start| start.elapsed().as_secs_f64() * 1000.0)
}

pub(crate) fn add_elapsed(target_ms: &mut f64, start: Option<Instant>) {
    if let Some(start) = start {
        *target_ms += start.elapsed().as_secs_f64() * 1000.0;
    }
}
