// ONNX model loading and inference. This is the core runtime:
//   1. Validate the model bundle (tokenizer.json, config.json, onnx/model_quantized.onnx)
//   2. Load tokenizer, labels from config.json id2label, and ONNX session
//   3. Validate the model/config contract for token-classification logits
//   4. Tokenize input text, run the ONNX graph, decode entity spans from logits

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use log::{debug, info};
use ndarray::{Array2, Ix3};
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::{Outlet, TensorRef};
use serde_json::Value;
use tokenizers::Tokenizer;
use tokenizers::utils::truncation::{TruncationDirection, TruncationParams, TruncationStrategy};

use crate::decode::{EntitySpan, argmax_label_indices_with_probs, decode_entities};
use crate::error::TiktagError;
use crate::model_bundle::validate_model_bundle;
use crate::profiles::ResolvedProfile;
use crate::window::WindowEntities;

#[derive(Debug)]
pub(crate) struct InferenceResult {
    pub entities: Vec<EntitySpan>,
    pub sequence_len: usize,
    pub window_count: usize,
}

#[derive(Debug)]
struct WindowedInferenceResult {
    entities: Vec<EntitySpan>,
    window_count: usize,
}

/// Holds everything needed to run inference: the tokenizer, label map, ONNX session,
/// and profile settings. Created via ModelRuntime::load().
#[derive(Debug)]
pub(crate) struct ModelRuntime {
    tokenizer: Tokenizer,
    labels: Vec<String>,
    session: Session,
    profile_name: String,
    max_tokens: usize,
    overlap_tokens: usize,
    logits_output_name: String,
    /// BERT-family models need token_type_ids; DistilBERT doesn't. Detected at load time.
    has_token_type_ids: bool,
}

impl ModelRuntime {
    pub(crate) fn load(profile: &ResolvedProfile) -> Result<Self, TiktagError> {
        validate_model_bundle(&profile.model_dir)?;

        let tokenizer_path = profile.model_dir.join("tokenizer.json");
        let config_path = profile.model_dir.join("config.json");
        let model_path = profile.model_dir.join("onnx/model_quantized.onnx");

        info!(
            "loading model assets for profile='{}' from '{}'",
            profile.name,
            profile.model_dir.display()
        );

        let mut tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|err| {
            TiktagError::Tokenizer(format!(
                "failed to load tokenizer at {}: {err}",
                tokenizer_path.display()
            ))
        })?;
        // Disable the tokenizer's built-in truncation so we can enforce max_tokens ourselves
        // with a clear error message instead of silently dropping tokens.
        tokenizer.with_truncation(None).map_err(|err| {
            TiktagError::Tokenizer(format!(
                "failed to disable truncation in tokenizer {}: {err}",
                tokenizer_path.display()
            ))
        })?;

        let labels = load_labels(&config_path)?;

        // Initialize ORT once per process. On macOS, CoreML compile cache is
        // not persisted to disk, so short-lived CLI runs re-pay compile cost.
        ort::init().commit();

        let session_builder = Session::builder().map_err(ort_error)?;

        #[cfg(target_os = "macos")]
        let session_builder = {
            // Register CoreML EP on macOS; ORT may silently fall back to CPU.
            let builder = session_builder
                .with_execution_providers([ort::ep::CoreML::default().build()])
                .map_err(ort_error)?;
            info!("coreml execution provider registered");
            builder
        };

        #[cfg(not(target_os = "macos"))]
        {
            info!("cpu execution provider (default)");
        }

        let session = session_builder
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(ort_error)?
            .commit_from_file(model_path)
            .map_err(|err| {
                TiktagError::OrtRuntime(format!("failed to create ONNX Runtime session: {err}"))
            })?;

        let logits_output_name =
            validate_logits_output_metadata(&profile.name, &labels, session.outputs())?;

        // Some ONNX exports (e.g. BERT-family) require a token_type_ids input; others
        // (e.g. DistilBERT) don't. Probe the session graph to decide at load time.
        let has_token_type_ids = session
            .inputs()
            .iter()
            .any(|input| input.name() == "token_type_ids");

        Ok(Self {
            tokenizer,
            labels,
            session,
            profile_name: profile.name.clone(),
            max_tokens: profile.max_tokens,
            overlap_tokens: profile.overlap_tokens,
            logits_output_name,
            has_token_type_ids,
        })
    }

    pub(crate) fn infer(&mut self, text: &str) -> Result<InferenceResult, TiktagError> {
        info!("running inference on {} input bytes", text.len());
        let show_tokens = log::log_enabled!(target: "tokens", log::Level::Debug);

        // Tokenize without truncation to measure true sequence length.
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|err| TiktagError::Tokenizer(format!("tokenization failed: {err}")))?;
        let seq_len = encoding.len();
        if seq_len == 0 {
            return Err(TiktagError::Tokenizer(
                "tokenizer returned an empty encoding".to_owned(),
            ));
        }
        info!("encoded sequence length: {seq_len}");

        if seq_len <= self.max_tokens {
            let entities = self.infer_single_encoding(text, &encoding, show_tokens)?;
            Ok(InferenceResult {
                entities,
                sequence_len: seq_len,
                window_count: 1,
            })
        } else if self.overlap_tokens == 0 {
            ensure_sequence_within_limit(&self.profile_name, seq_len, self.max_tokens)?;
            unreachable!()
        } else {
            info!(
                "input exceeds max_tokens={}, using sliding window (overlap_tokens={})",
                self.max_tokens, self.overlap_tokens
            );
            let result = self.infer_windowed(text, show_tokens)?;
            Ok(InferenceResult {
                entities: result.entities,
                sequence_len: seq_len,
                window_count: result.window_count,
            })
        }
    }

    /// Run ONNX inference on a single encoding and decode entities.
    /// Shared by both the single-pass fast path and each window of the windowed path.
    fn infer_single_encoding(
        &mut self,
        text: &str,
        encoding: &tokenizers::Encoding,
        show_tokens: bool,
    ) -> Result<Vec<EntitySpan>, TiktagError> {
        let seq_len = encoding.len();

        let input_ids = Array2::from_shape_vec(
            (1, seq_len),
            encoding.get_ids().iter().map(|&id| i64::from(id)).collect(),
        )
        .map_err(|err| {
            TiktagError::OrtRuntime(format!("failed to build input_ids tensor: {err}"))
        })?;
        let attention_mask = Array2::from_shape_vec(
            (1, seq_len),
            encoding
                .get_attention_mask()
                .iter()
                .map(|&mask| i64::from(mask))
                .collect(),
        )
        .map_err(|err| {
            TiktagError::OrtRuntime(format!("failed to build attention_mask tensor: {err}"))
        })?;

        let outputs = if self.has_token_type_ids {
            let token_type_ids = Array2::<i64>::zeros((1, seq_len));
            self.session
                .run(ort::inputs! {
                    "input_ids" => TensorRef::from_array_view(input_ids.view()).map_err(ort_error)?,
                    "attention_mask" => TensorRef::from_array_view(attention_mask.view()).map_err(ort_error)?,
                    "token_type_ids" => TensorRef::from_array_view(token_type_ids.view()).map_err(ort_error)?,
                })
                .map_err(ort_error)?
        } else {
            self.session
                .run(ort::inputs! {
                    "input_ids" => TensorRef::from_array_view(input_ids.view()).map_err(ort_error)?,
                    "attention_mask" => TensorRef::from_array_view(attention_mask.view()).map_err(ort_error)?,
                })
                .map_err(ort_error)?
        };

        let logits = outputs[0].try_extract_array::<f32>().map_err(|err| {
            TiktagError::OrtRuntime(format!("failed to extract logits as f32 tensor: {err}"))
        })?;
        let logits = logits.into_dimensionality::<Ix3>().map_err(|err| {
            TiktagError::OrtRuntime(format!(
                "expected logits with shape [batch, seq, labels]: {err}"
            ))
        })?;
        validate_runtime_logits_shape(
            &self.profile_name,
            &self.logits_output_name,
            &self.labels,
            logits.shape(),
        )?;

        let predictions = argmax_label_indices_with_probs(logits);

        if show_tokens {
            for (index, token) in encoding.get_tokens().iter().enumerate() {
                let (label_idx, prob) = predictions.get(index).copied().unwrap_or((0, 0.0));
                let label = self
                    .labels
                    .get(label_idx)
                    .map(String::as_str)
                    .unwrap_or("<unknown>");
                let (start, end) = encoding.get_offsets().get(index).copied().unwrap_or((0, 0));
                debug!(target: "tokens", "{index:>3}: {token:<20} {label:<32} (prob={prob:.2}) [{start}..{end}]");
            }
        }

        let entities = decode_entities(text, encoding, &predictions, &self.labels);
        Ok(entities)
    }

    /// Sliding-window inference: re-tokenize with truncation+stride, run each window
    /// through ONNX, stitch entities across windows.
    fn infer_windowed(
        &mut self,
        text: &str,
        show_tokens: bool,
    ) -> Result<WindowedInferenceResult, TiktagError> {
        let mut windowed_tokenizer = self.tokenizer.clone();
        windowed_tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: self.max_tokens,
                stride: self.overlap_tokens,
                strategy: TruncationStrategy::OnlyFirst,
                direction: TruncationDirection::Right,
            }))
            .map_err(|err| {
                TiktagError::Tokenizer(format!("failed to configure windowed truncation: {err}"))
            })?;

        let mut primary = windowed_tokenizer.encode(text, true).map_err(|err| {
            TiktagError::Tokenizer(format!("windowed tokenization failed: {err}"))
        })?;
        let overflow = primary.take_overflowing();

        let mut encodings = Vec::with_capacity(1 + overflow.len());
        encodings.push(primary);
        encodings.extend(overflow);
        info!("sliding window: {} windows", encodings.len());

        let mut window_results = Vec::with_capacity(encodings.len());
        for (i, encoding) in encodings.iter().enumerate() {
            if show_tokens {
                debug!(target: "tokens", "--- window {i} ---");
            }

            let entities = self.infer_single_encoding(text, encoding, show_tokens)?;
            let (emit_start, emit_end) =
                compute_emit_region(text, encoding, i, encodings.len(), self.overlap_tokens);
            info!(
                "window {i}: {} entities, emit region [{emit_start}..{emit_end})",
                entities.len()
            );

            window_results.push(WindowEntities {
                entities,
                emit_start,
                emit_end,
            });
        }

        let entities = crate::window::stitch(window_results);
        info!("stitched to {} entities", entities.len());

        Ok(WindowedInferenceResult {
            entities,
            window_count: encodings.len(),
        })
    }
}

/// Parse config.json's id2label map into a contiguous Vec<String> indexed by label id.
/// Rejects sparse maps (gaps would create silent decode bugs).
fn load_labels(config_path: &Path) -> Result<Vec<String>, TiktagError> {
    let config_text = fs::read_to_string(config_path).map_err(|err| {
        TiktagError::Config(format!("failed to read {}: {err}", config_path.display()))
    })?;
    let config: Value = serde_json::from_str(&config_text).map_err(|err| {
        TiktagError::Config(format!("failed to parse {}: {err}", config_path.display()))
    })?;

    let id2label = config
        .get("id2label")
        .and_then(Value::as_object)
        .ok_or_else(|| TiktagError::Config("config.json is missing id2label".to_owned()))?;

    let mut labels = BTreeMap::new();
    for (key, value) in id2label {
        let index = key
            .parse::<usize>()
            .map_err(|err| TiktagError::Config(format!("invalid label index {key}: {err}")))?;
        let label = value
            .as_str()
            .ok_or_else(|| TiktagError::Config(format!("label {key} is not a string")))?;
        labels.insert(index, label.to_owned());
    }

    let max_index = labels
        .keys()
        .next_back()
        .copied()
        .ok_or_else(|| TiktagError::Config("id2label is empty".to_owned()))?;

    let mut ordered = vec![String::new(); max_index + 1];
    for (index, label) in labels {
        ordered[index] = label;
    }

    if let Some(gap) = ordered.iter().position(|label| label.is_empty()) {
        return Err(TiktagError::Config(format!(
            "id2label has a gap at index {gap} — all indices from 0 to {max_index} must be present"
        )));
    }

    Ok(ordered)
}

fn validate_logits_output_metadata(
    profile_name: &str,
    labels: &[String],
    outputs: &[Outlet],
) -> Result<String, TiktagError> {
    let output = outputs.first().ok_or_else(|| {
        TiktagError::Config(format!(
            "profile '{profile_name}' ONNX model has no outputs; expected token-classification logits output"
        ))
    })?;
    let output_name = output.name().to_owned();
    let shape = output.dtype().tensor_shape().ok_or_else(|| {
        TiktagError::Config(format!(
            "profile '{profile_name}' output '{output_name}' is not a tensor; expected token-classification logits [batch, seq, labels]"
        ))
    })?;
    let shape = &shape[..];

    if shape.len() != 3 {
        return Err(TiktagError::Config(format!(
            "profile '{profile_name}' output '{output_name}' has shape {shape:?}; expected rank-3 logits [batch, seq, labels]"
        )));
    }

    let onnx_label_count = shape[2];
    if onnx_label_count >= 0 && onnx_label_count as usize != labels.len() {
        return Err(TiktagError::Config(format!(
            "profile '{profile_name}' label schema mismatch: config.json id2label has {} labels, but ONNX output '{output_name}' declares {} labels in shape {:?}. Ensure config.json and onnx/model_quantized.onnx come from the same model export.",
            labels.len(),
            onnx_label_count,
            shape
        )));
    }

    Ok(output_name)
}

fn validate_runtime_logits_shape(
    profile_name: &str,
    output_name: &str,
    labels: &[String],
    shape: &[usize],
) -> Result<(), TiktagError> {
    if shape.len() != 3 {
        return Err(TiktagError::Config(format!(
            "profile '{profile_name}' output '{output_name}' has shape {shape:?}; expected rank-3 logits [batch, seq, labels]"
        )));
    }

    let actual_count = shape[2];
    if actual_count != labels.len() {
        return Err(TiktagError::Config(format!(
            "profile '{profile_name}' label schema mismatch: config.json id2label has {} labels, but ONNX output '{output_name}' produced logits with {} labels (shape {:?}). Ensure config.json and onnx/model_quantized.onnx come from the same model export.",
            labels.len(),
            actual_count,
            shape
        )));
    }

    Ok(())
}

/// Compute the byte range where a window's predictions are authoritative.
///
/// Uses the same symmetric overlap trimming as windowed-span: trim `overlap/2` tokens
/// from the left (except first window), `overlap - overlap/2` from the right (except last).
/// The emit region is defined by the byte offsets of the resulting usable token boundaries.
fn compute_emit_region(
    text: &str,
    encoding: &tokenizers::Encoding,
    window_index: usize,
    window_count: usize,
    overlap_tokens: usize,
) -> (usize, usize) {
    let offsets = encoding.get_offsets();
    let special_mask = encoding.get_special_tokens_mask();

    let usable: Vec<(usize, usize)> = offsets
        .iter()
        .zip(special_mask.iter())
        .filter_map(|(&(start, end), &special)| {
            if special == 0 && end > start {
                Some((start, end))
            } else {
                None
            }
        })
        .collect();

    if usable.is_empty() {
        return (0, text.len());
    }

    let left_trim = if window_index == 0 {
        0
    } else {
        overlap_tokens / 2
    };
    let right_trim = if window_index + 1 == window_count {
        0
    } else {
        overlap_tokens - overlap_tokens / 2
    };

    let emit_start_idx = left_trim.min(usable.len().saturating_sub(1));
    let emit_end_idx = usable
        .len()
        .saturating_sub(right_trim)
        .max(emit_start_idx + 1)
        .min(usable.len());

    let emit_start = if window_index == 0 {
        0
    } else {
        usable[emit_start_idx].0
    };
    let emit_end = if window_index + 1 == window_count {
        text.len()
    } else {
        usable[emit_end_idx - 1].1
    };

    (emit_start, emit_end)
}

fn ensure_sequence_within_limit(
    profile_name: &str,
    seq_len: usize,
    max_tokens: usize,
) -> Result<(), TiktagError> {
    if seq_len > max_tokens {
        return Err(TiktagError::SequenceTooLong {
            profile: profile_name.to_owned(),
            seq_len,
            max_tokens,
        });
    }

    Ok(())
}

fn ort_error<E>(err: ort::Error<E>) -> TiktagError {
    TiktagError::OrtRuntime(err.to_string())
}

#[cfg(test)]
mod tests {
    use ort::value::{Outlet, Shape, SymbolicDimensions, TensorElementType, ValueType};

    use super::{
        ensure_sequence_within_limit, validate_logits_output_metadata,
        validate_runtime_logits_shape,
    };

    fn labels(count: usize) -> Vec<String> {
        (0..count).map(|index| format!("LABEL_{index}")).collect()
    }

    fn tensor_output(name: &str, shape: Vec<i64>) -> Outlet {
        Outlet::new(
            name,
            ValueType::Tensor {
                ty: TensorElementType::Float32,
                shape: Shape::new(shape.iter().copied()),
                dimension_symbols: SymbolicDimensions::empty(shape.len()),
            },
        )
    }

    #[test]
    fn validates_matching_declared_label_count() {
        let outputs = vec![tensor_output("logits", vec![1, -1, 9])];

        let output_name =
            validate_logits_output_metadata("distilbert_ner_hrl", &labels(9), &outputs)
                .expect("matching schema should validate");

        assert_eq!(output_name, "logits");
    }

    #[test]
    fn rejects_declared_label_count_mismatch() {
        let outputs = vec![tensor_output("logits", vec![1, -1, 8])];

        let err = validate_logits_output_metadata("distilbert_ner_hrl", &labels(9), &outputs)
            .expect_err("mismatch should fail");

        assert!(err.to_string().contains("label schema mismatch"));
    }

    #[test]
    fn skips_declared_count_check_for_dynamic_label_dimension() {
        let outputs = vec![tensor_output("logits", vec![1, -1, -1])];

        validate_logits_output_metadata("distilbert_ner_hrl", &labels(9), &outputs)
            .expect("dynamic label dimension should be allowed");
    }

    #[test]
    fn rejects_missing_outputs() {
        let err = validate_logits_output_metadata("distilbert_ner_hrl", &labels(9), &[])
            .expect_err("missing outputs should fail");

        assert!(err.to_string().contains("has no outputs"));
    }

    #[test]
    fn rejects_non_tensor_outputs() {
        let outputs = vec![Outlet::new(
            "logits",
            ValueType::Sequence(Box::new(ValueType::Tensor {
                ty: TensorElementType::Float32,
                shape: Shape::new([1]),
                dimension_symbols: SymbolicDimensions::empty(1),
            })),
        )];

        let err = validate_logits_output_metadata("distilbert_ner_hrl", &labels(9), &outputs)
            .expect_err("non-tensor outputs should fail");

        assert!(err.to_string().contains("is not a tensor"));
    }

    #[test]
    fn rejects_wrong_rank_outputs() {
        let outputs = vec![tensor_output("logits", vec![1, 9])];

        let err = validate_logits_output_metadata("distilbert_ner_hrl", &labels(9), &outputs)
            .expect_err("wrong rank should fail");

        assert!(err.to_string().contains("expected rank-3 logits"));
    }

    #[test]
    fn rejects_runtime_logits_shape_mismatch() {
        let err =
            validate_runtime_logits_shape("distilbert_ner_hrl", "logits", &labels(9), &[1, 32, 8])
                .expect_err("runtime mismatch should fail");

        assert!(err.to_string().contains("produced logits with 8 labels"));
    }

    #[test]
    fn rejects_sequence_lengths_over_profile_limit() {
        let err = ensure_sequence_within_limit("distilbert_ner_hrl", 513, 512)
            .expect_err("limit overflow should fail");

        let message = err.to_string();
        assert!(message.contains("exceeding max_tokens=512"));
        assert!(message.contains("overlap_tokens"));
    }
}
