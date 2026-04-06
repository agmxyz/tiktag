use std::fmt;

use ndarray::ArrayView3;
use serde::{Deserialize, Serialize};
use tokenizers::Encoding;

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct EntitySpan {
    pub label: String,
    pub start: usize,
    pub end: usize,
    pub text: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DecodeStrategy {
    GenericBio,
    PiiRelaxed,
}

impl Default for DecodeStrategy {
    fn default() -> Self {
        Self::GenericBio
    }
}

impl fmt::Display for DecodeStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let value = match self {
            Self::GenericBio => "generic_bio",
            Self::PiiRelaxed => "pii_relaxed",
        };
        f.write_str(value)
    }
}

pub fn argmax_label_indices(logits: ArrayView3<'_, f32>) -> Vec<usize> {
    let seq_len = logits.shape()[1];
    let label_count = logits.shape()[2];
    let mut predictions = Vec::with_capacity(seq_len);

    for token_index in 0..seq_len {
        let mut best_label = 0usize;
        let mut best_score = f32::NEG_INFINITY;

        for label_index in 0..label_count {
            let score = logits[(0, token_index, label_index)];
            if score > best_score {
                best_score = score;
                best_label = label_index;
            }
        }

        predictions.push(best_label);
    }

    predictions
}

pub fn decode_entities(
    text: &str,
    encoding: &Encoding,
    predictions: &[usize],
    labels: &[String],
    strategy: DecodeStrategy,
) -> Vec<EntitySpan> {
    let special_mask = encoding.get_special_tokens_mask();
    let offsets = encoding.get_offsets();
    let upper = predictions.len().min(offsets.len()).min(special_mask.len());

    let mut entities = Vec::new();
    let mut current: Option<EntitySpan> = None;

    for index in 0..upper {
        if special_mask[index] != 0 {
            flush_entity(&mut current, &mut entities);
            continue;
        }

        let (start, end) = offsets[index];
        if start >= end || end > text.len() || text.get(start..end).is_none() {
            flush_entity(&mut current, &mut entities);
            continue;
        }

        let label = labels
            .get(predictions[index])
            .map(String::as_str)
            .unwrap_or("O");

        if label == "O" {
            flush_entity(&mut current, &mut entities);
            continue;
        }

        let (prefix, kind) = split_label(label);
        if kind == "O" {
            flush_entity(&mut current, &mut entities);
            continue;
        }

        let merge_label = current
            .as_ref()
            .and_then(|entity| {
                let gap_start = entity.end.min(start);
                let gap = text.get(gap_start..start).unwrap_or("");
                merged_label(strategy, entity.label.as_str(), kind, prefix, gap)
            })
            .map(str::to_owned);

        if current.is_none() || merge_label.is_none() {
            flush_entity(&mut current, &mut entities);
            current = Some(EntitySpan {
                label: kind.to_owned(),
                start,
                end,
                text: text.get(start..end).unwrap_or("").to_owned(),
            });
            continue;
        }

        if let Some(entity) = current.as_mut() {
            if let Some(merged) = merge_label {
                entity.label = merged;
            }
            entity.end = end;
            if let Some(span_text) = text.get(entity.start..entity.end) {
                entity.text = span_text.to_owned();
            }
        }
    }

    flush_entity(&mut current, &mut entities);
    entities
}

fn split_label(label: &str) -> (&str, &str) {
    match label.split_once('-') {
        Some((prefix, kind)) if !kind.is_empty() => (prefix, kind),
        _ => ("B", label),
    }
}

fn flush_entity(current: &mut Option<EntitySpan>, entities: &mut Vec<EntitySpan>) {
    if let Some(entity) = current.take() {
        entities.push(entity);
    }
}

fn merged_label<'a>(
    strategy: DecodeStrategy,
    current: &'a str,
    next: &'a str,
    prefix: &str,
    gap: &str,
) -> Option<&'a str> {
    match strategy {
        DecodeStrategy::GenericBio => merged_label_generic(current, next, prefix, gap),
        DecodeStrategy::PiiRelaxed => merged_label_pii_relaxed(current, next, gap),
    }
}

fn merged_label_generic<'a>(
    current: &'a str,
    next: &'a str,
    prefix: &str,
    gap: &str,
) -> Option<&'a str> {
    if current == next && prefix == "I" && is_generic_gap(gap) {
        return Some(current);
    }
    None
}

fn merged_label_pii_relaxed<'a>(current: &'a str, next: &'a str, gap: &str) -> Option<&'a str> {
    if current == next {
        let can_merge = match current {
            "EMAIL_ADDRESS" => gap.is_empty() || gap.chars().all(|ch| "._-+@".contains(ch)),
            "PHONE_NUMBER" => gap
                .chars()
                .all(|ch| ch.is_whitespace() || "+-()./".contains(ch)),
            _ => is_generic_gap(gap),
        };

        if can_merge {
            return Some(current);
        }
    }

    if current == "EMAIL_ADDRESS" && next == "ORGANIZATION_NAME" && gap.is_empty() {
        return Some("EMAIL_ADDRESS");
    }

    None
}

fn is_generic_gap(gap: &str) -> bool {
    gap.is_empty() || gap.chars().all(char::is_whitespace)
}

#[cfg(test)]
mod tests {
    use tokenizers::Encoding;

    use super::{DecodeStrategy, EntitySpan, decode_entities};

    fn test_encoding(tokens: &[&str], offsets: &[(usize, usize)]) -> Encoding {
        let len = tokens.len();
        Encoding::new(
            vec![0; len],
            vec![0; len],
            tokens.iter().map(|token| (*token).to_owned()).collect(),
            vec![None; len],
            offsets.to_vec(),
            vec![0; len],
            vec![1; len],
            vec![],
            Default::default(),
        )
    }

    fn span(label: &str, start: usize, end: usize, text: &str) -> EntitySpan {
        EntitySpan {
            label: label.to_owned(),
            start,
            end,
            text: text.to_owned(),
        }
    }

    #[test]
    fn merges_generic_bio_spans() {
        let text = "John Doe";
        let encoding = test_encoding(&["John", "Doe"], &[(0, 4), (5, 8)]);
        let predictions = vec![1, 2];
        let labels = vec!["O", "B-PER", "I-PER"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();

        let entities = decode_entities(
            text,
            &encoding,
            &predictions,
            &labels,
            DecodeStrategy::GenericBio,
        );

        assert_eq!(entities, vec![span("PER", 0, 8, "John Doe")]);
    }

    #[test]
    fn merges_relaxed_email_tokens() {
        let text = "alice@example.com";
        let encoding = test_encoding(&["alice", "@example", ".com"], &[(0, 5), (5, 13), (13, 17)]);
        let predictions = vec![1, 2, 2];
        let labels = vec!["O", "B-EMAIL_ADDRESS", "I-EMAIL_ADDRESS"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();

        let entities = decode_entities(
            text,
            &encoding,
            &predictions,
            &labels,
            DecodeStrategy::PiiRelaxed,
        );

        assert_eq!(
            entities,
            vec![span("EMAIL_ADDRESS", 0, 17, "alice@example.com")]
        );
    }

    #[test]
    fn merges_relaxed_phone_tokens() {
        let text = "+39 347 123 4567";
        let encoding = test_encoding(
            &["+39", "347", "123", "4567"],
            &[(0, 3), (4, 7), (8, 11), (12, 16)],
        );
        let predictions = vec![1, 2, 2, 2];
        let labels = vec!["O", "B-PHONE_NUMBER", "I-PHONE_NUMBER"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();

        let entities = decode_entities(
            text,
            &encoding,
            &predictions,
            &labels,
            DecodeStrategy::PiiRelaxed,
        );

        assert_eq!(
            entities,
            vec![span("PHONE_NUMBER", 0, 16, "+39 347 123 4567")]
        );
    }

    #[test]
    fn does_not_merge_incompatible_labels() {
        let text = "John Paris";
        let encoding = test_encoding(&["John", "Paris"], &[(0, 4), (5, 10)]);
        let predictions = vec![1, 2];
        let labels = vec!["O", "B-PER", "I-LOC"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();

        let entities = decode_entities(
            text,
            &encoding,
            &predictions,
            &labels,
            DecodeStrategy::GenericBio,
        );

        assert_eq!(
            entities,
            vec![span("PER", 0, 4, "John"), span("LOC", 5, 10, "Paris")]
        );
    }
}
