// Post-processing: turns raw ONNX logits into merged entity spans for the
// built-in Xenova DistilBERT NER model. Merge rules are strict BIO only.

use ndarray::ArrayView3;
use serde::Serialize;
use tokenizers::Encoding;

/// A detected entity: label (e.g. "PER"), byte offsets into the original text, and the text itself.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct EntitySpan {
    pub label: String,
    pub start: usize,
    pub end: usize,
    pub text: String,
}

/// Pick the highest-scoring label for each token position. Returns label indices.
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

/// Walk token predictions and merge adjacent tokens into entity spans.
/// Skips special tokens ([CLS]/[SEP]), invalid offsets, and non-BIO labels.
pub fn decode_entities(
    text: &str,
    encoding: &Encoding,
    predictions: &[usize],
    labels: &[String],
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

        let Some((prefix, kind)) = split_bio_label(label) else {
            flush_entity(&mut current, &mut entities);
            continue;
        };

        let merge_label = current
            .as_ref()
            .and_then(|entity| {
                let gap_start = entity.end.min(start);
                let gap = text.get(gap_start..start).unwrap_or("");
                merged_label(entity.label.as_str(), kind, prefix, gap)
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

/// Splits a strict BIO label like "B-PER" into ("B", "PER").
/// Any bare or malformed label is ignored by decode_entities.
fn split_bio_label(label: &str) -> Option<(&str, &str)> {
    let (prefix, kind) = label.split_once('-')?;
    if kind.is_empty() {
        return None;
    }
    if prefix == "B" || prefix == "I" {
        return Some((prefix, kind));
    }
    None
}

fn flush_entity(current: &mut Option<EntitySpan>, entities: &mut Vec<EntitySpan>) {
    if let Some(entity) = current.take() {
        entities.push(entity);
    }
}

/// Decide if the next token should merge into the current entity span.
/// Returns Some(label) if yes, None if token starts a new span.
fn merged_label<'a>(current: &'a str, next: &'a str, prefix: &str, gap: &str) -> Option<&'a str> {
    if current != next {
        return None;
    }

    if gap.is_empty() || (prefix == "I" && is_generic_gap(gap)) {
        return Some(current);
    }

    None
}

fn is_generic_gap(gap: &str) -> bool {
    gap.is_empty() || gap.chars().all(char::is_whitespace)
}

#[cfg(test)]
mod tests {
    use tokenizers::Encoding;

    use super::{EntitySpan, decode_entities};

    fn test_encoding(tokens: &[&str], offsets: &[(usize, usize)]) -> Encoding {
        test_encoding_with_mask(tokens, offsets, &vec![0; tokens.len()])
    }

    fn test_encoding_with_mask(
        tokens: &[&str],
        offsets: &[(usize, usize)],
        special_tokens_mask: &[u32],
    ) -> Encoding {
        let len = tokens.len();
        Encoding::new(
            vec![0; len],
            vec![0; len],
            tokens.iter().map(|token| (*token).to_owned()).collect(),
            vec![None; len],
            offsets.to_vec(),
            special_tokens_mask.to_vec(),
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
    fn merges_bio_prefixed_spans() {
        let text = "John Doe";
        let encoding = test_encoding(&["John", "Doe"], &[(0, 4), (5, 8)]);
        let predictions = vec![1, 2];
        let labels = vec!["O", "B-PER", "I-PER"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();

        let entities = decode_entities(text, &encoding, &predictions, &labels);

        assert_eq!(entities, vec![span("PER", 0, 8, "John Doe")]);
    }

    #[test]
    fn merges_zero_gap_subword_fragments() {
        let text = "Microsoft";
        let encoding = test_encoding(&["Micro", "soft"], &[(0, 5), (5, 9)]);
        let predictions = vec![1, 2];
        let labels = vec!["O", "B-ORG", "I-ORG"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();

        let entities = decode_entities(text, &encoding, &predictions, &labels);

        assert_eq!(entities, vec![span("ORG", 0, 9, "Microsoft")]);
    }

    #[test]
    fn merges_org_bio_continuation_with_whitespace() {
        let text = "Open AI";
        let encoding = test_encoding(&["Open", "AI"], &[(0, 4), (5, 7)]);
        let predictions = vec![1, 2];
        let labels = vec!["O", "B-ORG", "I-ORG"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();

        let entities = decode_entities(text, &encoding, &predictions, &labels);

        assert_eq!(entities, vec![span("ORG", 0, 7, "Open AI")]);
    }

    #[test]
    fn merges_loc_bio_continuation_with_whitespace() {
        let text = "New York";
        let encoding = test_encoding(&["New", "York"], &[(0, 3), (4, 8)]);
        let predictions = vec![1, 2];
        let labels = vec!["O", "B-LOC", "I-LOC"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();

        let entities = decode_entities(text, &encoding, &predictions, &labels);

        assert_eq!(entities, vec![span("LOC", 0, 8, "New York")]);
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

        let entities = decode_entities(text, &encoding, &predictions, &labels);

        assert_eq!(
            entities,
            vec![span("PER", 0, 4, "John"), span("LOC", 5, 10, "Paris")]
        );
    }

    #[test]
    fn skips_special_tokens() {
        let text = "John";
        let encoding = test_encoding_with_mask(
            &["[CLS]", "John", "[SEP]"],
            &[(0, 0), (0, 4), (0, 0)],
            &[1, 0, 1],
        );
        let predictions = vec![1, 1, 1];
        let labels = vec!["O", "B-PER"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();

        let entities = decode_entities(text, &encoding, &predictions, &labels);

        assert_eq!(entities, vec![span("PER", 0, 4, "John")]);
    }

    #[test]
    fn ignores_bare_labels_without_bio_prefix() {
        let text = "alice";
        let encoding = test_encoding(&["alice"], &[(0, 5)]);
        let predictions = vec![1];
        let labels = vec!["O", "PER"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();

        let entities = decode_entities(text, &encoding, &predictions, &labels);

        assert!(entities.is_empty());
    }
}
