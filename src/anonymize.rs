use std::collections::BTreeMap;
use std::fmt;

use serde::Serialize;

use crate::decode::EntitySpan;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum PlaceholderFamily {
    Person,
    Org,
    Location,
}

impl PlaceholderFamily {
    fn as_str(self) -> &'static str {
        match self {
            Self::Person => "PERSON",
            Self::Org => "ORG",
            Self::Location => "LOCATION",
        }
    }

    fn priority_rank(self) -> u8 {
        match self {
            Self::Person => 0,
            Self::Org => 1,
            Self::Location => 2,
        }
    }
}

impl fmt::Display for PlaceholderFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Replacement {
    pub start: usize,
    pub end: usize,
    pub family: PlaceholderFamily,
    pub placeholder: String,
    pub original: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AnonymizationResult {
    pub anonymized_text: String,
    pub replacements: Vec<Replacement>,
    pub placeholder_map: BTreeMap<String, String>,
    pub counts_by_family: BTreeMap<String, usize>,
    pub detected_entity_count: usize,
    pub accepted_replacement_count: usize,
}

#[derive(Debug, Clone)]
struct ReplacementCandidate {
    start: usize,
    end: usize,
    original: String,
    normalized: String,
    family: PlaceholderFamily,
}

pub fn anonymize(text: &str, entities: &[EntitySpan]) -> anyhow::Result<AnonymizationResult> {
    let detected_entity_count = entities.len();
    let mut candidates = entities
        .iter()
        .filter_map(build_candidate)
        .collect::<Vec<_>>();
    candidates.sort_by(|a, b| {
        a.start
            .cmp(&b.start)
            .then(a.family.priority_rank().cmp(&b.family.priority_rank()))
            .then((b.end - b.start).cmp(&(a.end - a.start)))
            .then(a.family.as_str().cmp(b.family.as_str()))
    });

    let accepted = resolve_overlaps(candidates);
    let replacements = assign_placeholders(accepted);
    let anonymized_text = rewrite_text(text, &replacements);
    let counts_by_family = count_replacements_by_family(&replacements);
    let placeholder_map = replacements
        .iter()
        .fold(BTreeMap::new(), |mut map, replacement| {
            map.entry(replacement.placeholder.clone())
                .or_insert_with(|| replacement.original.clone());
            map
        });

    Ok(AnonymizationResult {
        anonymized_text,
        accepted_replacement_count: replacements.len(),
        detected_entity_count,
        counts_by_family,
        placeholder_map,
        replacements,
    })
}

fn build_candidate(entity: &EntitySpan) -> Option<ReplacementCandidate> {
    let family = map_label_to_family(entity.label.as_str())?;
    let normalized = entity.text.trim();
    if normalized.is_empty() || !is_valid_candidate(family, normalized) {
        return None;
    }

    Some(ReplacementCandidate {
        start: entity.start,
        end: entity.end,
        original: entity.text.clone(),
        normalized: normalized.to_owned(),
        family,
    })
}

fn resolve_overlaps(candidates: Vec<ReplacementCandidate>) -> Vec<ReplacementCandidate> {
    if candidates.is_empty() {
        return Vec::new();
    }

    let mut sorted = candidates;
    sorted.sort_by(|a, b| {
        a.start
            .cmp(&b.start)
            .then(a.end.cmp(&b.end))
            .then(a.family.priority_rank().cmp(&b.family.priority_rank()))
            .then((b.end - b.start).cmp(&(a.end - a.start)))
    });

    let mut accepted = Vec::new();
    let mut cluster_start = 0usize;

    while cluster_start < sorted.len() {
        let mut cluster_end_offset = cluster_start + 1;
        let mut cluster_end = sorted[cluster_start].end;

        while cluster_end_offset < sorted.len() && sorted[cluster_end_offset].start < cluster_end {
            cluster_end = cluster_end.max(sorted[cluster_end_offset].end);
            cluster_end_offset += 1;
        }

        let winner = sorted[cluster_start..cluster_end_offset]
            .iter()
            .min_by(|a, b| {
                a.family
                    .priority_rank()
                    .cmp(&b.family.priority_rank())
                    .then((b.end - b.start).cmp(&(a.end - a.start)))
                    .then(a.start.cmp(&b.start))
            })
            .expect("overlap cluster should contain at least one candidate")
            .clone();
        accepted.push(winner);
        cluster_start = cluster_end_offset;
    }

    accepted
}

fn assign_placeholders(candidates: Vec<ReplacementCandidate>) -> Vec<Replacement> {
    let mut next_indices = BTreeMap::<PlaceholderFamily, usize>::new();
    let mut seen = BTreeMap::<(PlaceholderFamily, String), String>::new();
    let mut replacements = Vec::with_capacity(candidates.len());

    for candidate in candidates {
        let key = (candidate.family, candidate.normalized.clone());
        let placeholder = if let Some(existing) = seen.get(&key) {
            existing.clone()
        } else {
            let next = next_indices
                .entry(candidate.family)
                .and_modify(|index| *index += 1)
                .or_insert(1);
            let placeholder = format!("[{}_{}]", candidate.family, next);
            seen.insert(key, placeholder.clone());
            placeholder
        };

        replacements.push(Replacement {
            start: candidate.start,
            end: candidate.end,
            family: candidate.family,
            placeholder,
            original: candidate.original,
        });
    }

    replacements
}

fn rewrite_text(text: &str, replacements: &[Replacement]) -> String {
    if replacements.is_empty() {
        return text.to_owned();
    }

    let mut rewritten = String::with_capacity(text.len());
    let mut cursor = 0usize;

    for replacement in replacements {
        rewritten.push_str(&text[cursor..replacement.start]);
        rewritten.push_str(&replacement.placeholder);
        cursor = replacement.end;
    }

    rewritten.push_str(&text[cursor..]);
    rewritten
}

fn count_replacements_by_family(replacements: &[Replacement]) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for replacement in replacements {
        *counts
            .entry(replacement.family.to_string())
            .or_insert(0usize) += 1;
    }
    counts
}

fn map_label_to_family(label: &str) -> Option<PlaceholderFamily> {
    // DATE intentionally dropped for Xenova baseline due unstable detection.
    match label {
        "PER" => Some(PlaceholderFamily::Person),
        "ORG" => Some(PlaceholderFamily::Org),
        "LOC" => Some(PlaceholderFamily::Location),
        _ => None,
    }
}

fn is_valid_candidate(family: PlaceholderFamily, text: &str) -> bool {
    if !text.chars().any(|ch| ch.is_alphanumeric()) {
        return false;
    }

    let lower = text.to_ascii_lowercase();
    let letter_count = text.chars().filter(|ch| ch.is_alphabetic()).count();
    let digit_count = text.chars().filter(|ch| ch.is_ascii_digit()).count();

    match family {
        PlaceholderFamily::Person => letter_count >= 2 && !is_common_junk_token(&lower),
        PlaceholderFamily::Org => {
            let looks_like_email = text.contains('@');
            let looks_like_domain = text.contains('.') && text == lower;
            !looks_like_email
                && !looks_like_domain
                && letter_count >= 2
                && !is_common_junk_token(&lower)
        }
        PlaceholderFamily::Location => {
            letter_count >= 2 && digit_count == 0 && !is_common_junk_token(&lower)
        }
    }
}

fn is_common_junk_token(lower: &str) -> bool {
    matches!(
        lower,
        "." | "," | "example" | "http" | "https" | "www" | "com" | "org" | "net" | "es"
    )
}

#[cfg(test)]
mod tests {
    use super::{AnonymizationResult, PlaceholderFamily, anonymize};
    use crate::decode::EntitySpan;

    fn span(label: &str, start: usize, end: usize, text: &str) -> EntitySpan {
        EntitySpan {
            label: label.to_owned(),
            start,
            end,
            text: text.to_owned(),
        }
    }

    fn anonymize_ok(text: &str, entities: &[EntitySpan]) -> AnonymizationResult {
        anonymize(text, entities).expect("anonymize")
    }

    #[test]
    fn reuses_exact_repeated_values() {
        let text = "Satya met Satya again.";
        let result = anonymize_ok(
            text,
            &[span("PER", 0, 5, "Satya"), span("PER", 10, 15, "Satya")],
        );

        assert_eq!(result.anonymized_text, "[PERSON_1] met [PERSON_1] again.");
        assert_eq!(
            result.placeholder_map.get("[PERSON_1]"),
            Some(&"Satya".to_owned())
        );
    }

    #[test]
    fn increments_placeholders_for_distinct_values_in_same_family() {
        let text = "Satya met Sundar.";
        let result = anonymize_ok(
            text,
            &[span("PER", 0, 5, "Satya"), span("PER", 10, 16, "Sundar")],
        );

        assert_eq!(result.anonymized_text, "[PERSON_1] met [PERSON_2].");
    }

    #[test]
    fn ignores_excluded_labels() {
        let text = "Condition diabetes was mentioned.";
        let result = anonymize_ok(text, &[span("MISC", 10, 18, "diabetes")]);

        assert_eq!(result.anonymized_text, text);
        assert!(result.replacements.is_empty());
    }

    #[test]
    fn rejects_common_junk_tokens() {
        let text = "example .";
        let result = anonymize_ok(
            text,
            &[span("ORG", 0, 7, "example"), span("LOC", 8, 9, ".")],
        );

        assert!(result.replacements.is_empty());
        assert_eq!(result.anonymized_text, text);
    }

    #[test]
    fn preserves_untouched_text_around_rewrites() {
        let text = "Hello Satya, welcome to London.";
        let result = anonymize_ok(
            text,
            &[span("PER", 6, 11, "Satya"), span("LOC", 24, 30, "London")],
        );

        assert_eq!(
            result.anonymized_text,
            "Hello [PERSON_1], welcome to [LOCATION_1]."
        );
    }

    #[test]
    fn prefers_full_span_over_fragment_in_same_cluster() {
        let text = "Satya Nadella arrived.";
        let result = anonymize_ok(
            text,
            &[
                span("PER", 0, 5, "Satya"),
                span("PER", 0, 13, "Satya Nadella"),
            ],
        );

        assert_eq!(result.replacements.len(), 1);
        assert_eq!(result.replacements[0].original, "Satya Nadella");
    }

    #[test]
    fn rejects_lowercase_domain_like_org_fragments() {
        let text = "atlas-soluciones.es";
        let result = anonymize_ok(text, &[span("ORG", 0, 19, "atlas-soluciones.es")]);

        assert!(result.replacements.is_empty());
        assert_eq!(result.anonymized_text, text);
    }

    #[test]
    fn accepts_standard_per_label_without_uppercase() {
        let text = "maria arrived.";
        let result = anonymize_ok(text, &[span("PER", 0, 5, "maria")]);

        assert_eq!(result.anonymized_text, "[PERSON_1] arrived.");
    }

    #[test]
    fn accepts_multilingual_location_label() {
        let text = "\u{00E0}rea central";
        let result = anonymize_ok(text, &[span("LOC", 0, 5, "\u{00E0}rea")]);

        assert_eq!(result.anonymized_text, "[LOCATION_1] central");
        assert_eq!(result.replacements[0].family, PlaceholderFamily::Location);
    }
}
