use std::collections::BTreeMap;
use std::fmt;
use std::net::IpAddr;
use std::str::FromStr;

use serde::Serialize;

use crate::decode::EntitySpan;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum PlaceholderFamily {
    Person,
    Email,
    Phone,
    Org,
    OrgId,
    PersonId,
    DocId,
    Account,
    BankAccount,
    Card,
    CardSecurity,
    Ip,
    Url,
    Address,
    Location,
    Dob,
    Secret,
    DeviceId,
    VehicleId,
    Handle,
}

impl PlaceholderFamily {
    fn as_str(self) -> &'static str {
        match self {
            Self::Person => "PERSON",
            Self::Email => "EMAIL",
            Self::Phone => "PHONE",
            Self::Org => "ORG",
            Self::OrgId => "ORG_ID",
            Self::PersonId => "PERSON_ID",
            Self::DocId => "DOC_ID",
            Self::Account => "ACCOUNT",
            Self::BankAccount => "BANK_ACCOUNT",
            Self::Card => "CARD",
            Self::CardSecurity => "CARD_SECURITY",
            Self::Ip => "IP",
            Self::Url => "URL",
            Self::Address => "ADDRESS",
            Self::Location => "LOCATION",
            Self::Dob => "DOB",
            Self::Secret => "SECRET",
            Self::DeviceId => "DEVICE_ID",
            Self::VehicleId => "VEHICLE_ID",
            Self::Handle => "HANDLE",
        }
    }

    fn priority_rank(self) -> u8 {
        match self {
            Self::Secret => 0,
            Self::Email => 1,
            Self::Url => 2,
            Self::Ip => 3,
            Self::BankAccount => 4,
            Self::Card => 5,
            Self::CardSecurity => 6,
            Self::PersonId => 7,
            Self::DocId => 8,
            Self::OrgId => 9,
            Self::Account => 10,
            Self::DeviceId => 11,
            Self::VehicleId => 12,
            Self::Phone => 13,
            Self::Dob => 14,
            Self::Address => 15,
            Self::Person => 16,
            Self::Org => 17,
            Self::Location => 18,
            Self::Handle => 19,
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
    match label {
        "PERSON_NAME" | "PERSON_ALIAS" | "PROPER_NAME" => Some(PlaceholderFamily::Person),
        "EMAIL_ADDRESS" => Some(PlaceholderFamily::Email),
        "PHONE_NUMBER" => Some(PlaceholderFamily::Phone),
        "ORGANIZATION_NAME" => Some(PlaceholderFamily::Org),
        "ORGANIZATION_IDENTIFIER" => Some(PlaceholderFamily::OrgId),
        "PERSON_IDENTIFIER" => Some(PlaceholderFamily::PersonId),
        "DOCUMENT_IDENTIFIER" | "DOCUMENT_REFERENCE" => Some(PlaceholderFamily::DocId),
        "ACCOUNT_IDENTIFIER" => Some(PlaceholderFamily::Account),
        "BANK_ACCOUNT_IDENTIFIER" => Some(PlaceholderFamily::BankAccount),
        "PAYMENT_CARD" => Some(PlaceholderFamily::Card),
        "PAYMENT_CARD_SECURITY" => Some(PlaceholderFamily::CardSecurity),
        "IP_ADDRESS" => Some(PlaceholderFamily::Ip),
        "IDENTIFYING_LINK" => Some(PlaceholderFamily::Url),
        "POSTAL_ADDRESS" => Some(PlaceholderFamily::Address),
        "LOCATION" | "GEO_LOCATION" => Some(PlaceholderFamily::Location),
        "DATE_OF_BIRTH" => Some(PlaceholderFamily::Dob),
        "AUTH_SECRET" => Some(PlaceholderFamily::Secret),
        "DEVICE_IDENTIFIER" => Some(PlaceholderFamily::DeviceId),
        "VEHICLE_IDENTIFIER" => Some(PlaceholderFamily::VehicleId),
        "CONTACT_HANDLE" => Some(PlaceholderFamily::Handle),
        _ => None,
    }
}

fn is_valid_candidate(family: PlaceholderFamily, text: &str) -> bool {
    if !text.chars().any(|ch| ch.is_alphanumeric()) {
        return false;
    }

    let lower = text.to_ascii_lowercase();
    let alnum_count = text.chars().filter(|ch| ch.is_alphanumeric()).count();
    let digit_count = text.chars().filter(|ch| ch.is_ascii_digit()).count();
    let letter_count = text.chars().filter(|ch| ch.is_alphabetic()).count();
    let has_uppercase = text.chars().any(|ch| ch.is_uppercase());
    let has_whitespace = text.chars().any(|ch| ch.is_whitespace());

    match family {
        PlaceholderFamily::Email => is_valid_email(text),
        PlaceholderFamily::Url => lower.starts_with("http://") || lower.starts_with("https://"),
        PlaceholderFamily::Ip => IpAddr::from_str(text).is_ok(),
        PlaceholderFamily::Phone => digit_count >= 7,
        PlaceholderFamily::BankAccount => digit_count >= 12 && text.len() >= 18,
        PlaceholderFamily::Account => digit_count >= 6 && text.len() >= 8,
        PlaceholderFamily::Card => (12..=19).contains(&digit_count) && text.len() >= 14,
        PlaceholderFamily::DocId
        | PlaceholderFamily::PersonId
        | PlaceholderFamily::OrgId
        | PlaceholderFamily::DeviceId
        | PlaceholderFamily::VehicleId => {
            alnum_count >= 6 && digit_count >= 2 && (letter_count > 0 || digit_count >= 7)
        }
        PlaceholderFamily::CardSecurity => {
            text.chars().all(|ch| ch.is_ascii_digit()) && (3..=4).contains(&text.len())
        }
        PlaceholderFamily::Address => {
            letter_count >= 5
                && digit_count > 0
                && first_alpha_is_uppercase(text)
                && !is_common_junk_token(&lower)
        }
        PlaceholderFamily::Location => {
            letter_count >= 2
                && digit_count == 0
                && first_alpha_is_uppercase(text)
                && !is_common_junk_token(&lower)
        }
        PlaceholderFamily::Person => {
            letter_count >= 2 && first_alpha_is_uppercase(text) && !is_common_junk_token(&lower)
        }
        PlaceholderFamily::Org => {
            letter_count >= 2
                && !text.contains('@')
                && !text.contains('/')
                && !(text.contains('.') && !has_whitespace)
                && (has_uppercase || has_whitespace)
                && !is_common_junk_token(&lower)
        }
        PlaceholderFamily::Dob => alnum_count >= 4 && digit_count >= 1,
        PlaceholderFamily::Secret => alnum_count >= 4,
        PlaceholderFamily::Handle => alnum_count >= 2,
    }
}

fn is_valid_email(text: &str) -> bool {
    let Some((_, domain)) = text.split_once('@') else {
        return false;
    };
    !domain.is_empty() && domain.contains('.')
}

fn is_common_junk_token(lower: &str) -> bool {
    matches!(
        lower,
        "." | "," | "example" | "http" | "https" | "www" | "com" | "org" | "net" | "es"
    )
}

fn first_alpha_is_uppercase(text: &str) -> bool {
    text.chars()
        .find(|ch| ch.is_alphabetic())
        .is_some_and(|ch| ch.is_uppercase())
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
        let text = "Laura met Laura again.";
        let result = anonymize_ok(
            text,
            &[
                span("PERSON_NAME", 0, 5, "Laura"),
                span("PERSON_NAME", 10, 15, "Laura"),
            ],
        );

        assert_eq!(result.anonymized_text, "[PERSON_1] met [PERSON_1] again.");
        assert_eq!(
            result.placeholder_map.get("[PERSON_1]"),
            Some(&"Laura".to_owned())
        );
    }

    #[test]
    fn increments_placeholders_for_distinct_values_in_same_family() {
        let text = "Laura met Lukas.";
        let result = anonymize_ok(
            text,
            &[
                span("PERSON_NAME", 0, 5, "Laura"),
                span("PERSON_NAME", 10, 15, "Lukas"),
            ],
        );

        assert_eq!(result.anonymized_text, "[PERSON_1] met [PERSON_2].");
    }

    #[test]
    fn ignores_excluded_labels() {
        let text = "Condition diabetes was mentioned.";
        let result = anonymize_ok(text, &[span("HEALTH_DATA", 10, 18, "diabetes")]);

        assert_eq!(result.anonymized_text, text);
        assert!(result.replacements.is_empty());
    }

    #[test]
    fn rejects_common_junk_tokens() {
        let text = "example .";
        let result = anonymize_ok(
            text,
            &[
                span("ORGANIZATION_NAME", 0, 7, "example"),
                span("ACCOUNT_IDENTIFIER", 8, 9, "."),
            ],
        );

        assert!(result.replacements.is_empty());
        assert_eq!(result.anonymized_text, text);
    }

    #[test]
    fn prefers_email_over_overlapping_org() {
        let text = "Reach me at laura@example.com today.";
        let result = anonymize_ok(
            text,
            &[
                span("EMAIL_ADDRESS", 12, 29, "laura@example.com"),
                span("ORGANIZATION_NAME", 18, 29, "example.com"),
            ],
        );

        assert_eq!(result.replacements.len(), 1);
        assert_eq!(result.replacements[0].family, PlaceholderFamily::Email);
    }

    #[test]
    fn prefers_url_over_overlapping_location() {
        let text = "Visit https://madrid.example/path now.";
        let result = anonymize_ok(
            text,
            &[
                span("IDENTIFYING_LINK", 6, 33, "https://madrid.example/path"),
                span("LOCATION", 14, 20, "madrid"),
            ],
        );

        assert_eq!(result.replacements.len(), 1);
        assert_eq!(result.replacements[0].family, PlaceholderFamily::Url);
    }

    #[test]
    fn prefers_address_over_overlapping_location() {
        let text = "Send mail to Calle de Alcala 147, 28009 Madrid.";
        let result = anonymize_ok(
            text,
            &[
                span(
                    "POSTAL_ADDRESS",
                    13,
                    45,
                    "Calle de Alcala 147, 28009 Madrid",
                ),
                span("LOCATION", 37, 43, "Madrid"),
            ],
        );

        assert_eq!(result.replacements.len(), 1);
        assert_eq!(result.replacements[0].family, PlaceholderFamily::Address);
    }

    #[test]
    fn prefers_bank_account_over_overlapping_person() {
        let text = "Account DE44 5001 0517 5407 3249 31 belongs to Laura.";
        let result = anonymize_ok(
            text,
            &[
                span(
                    "BANK_ACCOUNT_IDENTIFIER",
                    8,
                    35,
                    "DE44 5001 0517 5407 3249 31",
                ),
                span("PERSON_NAME", 8, 12, "DE44"),
            ],
        );

        assert_eq!(result.replacements.len(), 1);
        assert_eq!(
            result.replacements[0].family,
            PlaceholderFamily::BankAccount
        );
    }

    #[test]
    fn preserves_untouched_text_around_rewrites() {
        let text = "Hello Laura, welcome to Madrid.";
        let result = anonymize_ok(
            text,
            &[
                span("PERSON_NAME", 6, 11, "Laura"),
                span("LOCATION", 24, 30, "Madrid"),
            ],
        );

        assert_eq!(
            result.anonymized_text,
            "Hello [PERSON_1], welcome to [LOCATION_1]."
        );
    }

    #[test]
    fn prefers_full_address_over_fragment_in_same_cluster() {
        let text = "Send mail to Calle de Alcala 147, 28009 Madrid.";
        let result = anonymize_ok(
            text,
            &[
                span("POSTAL_ADDRESS", 13, 18, "Calle"),
                span(
                    "POSTAL_ADDRESS",
                    13,
                    45,
                    "Calle de Alcala 147, 28009 Madrid",
                ),
            ],
        );

        assert_eq!(result.replacements.len(), 1);
        assert_eq!(
            result.replacements[0].original,
            "Calle de Alcala 147, 28009 Madrid"
        );
    }

    #[test]
    fn rejects_lowercase_domain_like_org_fragments() {
        let text = "portal atlas-soluciones.es";
        let result = anonymize_ok(
            text,
            &[
                span("ORGANIZATION_NAME", 0, 6, "portal"),
                span("ORGANIZATION_NAME", 7, 26, "atlas-soluciones.es"),
            ],
        );

        assert!(result.replacements.is_empty());
        assert_eq!(result.anonymized_text, text);
    }

    #[test]
    fn rejects_short_numeric_account_fragments() {
        let text = "Codes 2100 and 0418 4502.";
        let result = anonymize_ok(
            text,
            &[
                span("ACCOUNT_IDENTIFIER", 6, 10, "2100"),
                span("CARD", 15, 24, "0418 4502"),
            ],
        );

        assert!(result.replacements.is_empty());
    }
}
