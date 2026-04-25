use std::borrow::Cow;
use std::sync::LazyLock;

use log::debug;
use regex::Regex;

use crate::decode::EntitySpan;

const EMAIL_LABEL: &str = "EMAIL_ADDRESS";
const EMAIL_SCORE: f32 = 0.95;

static EMAIL_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\b[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*@[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)+\b",
    )
    .expect("valid email regex")
});

pub(crate) fn detect(text: &str) -> Vec<EntitySpan> {
    let results: Vec<_> = EMAIL_REGEX
        .find_iter(text)
        .filter_map(|mat| {
            let email = mat.as_str();

            if is_valid_email_domain(email) {
                Some(EntitySpan {
                    label: Cow::Borrowed(EMAIL_LABEL),
                    start: mat.start(),
                    end: mat.end(),
                    text: email.to_owned(),
                    score: EMAIL_SCORE,
                })
            } else {
                None
            }
        })
        .collect();

    if !results.is_empty() {
        debug!("Regex emails: {} found", results.len());
    }

    results
}

fn is_valid_email_domain(email: &str) -> bool {
    let Some((_, domain)) = email.rsplit_once('@') else {
        return false;
    };

    if domain.len() > 253 || !domain.contains('.') {
        return false;
    }

    let labels: Vec<&str> = domain.split('.').collect();

    if labels
        .iter()
        .any(|label| label.is_empty() || label.len() > 63)
    {
        return false;
    }

    for label in &labels {
        let bytes = label.as_bytes();

        if !bytes[0].is_ascii_alphanumeric() || !bytes[bytes.len() - 1].is_ascii_alphanumeric() {
            return false;
        }

        if !bytes
            .iter()
            .all(|b| b.is_ascii_alphanumeric() || *b == b'-')
        {
            return false;
        }
    }

    let tld = labels
        .last()
        .expect("labels always has at least one element after split");

    tld.len() >= 2 && tld.as_bytes().iter().all(|b| b.is_ascii_alphabetic())
}

#[cfg(test)]
mod tests {
    use super::detect;

    #[test]
    fn detects_basic_email_address() {
        let text = "Contact us at team.lead+ops@example-domain.com now.";
        let entities = detect(text);

        assert_eq!(entities.len(), 1);
        assert_eq!(
            &text[entities[0].start..entities[0].end],
            "team.lead+ops@example-domain.com"
        );
        assert_eq!(entities[0].label.as_ref(), "EMAIL_ADDRESS");
    }

    #[test]
    fn filters_out_invalid_tld() {
        let text = "Bad: user@example.123";
        let entities = detect(text);

        assert!(entities.is_empty());
    }

    #[test]
    fn uses_byte_offsets_with_utf8_prefix() {
        let text = "Željko kontakt: sara@example.com i više.";
        let entities = detect(text);

        assert_eq!(entities.len(), 1);
        let start = entities[0].start;
        let end = entities[0].end;
        assert_eq!(&text[start..end], "sara@example.com");
    }
}
