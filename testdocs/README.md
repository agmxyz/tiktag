# Test Fixtures

`testdocs/` now has two kinds of realistic synthetic inputs for `eu_pii`.

Automated regression fixtures:

- `eu_pii_windowed_*`: medium-size privacy incident dossier that should force multi-window inference and verify stable placeholder reuse.
- `eu_pii_stress_windowed_*`: larger incident bundle that pushes the same repeated values across deep sliding-window inference.

These expected manifests focus on anonymization behavior:

- anonymized text contains expected placeholders
- repeated exact values reuse the same placeholder
- source literals used in the stable fixture set do not remain in anonymized output
- long inputs still produce multiple windows and stable total timing metadata

Exploratory manual probes:

- `eu_pii_hard_probe_input.md`
- `eu_pii_hard_stress_input.md`

The hard probes intentionally include harder families such as emails, URLs, IPs, IDs, addresses, and bank data. They are useful for characterizing current model behavior, but they are not treated as stable pass/fail regression fixtures yet.
