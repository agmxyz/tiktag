# Contributing

Small, targeted PRs are welcome.

Good fits:
- bug fixes
- new recognizers that preserve the current library, CLI, and JSON contract
- docs, rustdoc, and tests tied to real behavior changes

Before opening a PR:

```bash
just verify
```

Run these when relevant:

```bash
just test-fixtures  # inference or fixture changes; requires downloaded assets
just bench          # performance-sensitive changes
```

For larger changes, breaking changes, or model/profile contract changes, open an issue first.
