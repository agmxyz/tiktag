# Contributing

`tiktag` accepts small, targeted improvements.

Good fits:
- bug fixes
- docs and rustdoc improvements
- tests and fixture updates tied to real behavior changes
- packaging, CI, and release polish
- small recognizer or inference improvements that preserve the current library and CLI contract

Open an issue first for:
- breaking API or JSON changes
- profile or model contract changes
- larger refactors
- broader product direction changes

## Before opening a PR

Run the local baseline:

```bash
just verify
```

Run manual checks when relevant:

```bash
just test-fixtures  # inference or fixture changes; requires downloaded assets
just bench          # performance-sensitive changes
just smoke-package  # packaging or release-path changes
```

## PR expectations

- keep changes small and directly tied to the problem
- avoid speculative features and unrelated refactors
- update docs or rustdoc when public behavior changes
- update `CHANGELOG.md` when user-visible behavior changes

## Issues

Use normal GitHub issues for bugs, ideas, and release-impacting problems. There are no issue forms or separate security intake in this repo right now.
