# AGENTS.md

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:

- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a staff engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:

- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:

- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:

```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## Living document policy

This file is shared project memory for future sessions.

Keep only high-signal information:

- current product contract and invariants
- model-specific behavior that changes implementation decisions
- proven defaults and operational shortcuts
- recurring footguns worth preventing

Keep it short. Replace stale guidance instead of accumulating history.

## Project intent

- `tiktag` is an anonymization node.
- It receives text input and outputs anonymized text.
- It ships a single built-in model: @models/profiles.toml.

## Node contract

- Canonical input is text passed directly or through `--stdin`.
- Prefer `--stdin` for large documents and pipeline use.
- The CLI does not expose model selection or profile overrides.
- Default stdout is anonymized text only.
- `--json` is optional machine/debug output with `anonymized_text`, `replacements`, `placeholder_map`, and `stats`.
- Diagnostics and total timing logs go to stderr through logging.
- Placeholder assignment is stable within one document only; there is no cross-document identity.
