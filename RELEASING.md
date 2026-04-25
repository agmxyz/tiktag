# Releasing

Manual release flow for `tiktag`.

## Checklist

1. Update the version in `Cargo.toml` and refresh `Cargo.lock` if needed.
2. Add or finalize the matching entry in `CHANGELOG.md`.
3. Run the baseline checks:

   ```bash
   cargo fmt --check
   just verify
   cargo doc --no-deps
   cargo package
   ```

4. Run extra manual checks when relevant:

   ```bash
   just test-fixtures  # inference or fixture changes; requires downloaded assets
   just bench          # performance-sensitive changes
   just smoke-package  # packaging changes
   ```

5. Commit the release changes and tag `vX.Y.Z`.
6. Push the commit and tag to GitHub.
7. Publish to crates.io:

   ```bash
   cargo publish
   ```

8. Create the GitHub Release from the matching `CHANGELOG.md` entry.
9. Verify the public surfaces after publish:
   - crates.io page shows the new version
   - docs.rs built the new crate docs
   - GitHub release notes match the changelog summary
