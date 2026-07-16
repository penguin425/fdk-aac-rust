# Release process

This project follows the original fdk-aac practice of preparing a version and
changelog commit, validating it, and creating an annotated `rust-vX.Y.Z` Git
tag. The `rust-` prefix distinguishes this port from the inherited upstream
`v0.1.x` and `v2.0.x` tags.
Distribution is adapted for Rust: crates are published to crates.io and the
same source packages are attached to a GitHub Release. Compiled binary artifacts
are intentionally excluded; adding any requires satisfying the separate
[`DISTRIBUTION.md`](DISTRIBUTION.md) binary-distribution checklist.

## One-time repository setup

1. Create or claim the `fdk-aac-rust-sys` and `fdk-aac-rust` crates under the same
   crates.io owner or team.
2. Add a crates.io API token as the GitHub Actions secret
   `CARGO_REGISTRY_TOKEN`.
3. Allow GitHub Actions read/write access to repository contents so the
   workflow can create a GitHub Release. The workflow also declares the
   required `contents: write` permission.
4. Protect the release branch and require the CI workflow before merging.

The two crates always use the same version. `fdk-aac-rust-sys` must be published
first because `fdk-aac-rust` references that exact compatible version when its
default `ffi` feature is enabled.

## Preparing a release

1. Choose a semantic version.
2. Update `workspace.package.version` in the root `Cargo.toml`.
3. Update the `fdk-aac-rust-sys` version requirement in
   `crates/fdk-aac/Cargo.toml` to the same version.
4. Add a dated entry to `CHANGELOG.md`.
5. If needed, advance and validate `upstream/revision` separately before the
   release; do not combine an unverified upstream movement with a release.
6. Regenerate and commit `Cargo.lock`.
7. Run the release checks:

   ```sh
   ./tools/release-check.sh rust-vX.Y.Z
   ```

8. Commit the version and changelog changes, then create and push an annotated
   tag:

   ```sh
   git tag -a rust-vX.Y.Z -m "fdk-aac Rust port X.Y.Z"
   git push origin HEAD rust-vX.Y.Z
   ```

## Automated publication

Pushing the tag starts `.github/workflows/release.yml`. It:

1. confirms that the tag is annotated and matches `Cargo.toml`;
2. checks the changelog, pinned upstream revision, retained license text,
   prominent modification notice, and patent warning;
3. runs formatting, compile checks, the Pure Rust tests, and the full
   differential suite;
4. verifies and packages `fdk-aac-rust-sys`, while validating that its source
   archive contains exact copies of `NOTICE` and `README.md` and validating the
   main crate's source-file set;
5. publishes `fdk-aac-rust-sys` to crates.io;
6. waits until that exact version is visible in the registry;
7. fully packages and verifies `fdk-aac-rust` against the published sys crate,
   including exact `NOTICE` and `README.md` archive contents;
8. publishes `fdk-aac-rust`;
9. generates SHA-256 checksums;
10. creates a GitHub Release from the existing tag and attaches both `.crate`
   archives and the checksum file.

The workflow never creates or moves a Git tag. A failed validation therefore
cannot silently redefine a released version. crates.io releases are immutable;
if publication partially succeeds, fix the cause and rerun the same tag job so
the already-published crate is detected and skipped. Do not delete and recreate
the tag with different source.

## Versioning policy

- Patch: compatible bug fixes, additional tests, and parity corrections.
- Minor: backward-compatible Rust API additions or newly supported profiles.
- Major: intentional Rust API or behavior compatibility breaks.

The Rust version is independent of the pinned upstream C/C++ version. The Rust
crate starts at `0.1.0`; upstream identity is recorded separately by the full
SHA in `upstream/revision`. Record an upstream SHA movement in `CHANGELOG.md`,
but do not copy its `2.0.x` package version into this project unless that
independently matches the Rust API release decision.
