# Release process

This project follows the original fdk-aac practice of preparing a version and
changelog commit, validating it, and creating an annotated `rust-vX.Y.Z` Git
tag. The `rust-` prefix distinguishes this port from the inherited upstream
`v0.1.x` and `v2.0.x` tags.
Distribution is adapted for Rust: a manually dispatched GitHub Actions workflow
publishes both source crates to crates.io, verifies their exact contents and
embedded commit, creates an annotated version tag,
and creates a GitHub Release containing the complete repository source archive,
the two `.crate` source archives, CycloneDX SBOMs, and their SHA-256 checksums.
GitHub records build-provenance attestations for every checksummed artifact.
Compiled binary
artifacts are intentionally excluded; adding any requires satisfying the separate
[`DISTRIBUTION.md`](DISTRIBUTION.md) binary-distribution checklist.

## One-time repository setup

1. Allow GitHub Actions read/write access to repository contents so the
   workflow can create a GitHub Release. The workflow also declares the
   required `contents: write` permission.
2. Protect `main`, require changes to arrive through pull requests, and require
   the CI `rust` job before merging.
3. Create a crates.io API token restricted to `publish-new` and
   `publish-update` for `fdk-aac-rust-sys` and `fdk-aac-rust`, then store it as
   the repository secret `CARGO_REGISTRY_TOKEN`.
4. After the registry-first workflow has been merged and validated, enable
   GitHub Immutable Releases. The workflow supplies every asset in the initial
   `gh release create` call and never relies on later asset replacement.

The two crates always use the same version. The workflow publishes the sys
crate first, waits for that exact version to appear in the crates.io index, and
then publishes the safe crate. It skips an already published matching version,
so a failed workflow can be rerun safely from the same commit. Published crate
versions are permanent and cannot be overwritten.

## Preparing a release

1. Choose a semantic version.
2. Update `workspace.package.version` in the root `Cargo.toml`.
3. Update the `fdk-aac-rust-sys` version requirement in
   `crates/fdk-aac/Cargo.toml` to the same version.
4. Add a dated entry to `CHANGELOG.md`.
5. If needed, advance and validate `upstream/revision` separately before the
   release; do not combine an unverified upstream movement with a release.
6. Regenerate and commit `Cargo.lock`.
7. Run the release checks locally if desired:

   ```sh
   ./tools/release-check.sh rust-vX.Y.Z
   ```

8. Commit the version and changelog changes on a feature branch and merge them
   to `main` through a pull request.
9. In GitHub Actions, open the **Release** workflow, select **Run workflow** on
   `main`, and enter `X.Y.Z` without the `rust-v` prefix. The workflow validates
   the version, publishes and verifies both crates, creates the annotated
   `rust-vX.Y.Z` tag, and publishes the GitHub Release. Do not create or move the
   tag manually and do not publish either crate separately.

## Automated publication

Manually dispatching `.github/workflows/release.yml` from `main`:

1. confirms that the requested version matches `Cargo.toml` and the changelog;
2. checks the changelog, pinned upstream revision, retained license text,
   prominent modification notice, and patent warning;
3. runs formatting, compile checks, the Pure Rust tests, and the full
   differential suite;
4. verifies the `fdk-aac-rust-sys` source package and the main crate's source
   file set;
5. publishes `fdk-aac-rust-sys`, waits for index propagation, and then
   publishes `fdk-aac-rust`;
6. downloads both canonical crates and verifies byte identity and the embedded
   Git commit before creating any Git reference;
7. creates a complete repository source archive and verifies that all three
   source archives contain the required `NOTICE` and `README.md`;
8. generates CycloneDX 1.5 SBOMs for both crates and a verified SHA-256 manifest;
9. creates GitHub Artifact Attestations for every entry in that manifest;
10. creates an annotated `rust-vX.Y.Z` tag at the tested `main` commit;
11. atomically creates the GitHub Release with the repository archive, both
    `.crate` archives, both SBOMs, and the checksum file;
12. verifies the tag, Release, and crates again, then waits for successful
    documentation pages for both crates on docs.rs.

The workflow never moves an existing Git tag or overwrites a GitHub Release
asset. A rerun accepts a tag only when it is annotated and still points to the
exact tested commit. It accepts an existing Release only when its complete asset
set is byte-identical to the locally reproduced set. The crates.io publication
step checks for existing versions before uploading, but it cannot replace a
version whose source differs.

The sys crate's build script deliberately skips native compilation when
`DOCS_RS` is set because docs.rs builds offline and Rust API declarations do
not require a linked reference library. The safe crate is documented with
default features disabled, so its primary Pure Rust API remains available even
when the reference C/C++ source cannot be fetched.

## Versioning policy

- Patch: compatible bug fixes, additional tests, and parity corrections.
- Minor: backward-compatible Rust API additions or newly supported profiles.
- Major: intentional Rust API or behavior compatibility breaks.

The Rust version is independent of the pinned upstream C/C++ version. The Rust
crate starts at `0.1.0`; upstream identity is recorded separately by the full
SHA in `upstream/revision`. Record an upstream SHA movement in `CHANGELOG.md`,
but do not copy its `2.0.x` package version into this project unless that
independently matches the Rust API release decision.
