# Release process

This project follows the original fdk-aac practice of preparing a version and
changelog commit, validating it, and creating an annotated `rust-vX.Y.Z` Git
tag. The `rust-` prefix distinguishes this port from the inherited upstream
`v0.1.x` and `v2.0.x` tags.
Distribution is adapted for Rust: a manually dispatched GitHub Actions workflow
creates an annotated version tag and a GitHub Release containing the complete
repository source archive and its SHA-256 checksum. Compiled binary artifacts
are intentionally excluded; adding any requires satisfying the separate
[`DISTRIBUTION.md`](DISTRIBUTION.md) binary-distribution checklist.

## One-time repository setup

1. Allow GitHub Actions read/write access to repository contents so the
   workflow can create a GitHub Release. The workflow also declares the
   required `contents: write` permission.
2. Protect `main`, require changes to arrive through pull requests, and require
   the CI `rust` job before merging.

The two crates always use the same version. The GitHub Release is a complete
source release; publishing either crate to crates.io is a separate operation
and is not performed by this workflow.

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
   the version, creates the annotated `rust-vX.Y.Z` tag, and publishes the
   GitHub Release. Do not create or move the tag manually.

## Automated publication

Manually dispatching `.github/workflows/release.yml` from `main`:

1. confirms that the requested version matches `Cargo.toml` and the changelog;
2. checks the changelog, pinned upstream revision, retained license text,
   prominent modification notice, and patent warning;
3. runs formatting, compile checks, the Pure Rust tests, and the full
   differential suite;
4. verifies the `fdk-aac-rust-sys` source package and the main crate's source
   file set;
5. creates a complete repository source archive and verifies that its `NOTICE`
   and `README.md` exactly match the repository;
6. generates its SHA-256 checksum;
7. creates an annotated `rust-vX.Y.Z` tag at the tested `main` commit;
8. creates or updates the GitHub Release and attaches the archive and checksum.

The workflow never moves an existing Git tag. A rerun accepts it only when it is
annotated and still points to the exact tested commit. If publication fails
after tag creation, rerun the same version from that `main` commit; do not delete
and recreate the tag with different source.

## Versioning policy

- Patch: compatible bug fixes, additional tests, and parity corrections.
- Minor: backward-compatible Rust API additions or newly supported profiles.
- Major: intentional Rust API or behavior compatibility breaks.

The Rust version is independent of the pinned upstream C/C++ version. The Rust
crate starts at `0.1.0`; upstream identity is recorded separately by the full
SHA in `upstream/revision`. Record an upstream SHA movement in `CHANGELOG.md`,
but do not copy its `2.0.x` package version into this project unless that
independently matches the Rust API release decision.
